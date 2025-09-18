import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

# from common.path_config import PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time
import os
import yaml
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
import threading
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation_real, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from whole_body_tracking.deploy.config_bydmimic import Config

import onnx
import onnxruntime as ort

joint_seq =['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 
 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 
 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 
 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 
 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 
 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
joint_xml = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint",  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",  "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",     "left_shoulder_roll_joint",     "left_shoulder_yaw_joint",
    "left_elbow_joint",     "left_wrist_roll_joint",    "left_wrist_pitch_joint",    "left_wrist_yaw_joint",    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",    "right_shoulder_yaw_joint",    "right_elbow_joint",    "right_wrist_roll_joint",    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"]

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def euler_single_axis_to_quat(angle, axis, degrees=False):
    """
    将单个欧拉角转换为四元数
    
    参数:
        angle: 旋转角度
        axis: 旋转轴，可以是 'x', 'y', 'z' 或者单位向量 [x, y, z]
        degrees: 如果为True，输入角度为度数；如果为False，输入角度为弧度
    
    返回:
        四元数 (w, x, y, z)
    """
    # 转换角度为弧度
    if degrees:
        angle = np.radians(angle)
    
    # 计算半角
    half_angle = angle * 0.5
    cos_half = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    
    # 根据旋转轴确定四元数分量
    if isinstance(axis, str):
        if axis.lower() == 'x':
            return np.array([cos_half, sin_half, 0.0, 0.0])
        elif axis.lower() == 'y':
            return np.array([cos_half, 0.0, sin_half, 0.0])
        elif axis.lower() == 'z':
            return np.array([cos_half, 0.0, 0.0, sin_half])
        else:
            raise ValueError("axis must be 'x', 'y', 'z' or a 3D unit vector")
    else:
        # 假设axis是一个3D向量 [x, y, z]
        axis = np.array(axis, dtype=np.float32)
        # 归一化轴向量
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("axis vector cannot be zero")
        axis = axis / axis_norm
        
        # 计算四元数分量
        w = cos_half
        x = sin_half * axis[0]
        y = sin_half * axis[1]
        z = sin_half * axis[2]
        
        return np.array([w, x, y, z])
    
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([w, x, y, z])
    
def matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()
        
        # Initialize the policy network
        self.policy =  ort.InferenceSession(config.policy_path)# torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.timestep = 0
        self.motion = np.load("/home/deepcyber-mk/Documents/unitree_rl_gym/deploy/deploy_real/bydmimic/dance_zui.npz")
        self.motionpos = self.motion['body_pos_w']
        self.motionquat = self.motion['body_quat_w']
        self.motioninputpos = self.motion['joint_pos']
        self.motioninputvel = self.motion['joint_vel']
        self.action_buffer = np.zeros((self.config.num_actions,), dtype=np.float32)
        self.dof_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                        12, 13, 14, 
                        15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 26, 27, 28]
        
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        
        
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.stiffness
        kds = self.config.damping
        # default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        default_pos = self.config.default_angles.copy()
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size): 
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]*5
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i+12]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]*3
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            quat = self.low_state.imu_state.quaternion
            print("quat",quat)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.dof_idx)):
            self.qj[i] = self.low_state.motor_state[self.dof_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.dof_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
        if self.config.imu_type == "pelvis":
            pelvis_quat_w = quat.copy()
            # pelvis imu data needs to be transformed to the torso frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            # waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            waist_roll = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[1]].q
            waist_pitch= self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[2]].q
            quat_yaw = euler_single_axis_to_quat(waist_yaw, 'z', degrees=False)
            quat_roll = euler_single_axis_to_quat(waist_roll, 'x', degrees=False)
            quat_pitch = euler_single_axis_to_quat(waist_pitch, 'y', degrees=False)
            temp1 = quat_mul(quat_roll, quat_pitch)
            temp2 = quat_mul(quat_yaw, temp1)
            robot_quat = quat_mul(pelvis_quat_w, temp2)
        
        print("robot_quat",robot_quat)
        print("quat",quat)
        qj = self.qj.copy()
        qj_obs = qj[self.config.real2lab]   # isaac lab idx
        qj_obs = (qj_obs - self.config.default_angles_seq)
        dqj = self.dqj.copy()
        dqj_obs = dqj[self.config.real2lab]     # isaac lab idx

        motioninput = np.concatenate((self.motioninputpos[self.timestep,:],self.motioninputvel[self.timestep,:]), axis=0)
        ref_anchor_ori_w = self.motionquat[self.timestep,9,:]
        if self.timestep < 1:
            init_to_anchor = matrix_from_quat(yaw_quat(ref_anchor_ori_w))
            world_to_anchor = matrix_from_quat(yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T

        motion_anchor_ori_b = matrix_from_quat(robot_quat).T @ self.init_to_world @ matrix_from_quat(
            ref_anchor_ori_w)
        
        offset = 0
        self.obs[offset:offset+58] = motioninput
        offset += 58
        self.obs[offset:offset+6] = motion_anchor_ori_b[:, :2].reshape(-1)
        offset += 6
        self.obs[offset:offset+3] = ang_vel
        offset += 3
        self.obs[offset:offset+29] = qj_obs
        offset += 29
        self.obs[offset:offset+29] = dqj_obs
        offset += 29
        self.obs[offset:offset+29] = self.action_buffer

        # mimic_obs_buf = np.concatenate((ref_joint_pos.squeeze(0),
        #                                 ref_joint_vel.squeeze(0),
        #                                 motion_anchor_ori_b[:, :2].reshape(-1),
        #                                 omega,
        #                                 qj,
        #                                 dqj,
        #                                 action.squeeze(0)),
        #                                axis=-1, dtype=np.float32)
                
        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        action = self.policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([self.timestep], dtype=np.float32).reshape(1,1)})[0]
        action = np.asarray(action).reshape(-1)     # isaac lab idx
        
        self.action = action.copy()
        self.action_buffer = action.copy()
        # transform action to target_dof_pos
        target_dof_pos_lab = self.config.default_angles_seq + self.action * self.config.action_scale_seq
        target_dof_pos_lab = target_dof_pos_lab.reshape(-1)
        target_dof_pos = np.zeros_like(target_dof_pos_lab)
        target_dof_pos[self.config.real2lab] = target_dof_pos_lab   # xml idx
        # target_dof_pos = target_dof_pos.reshape(-1,)
        # target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
        self.timestep += 1
        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i+12]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)
        
            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_for_bydmimic.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
# python  deploy_real4bydmimic.py enp4s0  g1_for_bydmimic.yaml