import sys
from pathlib import Path

import torch
sys.path.append(str(Path(__file__).parent.parent.absolute()))
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

import time
import mujoco_viewer
import mujoco
import numpy as np
import yaml
import os
import onnx
import onnxruntime as ort

YELLOW = '\033[33m'
RESET = '\033[0m'
RED = '\033[31m'

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "g1.yaml")

    print(f"{YELLOW}[INFO] 当前工作目录: {current_dir}")
    print(f"[INFO] 加载配置文件: {mujoco_yaml_path}")

    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = os.path.join(PROJECT_ROOT, config["policy_path"])
        print("[INFO] Loading policy from: ", policy_path)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        print("[INFO] Loading mujoco xml from: ", xml_path)
        motion_path = os.path.join(PROJECT_ROOT, config["motion_path"])
        print("[INFO] Loading motion from: ", motion_path)

        simulation_dt = config["simulation_dt"]
        simulation_duration = config["simulation_duration"]
        control_decimation = config["control_decimation"]
        print(f"[INFO] 仿真时间步长: {simulation_dt}, 控制降采样: {control_decimation}{RESET}")
        kps_lab = np.array(config["kp_lab"], dtype=np.float32)
        kds_lab = np.array(config["kd_lab"], dtype=np.float32)
        default_angles_lab = np.array(config["default_angles_lab"], dtype=np.float32)

        mj2lab = np.array(config["mj2lab"], dtype=np.int32)
        # default_angles_mj = np.array(config["default_angles_mj"], dtype=np.float32)
        tau_limit = np.array(config["tau_limit"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        action_scale_lab = np.array(config["action_scale_lab"], dtype=np.float32)
        # motion_length = config["motion_length"]

        motion = np.load(motion_path)
        motion_joint_pos = np.array(motion["joint_pos"], dtype=np.float32)
        motion_joint_vel = np.array(motion["joint_vel"], dtype=np.float32)
        motion_body_quat_w = np.array(motion["body_quat_w"], dtype=np.float32)

        print(f"{YELLOW}[INFO] 运动数据加载成功 - 关节位置形状: {motion_joint_pos.shape}, "
              f"关节速度形状: {motion_joint_vel.shape}, "
              f"身体四元数形状: {motion_body_quat_w.shape}")

    # ONNX策略模型加载和信息打印
    print("\n[INFO] 加载ONNX策略模型...")
    policy = ort.InferenceSession(policy_path)
    policy_input = policy.get_inputs()
    input_name = []
    print(f"[INFO] 模型输入信息:{RESET}")
    for i, inpt in enumerate(policy_input):
        input_name.append(inpt.name)
        print(f"  input {i}: {inpt.name}, shape: {inpt.shape}, type: {inpt.type}")

    policy_output = policy.get_outputs()
    output_name = []
    print(f"{YELLOW}[INFO] 模型输出信息:{RESET}")
    for i, outpt in enumerate(policy_output):
        output_name.append(outpt.name)
        print(f"  output {i}: {outpt.name}, shape: {outpt.shape}, type: {outpt.type}")



    # MuJoCo模型初始化
    print(f"\r{YELLOW}[INFO] 初始化MuJoCo模型...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu

    print(f"[INFO] MuJoCo模型初始化完成 - 关节数量: {num_joints}, "
          f"每步持续时间: {mj_per_step_duration}")

    target_dof_pos_mj = np.zeros(num_joints, dtype=np.float32)

    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    qj = np.zeros(num_joints, dtype=np.float32)
    dqj = np.zeros(num_joints, dtype=np.float32)
    action = np.zeros(num_actions)

    kps[mj2lab] = kps_lab
    kds[mj2lab] = kds_lab
    default_angles_mj = np.zeros_like(default_angles_lab)
    default_angles_mj[mj2lab] = default_angles_lab

    print(f"[INFO] 输出初始化参数，以mujoco关节顺序排列:{RESET}")
    print(f"[INFO] 动作维度: {num_actions}, 观测维度: {num_obs}")
    print(f"[INFO] Kp: {kps}, \r[INFO] Kd: {kds},\r[INFO] 默认角度: {default_angles_mj},\r[INFO] 力矩限制: {tau_limit}")

    d.qpos[3:7] = [0,0,0,1]
    # reset to initial pose
    sim_counter = 0
    counter_step = 0
    print("\n[INFO] 按 Enter 键开始仿真...")
    input("Press Enter to continue...")
    print("[INFO] 重置到初始姿态...")

    observation = {}
    observation[input_name[0]] = np.zeros((1, num_obs), dtype=np.float32)
    observation[input_name[1]] = np.zeros((1, 1), dtype=np.float32)

    print("[INFO] 运行初始策略推理...")
    outputs_result = policy.run(None, observation)
    # 处理多个输出
    # action, ref_joint_pos, ref_joint_vel, _, ref_body_quat_w, _, _ = outputs_result
    action, _, _, _, _, _, _ = outputs_result
    # 修改你的代码如下：
    ref_joint_pos = np.expand_dims(motion_joint_pos[counter_step, :], axis=0)
    ref_joint_vel = np.expand_dims(motion_joint_vel[counter_step, :], axis=0)
    ref_body_quat_w = np.expand_dims(motion_body_quat_w[counter_step, :, :], axis=0)
    print(f"[DEBUG] 参考动作形状: {ref_joint_pos.shape}, 参考身体四元数形状: {ref_body_quat_w.shape}")
    print(f"[INFO] 初始动作生成: {action.shape}")

    qj_obs = np.zeros(num_actions, dtype=np.float32)
    dqj_obs = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs)

    # 初始化查看器
    print("[INFO] 初始化MuJoCo查看器...")
    viewer = mujoco_viewer.MujocoViewer(m, d)
    flag = False

    print(f"[INFO] 开始主仿真循环...{RESET}")
    if True:
        start = time.time()
        last_log_time = time.time()

        while time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos_mj, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            # tau = np.clip(tau, -200, 200)
            # d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            sim_counter += 1
            if sim_counter % control_decimation == 0:
                print(f"[DEBUG] 控制周期: {counter_step}, 当前动作范数: {np.linalg.norm(action):.4f}")

                qj = d.qpos[7:][mj2lab]
                qj = (qj - default_angles_lab)
                dqj = d.qvel[6:][mj2lab]
                quat = d.qpos[3:7]

                omega = d.qvel[3:6]
                gravity_orientation = get_gravity_orientation(quat)

                pelvis_quat_w = quat.copy()

                abdomen_yaw = qj[2]
                abdomen_roll = qj[5]
                abdomen_pitch = qj[8]

                quat_yaw = euler_single_axis_to_quat(abdomen_yaw, 'z', degrees=False)
                quat_roll = euler_single_axis_to_quat(abdomen_roll, 'x', degrees=False)
                quat_pitch = euler_single_axis_to_quat(abdomen_pitch, 'y', degrees=False)
                temp1 = quat_mul(quat_roll, quat_pitch)
                temp2 = quat_mul(quat_yaw, temp1)
                robot_quat = quat_mul(pelvis_quat_w, temp2)
                # ref_anchor_ori_w = ref_body_quat_w[:, 7].squeeze(0)
                ref_anchor_ori_w = ref_body_quat_w[:, 9].squeeze(0)

                if counter_step < 1:
                    init_to_anchor = matrix_from_quat(yaw_quat(ref_anchor_ori_w))
                    world_to_anchor = matrix_from_quat(yaw_quat(robot_quat))
                    init_to_world = world_to_anchor @ init_to_anchor.T
                    print("[INFO] 初始化世界变换矩阵计算完成")
                    counter_step += 1
                    continue

                motion_anchor_ori_b = matrix_from_quat(robot_quat).T @ init_to_world @ matrix_from_quat(
                    ref_anchor_ori_w)

                mimic_obs_buf = np.concatenate((ref_joint_pos.squeeze(0),
                                                ref_joint_vel.squeeze(0),
                                                motion_anchor_ori_b[:, :2].reshape(-1),
                                                omega,
                                                qj,
                                                dqj,
                                                action.squeeze(0)),
                                               axis=-1, dtype=np.float32)

                mimic_obs_tensor = torch.from_numpy(mimic_obs_buf).unsqueeze(0).cpu().numpy()
                observation = {}

                observation[input_name[0]] = mimic_obs_tensor
                observation[input_name[1]] = np.array([[counter_step]], dtype=np.float32)
                outputs_result = policy.run(None, observation)

                # action, ref_joint_pos, ref_joint_vel, _, ref_body_quat_w, _, _ = outputs_result
                action, _, _, _, _, _, _ = outputs_result
                ref_joint_pos = np.expand_dims(motion_joint_pos[counter_step, :], axis=0)
                ref_joint_vel = np.expand_dims(motion_joint_vel[counter_step, :], axis=0)
                ref_body_quat_w = np.expand_dims(motion_body_quat_w[counter_step, :, :], axis=0)
                print(f"[DEBUG] 参考动作形状: {ref_joint_pos.shape}, 参考身体四元数形状: {ref_body_quat_w.shape}")
                print(ref_joint_pos,"wwwwwwwwwwwww")
                # print("Action:", action)
                target_dof_pos_lab = action * action_scale_lab + default_angles_lab
                target_dof_pos_mj[mj2lab] = target_dof_pos_lab.squeeze(0)
                print("target_dof_pos_mj:", target_dof_pos_mj)
                # counter_step += 1
                counter_step = 1
                if counter_step >= 600:
                    counter_step = 600

            viewer.render()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("[INFO] 仿真结束")
    viewer.close()