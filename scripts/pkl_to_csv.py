import pickle
import csv
import os
import numpy as np

# G1 的关节顺序（完整 30 DOF）
JOINT_ORDER = [
    'root_x', 'root_y', 'root_z', 'root_qx', 'root_qy', 'root_qz', 'root_qw',
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

def load_g1_motion(pkl_path):
    # 文件存在检查
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"找不到文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查必需字段
    required_keys = ['joint_pos', 'root_pos', 'root_quat']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"缺少必要字段: '{key}'")

    # 检查维度
    num_frames = data['joint_pos'].shape[0]
    if data['root_pos'].shape[0] != num_frames or data['root_quat'].shape[0] != num_frames:
        raise ValueError("关节位置、根位置和根四元数的帧数不一致")
    
    return data

def motion_to_csv(data, output_csv):
    num_frames = data['joint_pos'].shape[0]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(JOINT_ORDER)
        for i in range(num_frames):
            row = []
            # 根位置 (XYZ) + 四元数 (QXQYQZQW)
            row.extend(data['root_pos'][i].tolist())
            row.extend(data['root_quat'][i].tolist())
            # 按顺序加入关节角
            row.extend(data['joint_pos'][i].tolist())
            writer.writerow(row)

if __name__ == '__main__':
    pkl = 'humanoidverse/data/motions/g1_29dof_anneal_23dof/.../your_motion.pkl'
    csv_out = 'g1_motion.csv'
    
    try:
        data = load_g1_motion(pkl)
        motion_to_csv(data, csv_out)
        print(f"导出完成: {csv_out}")
    except Exception as e:
        print(f"处理失败: {e}")
