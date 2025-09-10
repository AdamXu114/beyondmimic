# 从 __future__ 导入 annotations 以支持类型提示中的前向引用（例如使用尚未定义的类名作为类型）
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# 导入 Isaac Lab 中用于数学变换的工具函数
# matrix_from_quat: 将四元数转换为旋转矩阵
# subtract_frame_transforms: 计算两个坐标系之间的相对变换（位置和旋转）
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

# 导入自定义的运动命令类
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

# TYPE_CHECKING 为 True 时才导入，避免运行时循环依赖
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取机器人锚点（robot anchor）在世界坐标系下的朝向（用旋转矩阵的前两行表示）。

    参数:
        env: 环境实例，包含命令管理器。
        command_name: 命令名称，用于从命令管理器中获取对应的 MotionCommand。

    返回:
        shape [num_envs, 6] 的张量，每个环境返回旋转矩阵的前两行展平后的结果（即6个元素），
        用于紧凑表示朝向，避免使用完整的9元素矩阵或非正交的四元数。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 将四元数转换为 3x3 旋转矩阵
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    # 取旋转矩阵的前两行（每行3个元素），然后展平为 [num_envs, 6]
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取机器人锚点在世界坐标系下的线速度。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, 3] 的张量，表示每个环境中锚点的 (vx, vy, vz)。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 提取线速度部分（前3个分量），并确保形状正确
    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    获取机器人锚点在世界坐标系下的角速度。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, 3] 的张量，表示每个环境中锚点的 (wx, wy, wz)。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 提取角速度部分（第3到第5个分量），并确保形状正确
    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算机器人各刚体（如躯干、四肢）相对于机器人锚点在锚点自身坐标系（body frame）中的位置。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, num_bodies * 3] 的张量，每个刚体的位置 (x, y, z) 被展平。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)  # 获取刚体数量

    # 将锚点的位置和朝向在 batch 维度上重复 num_bodies 次，以便与多个刚体并行计算
    anchor_pos_w_expanded = command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w_expanded = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    # 计算从锚点坐标系到各刚体坐标系的相对变换
    # 返回值: pos_b 是各刚体在锚点坐标系中的位置
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w_expanded,   # 锚点在世界系中的位置（已扩展）
        anchor_quat_w_expanded,  # 锚点在世界系中的朝向（已扩展）
        command.robot_body_pos_w, # 各刚体在世界系中的位置
        command.robot_body_quat_w, # 各刚体在世界系中的朝向（此处未使用）
    )

    # 展平为 [num_envs, num_bodies * 3]
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算机器人各刚体相对于机器人锚点在锚点自身坐标系中的朝向（用旋转矩阵的前两行表示）。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, num_bodies * 6] 的张量，每个刚体的朝向用旋转矩阵前两行的6个元素表示。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)

    # 扩展锚点变换以匹配刚体数量
    anchor_pos_w_expanded = command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w_expanded = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    # 计算相对变换，ori_b 是各刚体在锚点坐标系中的朝向（四元数）
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w_expanded,
        anchor_quat_w_expanded,
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    # 将四元数转换为旋转矩阵
    mat = matrix_from_quat(ori_b)
    # 取前两行并展平
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算“运动锚点”（motion anchor）相对于“机器人锚点”在机器人锚点坐标系中的位置。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, 3] 的张量，表示运动锚点在机器人锚点坐标系中的相对位置。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # 计算从机器人锚点到运动锚点的相对位置
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,  # 机器人锚点在世界系中的位置
        command.robot_anchor_quat_w, # 机器人锚点在世界系中的朝向
        command.anchor_pos_w,        # 运动锚点在世界系中的位置
        command.anchor_quat_w,       # 运动锚点在世界系中的朝向（此处未使用）
    )

    # 返回相对位置，形状为 [num_envs, 3]
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    计算“运动锚点”相对于“机器人锚点”在机器人锚点坐标系中的朝向（用旋转矩阵前两行表示）。

    参数:
        env: 环境实例。
        command_name: 命令名称。

    返回:
        shape [num_envs, 6] 的张量，表示运动锚点在机器人锚点坐标系中的相对朝向。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # 计算相对变换，ori 是运动锚点在机器人锚点坐标系中的朝向（四元数）
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    # 转换为旋转矩阵
    mat = matrix_from_quat(ori)
    # 取前两行并展平为6个元素
    return mat[..., :2].reshape(mat.shape[0], -1)