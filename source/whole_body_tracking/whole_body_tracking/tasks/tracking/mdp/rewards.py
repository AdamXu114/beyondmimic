# 支持未来语法中的类型注解（如使用尚未定义的类作为类型提示）
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# 导入 Isaac Lab 的相关模块
from isaaclab.managers import SceneEntityCfg          # 用于配置场景中的实体（如传感器）
from isaaclab.sensors import ContactSensor            # 接触传感器类
from isaaclab.utils.math import quat_error_magnitude  # 计算两个四元数之间的误差大小
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand  # 自定义运动命令

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 仅在类型检查时导入，避免循环依赖


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """
    辅助函数：根据给定的刚体名称列表，返回其在命令配置中的索引。

    参数:
        command: MotionCommand 实例，包含 cfg.body_names 刚体名称列表。
        body_names: 要查询的刚体名称列表。若为 None，则返回所有刚体的索引。

    返回:
        包含符合条件的刚体在 body_names 中索引的列表。
    """
    return [
        i for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    奖励函数：基于机器人锚点与目标“运动锚点”在世界坐标系下的位置误差，返回指数衰减形式的奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称，用于获取 MotionCommand。
        std: 标准差，控制误差对奖励的影响程度（越大越宽容）。

    返回:
        shape [num_envs] 的张量，值在 (0,1] 之间：
        - 误差越小，奖励越接近 1；
        - 误差越大，奖励趋近于 0。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算位置误差的平方和（L2 距离的平方）
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    # 指数衰减：exp(-error / std^2)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    奖励函数：基于机器人锚点与目标运动锚点在世界坐标系下的朝向误差，返回指数奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差，控制朝向误差的惩罚强度。

    返回:
        shape [num_envs] 的张量，值在 (0,1] 之间。
        使用 quat_error_magnitude 计算四元数之间的最小角度误差（弧度），然后平方并指数衰减。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算四元数误差的幅度（弧度），然后平方
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    奖励函数：基于机器人各刚体相对于锚点的位置与目标相对位置的误差，返回平均指数奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选，指定参与计算的刚体名称列表。若为 None，则使用所有刚体。

    返回:
        shape [num_envs] 的张量。
        对每个刚体计算位置误差平方，取所有刚体的平均误差后再进行指数衰减。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)

    # 获取目标相对位置与当前机器人各刚体位置的误差
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes] -  # 目标：各刚体相对于锚点的位置
            command.robot_body_pos_w[:, body_indexes]       # 实际：各刚体在世界系中的位置
        ),
        dim=-1  # 对每个刚体的3个坐标求平方和
    )
    # 对所有刚体的误差取平均，然后指数衰减
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    奖励函数：基于机器人各刚体相对于锚点的朝向误差，返回平均指数奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选刚体名称列表。

    返回:
        shape [num_envs] 的张量。
        对每个刚体计算四元数误差的平方，取平均后指数衰减。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)

    # 计算每个刚体的朝向误差（弧度）并平方
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],   # 目标相对朝向
            command.robot_body_quat_w[:, body_indexes]       # 当前朝向
        )
        ** 2
    )
    # 对所有刚体取平均误差，然后指数衰减
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    奖励函数：基于机器人各刚体在世界坐标系下的线速度与目标线速度的误差，返回指数奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选刚体列表。

    返回:
        shape [num_envs] 的张量。
        对每个刚体计算速度误差平方和，取平均后指数衰减。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)

    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes] -      # 目标线速度
            command.robot_body_lin_vel_w[:, body_indexes]   # 实际线速度
        ),
        dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """
    奖励函数：基于机器人各刚体在世界坐标系下的角速度误差，返回指数奖励。

    参数:
        env: 环境实例。
        command_name: 命令名称。
        std: 标准差。
        body_names: 可选刚体列表。

    返回:
        shape [num_envs] 的张量。
        类似线速度，但针对角速度 (wx, wy, wz)。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)

    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes] -      # 目标角速度
            command.robot_body_ang_vel_w[:, body_indexes]   # 实际角速度
        ),
        dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    奖励函数：鼓励足部在适当时机接触地面，用于步态控制等任务。

    参数:
        env: 环境实例。
        sensor_cfg: 接触传感器的配置，指定传感器名称和关联的刚体 ID（如脚部）。
        threshold: 时间阈值（秒），用于判断“刚离开地面”的状态。

    返回:
        shape [num_envs] 的张量，表示每个环境中满足条件的足部数量（可用于奖励）。

    逻辑说明：
        - first_air[i, j] == 1 表示第 i 个环境中的第 j 个刚体“刚刚”离开地面（即前一时刻接触，当前时刻未接触）。
        - last_contact_time[i, j] 表示第 i 个环境中第 j 个刚体距离上次接触的时间。
        - 条件 (last_contact_time < threshold) 表示该刚体最近才接触过地面。
        - 因此，(last_contact_time < threshold) * first_air 表示“短暂接触后立即离地”的足部。
        - 对所有足部求和，鼓励自然的步态周期（短暂触地、腾空）。
    """
    # 获取接触传感器实例
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 计算每个刚体是否“刚刚”离开地面（从接触变为非接触）
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]

    # 获取每个刚体距离上次接触的时间
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    # 若刚体在 threshold 时间内接触过，且现在“刚刚”离地，则计入奖励
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)

    return reward