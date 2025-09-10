from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

# Isaac Lab 模块导入
from isaaclab.assets import Articulation  # 可动模型（如机器人）
from isaaclab.managers import CommandTerm, CommandTermCfg  # 命令项基类和配置
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # 可视化标记
from isaaclab.markers.config import FRAME_MARKER_CFG  # 预设的坐标系可视化配置
from isaaclab.utils import configclass  # 配置类装饰器
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 类型提示用，避免循环导入


class MotionLoader:
    """
    运动数据加载器，用于加载并管理从外部文件（.npz）中读取的运动捕捉数据。
    
    支持按时间步访问关节位置/速度、身体部位（body）的位姿和速度等。
    """

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        """
        初始化运动数据加载器。
        
        Args:
            motion_file: .npz 文件路径，包含预处理的运动数据。
            body_indexes: 要提取的身体部位索引列表。
            device: 数据加载到的设备（CPU/GPU）。
        """
        assert os.path.isfile(motion_file), f"无效的文件路径: {motion_file}"
        data = np.load(motion_file)

        self.fps = data["fps"]  # 帧率（每秒帧数）
        # 将数据转换为 PyTorch 张量，并移动到指定设备
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)  # 关节位置
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)  # 关节速度
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)  # 全局坐标系下身体位置
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # 全局四元数姿态
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)  # 线速度
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)  # 角速度

        self._body_indexes = body_indexes  # 要跟踪的身体部位索引
        self.time_step_total = self.joint_pos.shape[0]  # 总时间步数

    @property
    def body_pos_w(self) -> torch.Tensor:
        """获取指定身体部位在全局坐标下的位置。"""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """获取指定身体部位在全局坐标下的姿态（四元数）。"""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """获取指定身体部位在全局坐标下的线速度。"""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """获取指定身体部位在全局坐标下的角速度。"""
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    """
    动作复现命令生成器，用于在强化学习环境中复现参考运动数据（motion imitation）。

    该命令项会：
    - 从 .npz 文件中加载参考运动轨迹
    - 在训练中动态采样时间步（支持自适应采样）
    - 提供观测（command），包含目标关节位置/速度
    - 支持随机扰动（位姿、速度、关节）
    - 计算跟踪误差并记录到 metrics
    - 支持可视化当前与目标姿态
    """
    cfg: MotionCommandCfg  # 类型提示：配置对象

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 获取机器人对象
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # 获取锚点身体部位（anchor body）在机器人和运动数据中的索引
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        
        # 获取所有目标身体部位在机器人中的索引（用于状态观测）
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device
        )

        # 加载运动数据
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        
        # 当前时间步计数器（每个环境独立）
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 存储相对位姿（相对于锚点 body 的变换）
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0  # 初始四元数为 (1,0,0,0)，表示无旋转

        # 自适应采样相关参数
        # 将整个运动轨迹划分为若干个 bin，用于统计失败分布
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)  # 历史失败次数
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)  # 当前失败计数

        # 构建指数衰减核，用于平滑失败分布（鼓励探索附近时间步）
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()  # 归一化

        # 初始化评估指标（metrics）
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """
        返回当前命令（作为观测输入给策略网络）。
        TODO: 可能需要重新评估是否为最佳观测形式。
        """
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    # --- 从 motion 数据中获取目标状态 ---
    @property
    def joint_pos(self) -> torch.Tensor:
        """当前时间步的目标关节位置"""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """当前时间步的目标关节速度"""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """全局坐标系下身体部位的目标位置（加上环境原点偏移）"""
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """全局坐标系下身体部位的目标姿态"""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """全局坐标系下身体部位的目标线速度"""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """全局坐标系下身体部位的目标角速度"""
        return self.motion.body_ang_vel_w[self.time_steps]

    # --- 锚点 body 的目标状态 ---
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """锚点身体部位的目标位置"""
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """锚点身体部位的目标姿态"""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """锚点身体部位的目标线速度"""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """锚点身体部位的目标角速度"""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    # --- 机器人当前状态 ---
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        """更新各项误差指标，用于监控训练过程"""
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """
        自适应时间步采样策略：根据失败分布调整采样概率，鼓励探索易失败区域。

        Args:
            env_ids: 需要重新采样的环境 ID 列表
        """
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 计算当前时间步所属的 bin
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
                0,
                self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            # 统计当前失败的 bin 分布
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # 构建采样概率分布
        # 并非与论文一样，这个均匀采样比率加的位置不对
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # sampling_probabilities = self.bin_failed_count 

        # 使用卷积核平滑分布（引入邻近 bin 的影响）
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        sampling_probabilities = (1 - self.cfg.adaptive_uniform_ratio) * sampling_probabilities \
                                + self.cfg.adaptive_uniform_ratio / float(self.bin_count)

        # 多项式采样
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # 第一个包含小数扰动，第二个是整数步
        # 应保留第一个以实现更平滑的插值
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        self.time_steps[env_ids] = (sampled_bins / self.bin_count * (self.motion.time_step_total - 1)).long()

        # 记录采样分布的统计信息
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)  # 归一化熵
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """
        为指定环境重新采样命令（包括时间步和随机扰动）

        Args:
            env_ids: 需要重置的环境 ID
        """
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        # 获取当前目标状态
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # 对根节点位姿添加随机扰动
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # 对根节点速度添加扰动
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # 对关节位置添加噪声并裁剪到软限制范围内
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        # 写入仿真状态
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """
        每一步更新命令状态：
        - 时间步递增
        - 超出范围时重新采样
        - 计算相对位姿
        - 更新失败统计
        """
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        # 计算锚点变换（用于对齐运动参考）
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]  # Z 轴对齐
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))  # Yaw 对齐

        # 计算相对目标位姿，跟踪目标，世界坐标系下
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # 更新失败 bin 的加权移动平均
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + 
            (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化"""
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                # 创建可视化器
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            # 显示所有可视化
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        """可视化回调函数，每帧调用"""
        if not self.robot.is_initialized:
            return

        # 可视化锚点
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # 可视化各个 body 部位
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """MotionCommand 的配置类"""
    class_type: type = MotionCommand  # 关联的实现类

    asset_name: str = MISSING  # 机器人在场景中的名称
    motion_file: str = MISSING  # 运动数据文件路径
    anchor_body_name: str = MISSING  # 用于对齐的锚点身体部位名称（如 'pelvis'）
    body_names: list[str] = MISSING  # 需要跟踪的身体部位名称列表

    # 添加的随机扰动范围
    pose_range: dict[str, tuple[float, float]] = {}  # 位姿扰动范围 (x, y, z, roll, pitch, yaw)
    velocity_range: dict[str, tuple[float, float]] = {}  # 速度扰动范围
    joint_position_range: tuple[float, float] = (-0.52, 0.52)  # 关节位置噪声范围

    # 自适应采样参数
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8  # 指数核衰减率
    adaptive_uniform_ratio: float = 0.1  # 均匀采样⽐率
    adaptive_alpha: float = 0.001  # 移动平均系数

    # 可视化配置
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)  # 锚点可视化尺寸

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # 身体部位可视化尺寸