# 导入 future 特性，启用类型注解的延迟求值（延迟解析类型提示）
# 这允许在类定义中使用尚未定义的类名作为类型提示（如 'Self' 或向前引用）
from __future__ import annotations

# 从 dataclasses 模块导入 MISSING，用于标记配置类中必须由子类重写的字段
from dataclasses import MISSING

# 导入 Isaac Lab 的仿真工具模块
import isaaclab.sim as sim_utils

# 导入 Isaac Lab 的核心组件配置类
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # 资产配置
from isaaclab.envs import ManagerBasedRLEnvCfg  # 基于管理器的强化学习环境配置基类
from isaaclab.managers import EventTermCfg as EventTerm  # 事件项配置
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 观察组配置
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 观察项配置
from isaaclab.managers import RewardTermCfg as RewTerm  # 奖励项配置
from isaaclab.managers import SceneEntityCfg  # 场景实体配置（用于指定机器人、传感器等）
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 终止条件配置
from isaaclab.scene import InteractiveSceneCfg  # 可交互场景配置
from isaaclab.sensors import ContactSensorCfg  # 接触传感器配置
from isaaclab.terrains import TerrainImporterCfg  # 地形导入器配置

##
# 预定义配置与工具导入
##

# 导入配置类装饰器，用于创建可序列化的配置类（类似 dataclass，但支持更多 Isaac Lab 特性）
from isaaclab.utils import configclass

# 导入噪声配置类，用于为观察值添加噪声（提升策略鲁棒性）
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入自定义的 MDP（马尔可夫决策过程）函数模块，包含命令、观察、奖励等逻辑
import whole_body_tracking.tasks.tracking.mdp as mdp

##
# 场景定义
##

# 定义全局变量：期望的速度命令范围（线速度和角速度）
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),     # 前后方向速度范围 (m/s)
    "y": (-0.5, 0.5),     # 左右方向速度范围 (m/s)
    "z": (-0.2, 0.2),     # 上下方向速度范围 (m/s)
    "roll": (-0.52, 0.52), # 横滚角速度范围 (rad/s)
    "pitch": (-0.52, 0.52),# 俯仰角速度范围 (rad/s)
    "yaw": (-0.78, 0.78),  # 偏航角速度范围 (rad/s)
}

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """配置一个包含地形和四足机器人的交互式场景。"""

    # 地形配置：使用平面地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 在仿真场景中的路径
        terrain_type="plane",       # 地形类型：平面
        collision_group=-1,         # 碰撞组，-1 表示默认组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦力组合模式
            restitution_combine_mode="multiply", # 恢复系数组合模式
            static_friction=1.0,    # 静摩擦系数
            dynamic_friction=1.0,   # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",  # 材质路径
            project_uvw=True,       # 是否投影 UVW 坐标
        ),
    )

    # 机器人配置：由子类具体指定（使用 MISSING 表示必须重写）
    robot: ArticulationCfg = MISSING

    # 环境光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),  # 平行光
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),  # 天空光（穹顶光）
    )

    # 接触传感器：检测机器人身体各部分的接触力
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 匹配所有机器人身体部件
        history_length=3,          # 保留最近 3 帧的接触数据
        track_air_time=True,       # 跟踪部件离地时间
        force_threshold=10.0,      # 触发力阈值（超过此值视为有效接触）
        debug_vis=True             # 是否在仿真中可视化接触点
    )


##
# MDP 设置
##

@configclass
class CommandsCfg:
    """MDP 的命令配置：定义机器人需要跟踪的目标（运动锚点）。"""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",                      # 关联的机器人资产
        resampling_time_range=(1.0e9, 1.0e9),   # 命令重采样时间范围（极大值表示几乎不重采样）
        debug_vis=True,                          # 是否可视化目标姿态
        pose_range={                             # 目标位姿的随机范围（相对当前位姿的小扰动）
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,           # 目标速度范围
        joint_position_range=(-0.1, 0.1),        # 关节位置目标范围（用于初始化）
    )

@configclass
class ActionsCfg:
    """MDP 的动作配置：定义智能体可以执行的动作。"""

    # 动作类型：关节位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",        # 作用对象
        joint_names=[".*"],        # 控制所有关节
        use_default_offset=True    # 使用默认关节位置作为目标偏移基础
    )

@configclass
class ObservationsCfg:
    """MDP 的观察配置：定义智能体可以感知的环境信息。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络使用的观察组（包含噪声，更贴近真实环境）。"""

        # 观察项定义：
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # 当前运动命令（位置、速度等）

        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        # 运动锚点在机器人基座坐标系下的位置 + 均匀噪声

        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # 运动锚点在机器人基座坐标系下的姿态（四元数） + 噪声

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # 基座线速度 + 噪声

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 基座角速度 + 噪声

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 相对默认位置的关节角度 + 小噪声

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # 关节速度 + 噪声

        actions = ObsTerm(func=mdp.last_action)
        # 上一时刻的动作（用于动作历史）

        def __post_init__(self):
            self.enable_corruption = True    # 启用观察值噪声（corruption）
            self.concatenate_terms = True    # 将所有观察项拼接成一个向量

    @configclass
    class PrivilegedCfg(ObsGroup):
        """批评家（Critic）或教师网络使用的观察组（无噪声，信息更完整）。"""
        # 与 PolicyCfg 类似，但不加噪声，用于提供更精确的状态信息
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        # 机器人各身体部件在基座坐标系下的位置
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        # 机器人各身体部件在基座坐标系下的姿态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # 定义两个观察组
    policy: PolicyCfg = PolicyCfg()      # 策略使用的观察
    critic: PrivilegedCfg = PrivilegedCfg()  # 批评家使用的观察（特权信息）

@configclass
class EventCfg:
    """环境事件配置：在特定时间点触发的随机化或初始化操作。"""

    # 启动时事件
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # 在环境启动时执行一次
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 作用于机器人所有刚体
            "static_friction_range": (0.3, 1.6),   # 随机化静摩擦系数
            "dynamic_friction_range": (0.3, 1.2),  # 随机化动摩擦系数
            "restitution_range": (0.0, 0.5),       # 随机化恢复系数
            "num_buckets": 64,                     # 用于离散化采样的桶数
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),  # 在默认关节位置上添加小随机偏移
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 仅作用于躯干
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心偏移范围
        },
    )

    # 间隔事件（在仿真过程中周期性触发）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",  # 周期性触发
        interval_range_s=(1.0, 3.0),  # 触发间隔在 1 到 3 秒之间随机
        params={"velocity_range": VELOCITY_RANGE},  # 施加的随机速度
    )

@configclass
class RewardsCfg:
    """奖励函数配置：定义智能体获得奖励的各项。"""

    # 各项奖励及其权重
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},  # 基于指数函数的锚点位置误差奖励
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},  # 锚点姿态误差奖励
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},  # 身体部件相对锚点位置误差奖励
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},  # 身体部件姿态误差奖励
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},  # 身体线速度跟踪奖励
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14}, # 身体角速度跟踪奖励
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)  # 惩罚动作变化率（平滑性）
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)  # 惩罚关节速度过大（平滑性）
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)  # 惩罚关节加速度过大（平滑性）
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},  # 惩罚关节接近极限
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
                # 正则表达式排除脚踝和手腕，检测其他部位的非预期接触
            ),
            "threshold": 1.0,
        },
    )

@configclass
class TerminationsCfg:
    """终止条件配置：定义 episode 结束的条件。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时终止
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},  # 锚点高度异常（如摔倒）
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        # 基座姿态异常（如翻倒）
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            # 末端执行器（脚、手）位置异常（如悬空过高）
        },
    )

@configclass
class CurriculumCfg:
    """课程学习配置：用于逐步增加任务难度（当前为空，未启用）。"""
    pass

##
# 环境配置
##

@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """整体的全身运动跟踪环境配置。"""

    # 场景配置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # 创建 4096 个并行环境，每个环境间距 2.5 米

    # MDP 基本组件
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP 高级组件
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """配置初始化后的设置。"""
        # 通用设置
        self.decimation = 4  # 控制频率与仿真频率的比率（控制 1 步 = 仿真 4 步）
        self.episode_length_s = 10.0  # 每个 episode 的最大时长（秒）

        # 仿真设置
        self.sim.dt = 0.005  # 仿真时间步长（秒）
        self.sim.render_interval = self.decimation  # 每 decimation 步渲染一次
        self.sim.physics_material = self.scene.terrain.physics_material  # 使用场景中定义的物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # PhysX GPU 参数优化

        # 可视化设置
        self.viewer.eye = (1.5, 1.5, 1.5)  # 相机位置
        self.viewer.origin_type = "asset_root"  # 相机关注点
        self.viewer.asset_name = "robot"  # 关注的资产