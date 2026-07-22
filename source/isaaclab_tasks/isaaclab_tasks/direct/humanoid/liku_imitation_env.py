# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Robot asset
# -----------------------------------------------------------------------------

# 22축 전체 모델 USD 경로.
# 네 실제 22-motor USD 파일명에 맞게 여기만 바꾸면 됨.
LIKU_USD_PATH = "D:/isaacsim/jhusd/new_test/original.usd"

# trajectory column 순서도 반드시 아래 CONTROLLED_JOINTS 순서와 같아야 함.
# 아래 순서는 사용자가 보여준 USD Stage 트리의 PhysicsRevolute 순서를 기준으로 맞춘다.
# HeadFixedJoint/HeadFixedJoint2는 fixed joint라 제어하지 않는다.
# D13은 Revolute로 보이지만 로그에 ID13/14가 없으므로 0 rad target으로 고정한다.
RIGHT_LEG_JOINTS = ["A1", "A2", "A3", "A4", "A5", "A6"]
LEFT_LEG_JOINTS = ["B7", "B8", "B9", "B10", "B11", "B12"]
HEAD_HOLD_JOINTS = ["D13"]
RIGHT_ARM_JOINTS = ["E15", "E16", "E17", "E18"]
LEFT_ARM_JOINTS = ["F19", "F20", "F21", "F22"]

# Main order: visible USD-stage order from the screenshot.
CONTROLLED_JOINTS = [
    "D13",
    "F19", "E15", "F20", "E16", "E17", "F21", "F22", "E18",
    "A1", "B7", "B8", "A2", "B9", "A3", "A5", "A4", "B10", "B11", "B12", "A6",
]

RESIDUAL_FIXED_JOINTS = ["D13"]


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q: [N, 4] quaternion in (w, x, y, z)
    v: [N, 3] vector
    """
    q_xyz = q[:, 1:4]
    q_w = q[:, 0:1]

    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)

    return v + 2.0 * (q_w * uv + uuv)


def quat_angle_diff_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    q1, q2: [N, 4] quaternion in (w, x, y, z)
    return: angle difference [N] in radians
    """
    q1 = F.normalize(q1, dim=-1, eps=1e-6)
    q2 = F.normalize(q2, dim=-1, eps=1e-6)

    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, 0.0, 1.0)

    return 2.0 * torch.acos(dot)


LIKU_IMI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIKU_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.015),

        # local +Y -> world +Z
        # local -X -> world +X
        rot=(0.0, 0.0, 0.7071068, 0.7071068),

        joint_pos={
            # 180deg neutral -> IsaacLab 0rad.
            # D13은 head hold target이므로 0 rad에서 시작한다.
            "D13": 0.0,

            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "A4": 0.0,
            "A5": 0.0,
            "A6": 0.0,

            "B7": 0.0,
            "B8": 0.0,
            "B9": 0.0,
            "B10": 0.0,
            "B11": 0.0,
            "B12": 0.0,

            "E15": 0.0,
            "E16": 0.0,
            "E17": 0.0,
            "E18": 0.0,

            "F19": 0.0,
            "F20": 0.0,
            "F21": 0.0,
            "F22": 0.0,
        },
    ),
    actuators={
        # 하체는 보행 안정성이 중요해서 조금 강하게.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_LEG_JOINTS + LEFT_LEG_JOINTS,
            effort_limit_sim=300000.0,
            velocity_limit_sim=300.0,
            # 약간 더 강하게: trajectory/잔차 target을 더 단단히 잡는 테스트용.
            # 너무 올리면 바닥 접촉에서 튈 수 있으니 우선 +25% 정도만 적용한다.
            stiffness=250000.0,
            damping=25000.0,
        ),
        # 팔은 실제 로그 궤적을 따라가되 처음에는 낮은 stiffness로 둔다.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS,
            effort_limit_sim=15000.0,
            velocity_limit_sim=200.0,
            stiffness=30000.0,
            damping=3000.0,
        ),
        # Head는 imitation 대상에서 고정. policy residual도 mask로 0 처리한다.
        "head_hold": ImplicitActuatorCfg(
            joint_names_expr=HEAD_HOLD_JOINTS,
            effort_limit_sim=15000.0,
            velocity_limit_sim=100.0,
            # D13은 residual 대상이 아니라 고정 기준점이라 강하게 잡는다.
            stiffness=30000.0,
            damping=3000.0,
        ),
    },
)


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------

@configclass
class LikuImitationEnvCfg(DirectRLEnvCfg):

    decimation = 2

    # action 의미:
    #   최종 joint position target = q_zmp + action_scale * policy_action
    # 단위: rad
    # 일단 넘어짐을 줄이기 위해 residual 보정량은 작게 둔다.
    # 0.005 rad ≈ 0.29 deg. action이 포화돼도 실제 target 변화는 작다.
    action_scale = 0.012

    action_space = 21

    # reset 직후 random residual이 바로 들어가면 넘어진다.
    # 30 step = decimation=2, dt=1/120 기준 약 0.5초.
    residual_warmup_steps: int = 30
    reset_to_zmp_first_frame: bool = False

    # trajectory 시작 전에 q_zmp[0] 자세로 잠깐 버틴다.
    # 60 step = 약 1초. phase가 이 시간 동안 0에 고정된다.
    zmp_start_delay_steps: int = 0

    # -------------------------------------------------------------------------
    # walking / soft imitation blend
    # -------------------------------------------------------------------------
    # trajectory를 처음부터 100% 따라가지 않고 default standing pose와 섞는다.
    # 0.0 = 서 있는 default 자세만 사용, 1.0 = q_zmp trajectory 100% 사용.
    # stand checkpoint에서 이어서 걸음 학습을 시작할 때는 0.20 정도로 옅게 시작한다.
    zmp_follow_alpha: float = 0.35

    # 발목은 실제 로봇 trajectory를 그대로 따라가면 시뮬 접촉에서 불안정할 수 있다.
    # 그래서 발목만 별도 alpha를 낮춰서 default pose + policy residual 중심으로 보정하게 한다.
    # 0.00 = 발목은 trajectory를 안 따라감, 0.05 = 아주 약하게만 따라감.
    ankle_zmp_follow_alpha: float = 0.05
    ankle_follow_joint_names: list[str] = ["A5", "A6", "B11", "B12"]

    # custom observation: 기존 95 + forward/side displacement 2개 = 97
    observation_space = 97
    state_space = 0

    # required by parent LocomotionEnv observations
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1
    contact_force_scale: float = 0.01

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=4.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = LIKU_IMI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -------------------------------------------------------------------------
    # imitation / ZMP trajectory
    # -------------------------------------------------------------------------
    # "residual": q_target = q_zmp + action_scale * actions
    # "replay":   q_target = q_zmp only. ZMP trajectory 검증용.
    # "direct":   q_target = default_joint_pos + action_scale * actions. ZMP 없이 position action 테스트용.

    # default: residual
    control_mode: str = "residual"
    # control_mode: str = "replay"
    # control_mode: str = "hold_current"
    # control_mode: str = "teleport_replay"
    # "visual_replay": 그래픽 확인용. root 고정 + trajectory index 직접 증가 + reset 방지.
    # control_mode: str = "visual_replay"
    # control_mode: str = "direct"

    # visual_replay 전용. 1이면 원본 속도, 2~5면 빠르게 전체 모션 확인.
    visual_replay_stride: int = 10
    visual_replay_height_offset: float = 1.0

    # trajectory 값을 코드에서 직접 수정하고 싶을 때 True.
    # 예: _apply_user_joint_value_edit() 안에서 A3 = -A3, A3 = A3 + math.radians(5.0) 처럼 수정.
    enable_user_joint_value_edit: bool = True

    # ZMP trajectory 파일 경로.
    # 비워두면 q_zmp = 0으로 동작함.
    # 지원 형식:
    #   1) .pt / .pth: Tensor [T, 22] 또는 dict 안의 "q", "joint_pos", "positions", "trajectory"
    #   2) .csv: header 있어도 됨. [T, 22] 또는 첫 column phase/time + [T, 22]
    zmp_traj_path: str = "D:/IsaacLab/traj/traj_headfix_60hz.pt"
    # zmp_traj_path: str = ""

    zmp_traj_in_degrees: bool = False
    zmp_cycle_time: float = 13.109
    episode_length_s = 13.6
    randomize_initial_phase: bool = False

    # phase를 observation에 sin/cos로 추가할지 여부.
    # True면 custom observation에 phase sin/cos 2개가 포함됨.
    use_phase_observation: bool = True

    # -------------------------------------------------------------------------
    # termination / posture
    # -------------------------------------------------------------------------
    death_cost: float = -6.0
    # 초반 학습에서는 너무 빨리 죽이면 복구 동작을 배울 시간이 없다.
    termination_height: float = 1.70
    bad_posture_threshold: float = 0.75

    stand_target_height: float = 2.01
    stand_height_sigma: float = 0.05
    upright_sigma: float = 0.20
    heading_sigma: float = 0.45

    # -------------------------------------------------------------------------
    # reward scales
    # -------------------------------------------------------------------------
    alive_scale: float = 0.10

    # 초반 residual 학습은 imitation보다 "안 넘어지기"를 우선한다.
    # trajectory는 기준 자세로만 쓰고, policy가 균형 보정을 배우게 한다.
    imitation_pos_scale: float = 0.70
    imitation_vel_scale: float = 0.02

    # imitation reward는 자세가 어느 정도 안정적일 때만 크게 준다.
    # 넘어지는 중에도 trajectory만 잘 따라가면 보상받는 문제를 막기 위한 gate.
    stable_imi_up_threshold: float = 0.90
    stable_imi_up_width: float = 0.08
    stable_imi_height_threshold: float = 1.86
    stable_imi_height_width: float = 0.12

    upright_scale: float = 2.50
    height_scale: float = 2.00
    heading_scale: float = 0.10

    # 몸의 yaw/heading이 틀어진 상태로 앞으로 이동하는 해법을 막기 위한 보강.
    # deadband 이내의 작은 흔들림은 허용하고, 전진/누적전진 보상도 heading gate로 같이 줄인다.
    heading_angle_cost: float = 0.50
    heading_angle_deadband: float = 0.1  # rad, 약 2.9도
    heading_forward_gate_width: float = 0.35  # rad, 약 20도

    # 걷는 목표를 유지하되, 초반에는 너무 앞으로 던지지 않도록 약하게만 준다.
    forward_scale: float = 0.45
    backward_vel_cost: float = 2.50
    side_vel_cost: float = 1.00
    z_vel_cost: float = 1.50
    target_forward_vel: float = 0.18
    forward_vel_sigma: float = 0.15

    # 순간 forward velocity만 보상하면 앞으로 갔다가 다시   밀리는 해법이 남을 수 있다.
    # 그래서 reset 기준 누적 전진거리도 별도 보상으로 준다.
    forward_disp_scale: float = 0.95
    forward_disp_target: float = 0.40

    # 앞으로 갔다가 다시 뒤로 밀리는 왕복 해법을 막기 위한 max-progress 기준 penalty.
    # deadband 이내의 작은 흔들림은 보행 균형 보정으로 보고 허용한다.
    backtrack_cost: float = 1.00
    backtrack_deadband: float = 0.08

    # 옆으로 많이 새면서 forward_disp_reward를 받는 해법을 막는다.
    # side_disp가 커질수록 forward_disp_reward를 gate로 깎는다.
    forward_disp_side_gate_width: float = 0.22

    # 옆으로 누적 drift 되는 현상을 막기 위한 reset 기준 lateral position penalty.
    side_pos_cost: float = 6.0
    side_pos_deadband: float = 0.08

    # 낮아지거나 기울어지는 중에도 healthy_gate 때문에 penalty가 약해지지 않게 별도 penalty를 둔다.
    low_height_cost: float = 8.0
    tilt_cost: float = 3.0



    # imitation reward sigma
    # pos는 rad 기준, vel은 rad/s 기준.
    zmp_joint_sigma: float = 0.35
    zmp_vel_sigma: float = 1.50

    # regularization
    action_cost_scale: float = 0.050
    action_rate_cost_scale: float = 0.120
    residual_target_cost_scale: float = 0.050
    action_saturation_cost_scale: float = 0.100
    action_soft_limit: float = 0.80
    joint_vel_cost_scale: float = 0.002
    joint_limit_cost_scale: float = 0.50
    joint_limit_margin: float = 0.05

    # debug
    # POS_CTRL는 target 검증이 끝났으면 꺼둔다. MOVE/POSTURE/REWARD만 주기적으로 본다.
    debug_every: int = 300
    debug_pos_ctrl: bool = False

    # parent LocomotionEnv 호환용. imitation env에서는 직접 사용하지 않음.
    joint_gears: list = [1.0] * 21

    # LHIP/RHIP 명칭이어도 실제로는 foot body로 취급
    foot_body_names = ["Hip6", "LHip6"]

    # -------------------------------------------------------------------------
    # CUSTOM AXIS
    # -------------------------------------------------------------------------
    robot_up_axis_local = (0.0, 1.0, 0.0)
    robot_forward_axis_local = (-1.0, 0.0, 0.0)

    world_up_axis = (0.0, 0.0, 1.0)

    # 보는 방향 기준
    world_forward_axis = (1.0, 0.0, 0.0)

    # 실제 이동 보상 기준
    # 기존 코드와 동일하게 root가 world -X로 이동하면 forward reward.
    world_move_axis = (-1.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------

class LikuImitationEnv(LocomotionEnv):
    cfg: LikuImitationEnvCfg

    def __init__(self, cfg: LikuImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug_every = int(self.cfg.debug_every)
        self._debug_env_id = 0
        self._printed_initial_axis = False

        # ---------------------------------------------------------------------
        # controlled joints
        # ---------------------------------------------------------------------
        joint_ids = []
        joint_names = []

        for name in CONTROLLED_JOINTS:
            ids, names = self.robot.find_joints(name)

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(names, tuple):
                names = list(names)

            if len(ids) == 0:
                raise RuntimeError(f"[LIKU-IMI] joint '{name}' not found in robot articulation.")

            joint_ids.extend(ids)
            joint_names.extend(names)

        self._joint_dof_idx = joint_ids
        self._joint_names = joint_names

        if len(self._joint_dof_idx) != self.cfg.action_space:
            raise RuntimeError(
                f"[LIKU-IMI] controlled joint count ({len(self._joint_dof_idx)}) "
                f"!= action_space ({self.cfg.action_space})"
            )

        # ---------------------------------------------------------------------
        # joint limits
        # ---------------------------------------------------------------------
        num_envs = self._get_root_pos_w().shape[0]

        if hasattr(self.robot.data, "soft_joint_pos_limits"):
            joint_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_dof_idx, :]
            self._joint_lower_limits = joint_limits[:, :, 0].clone()
            self._joint_upper_limits = joint_limits[:, :, 1].clone()
        elif hasattr(self.robot.data, "joint_pos_limits"):
            joint_limits = self.robot.data.joint_pos_limits[:, self._joint_dof_idx, :]
            self._joint_lower_limits = joint_limits[:, :, 0].clone()
            self._joint_upper_limits = joint_limits[:, :, 1].clone()
        else:
            logger.warning("[LIKU-IMI] joint position limits not found. Fallback to [-pi, pi].")
            self._joint_lower_limits = torch.full(
                (num_envs, self.cfg.action_space),
                -math.pi,
                dtype=torch.float32,
                device=self.device,
            )
            self._joint_upper_limits = torch.full(
                (num_envs, self.cfg.action_space),
                math.pi,
                dtype=torch.float32,
                device=self.device,
            )

        self._default_leg_q = self.robot.data.default_joint_pos[:, self._joint_dof_idx].detach().clone()

        # ---------------------------------------------------------------------
        # foot bodies. 현재 reward에는 안 쓰지만, 디버그/확장용으로 유지.
        # ---------------------------------------------------------------------
        foot_body_ids = []
        foot_body_names_found = []

        for name in self.cfg.foot_body_names:
            ids, names = self.robot.find_bodies(name)

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(names, tuple):
                names = list(names)

            if len(ids) == 0:
                raise RuntimeError(f"[LIKU-IMI] foot body '{name}' not found in robot bodies.")

            foot_body_ids.extend(ids)
            foot_body_names_found.extend(names)

        self._foot_body_ids = foot_body_ids
        self._foot_body_names = foot_body_names_found

        # ---------------------------------------------------------------------
        # axes
        # ---------------------------------------------------------------------
        self._robot_up_axis_local = torch.tensor(
            self.cfg.robot_up_axis_local,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._robot_forward_axis_local = torch.tensor(
            self.cfg.robot_forward_axis_local,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._world_up_axis = torch.tensor(
            self.cfg.world_up_axis,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._world_forward_axis = torch.tensor(
            self.cfg.world_forward_axis,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._world_move_axis = torch.tensor(
            self.cfg.world_move_axis,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # ---------------------------------------------------------------------
        # buffers
        # ---------------------------------------------------------------------
        self._prev_root_pos_w = self._get_root_pos_w().detach().clone()
        self._initial_root_pos_w = self._get_root_pos_w().detach().clone()

        # reset 이후 지금까지 가장 많이 전진한 거리.
        # forward_disp가 이 값보다 뒤로 물러나면 backtrack_penalty를 준다.
        self._max_forward_disp = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self._prev_actions = torch.zeros(
            num_envs,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )

        # ---------------------------------------------------------------------
        # residual action mask
        # ---------------------------------------------------------------------
        # residual 모드에서 policy가 보정할 joint를 제한한다.
        #
        # 현재 CONTROLLED_JOINTS 순서:
        # [
        #   D13,
        #   F19, E15, F20, E16, E17, F21, F22, E18,
        #   A1, B7, B8, A2, B9, A3, A5, A4, B10, B11, B12, A6
        # ]
        #
        # 처음 학습에서는 head + arms는 trajectory 그대로 따라가게 두고,
        # policy는 다리만 residual 보정하게 한다.

        self._residual_action_mask = torch.ones(
            1,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )

        # head + arms residual 금지
        _residual_fixed_names = [
            "D13",
            "F19", "E15", "F20", "E16", "E17", "F21", "F22", "E18",
        ]

        for _fixed_name in _residual_fixed_names:
            if _fixed_name in self._joint_names:
                self._residual_action_mask[0, self._joint_names.index(_fixed_name)] = 0.0

        logger.info(
            "[LIKU-IMI] residual_action_mask = "
            f"{self._residual_action_mask[0].detach().cpu().tolist()}"
        )

        # ---------------------------------------------------------------------
        # per-joint ZMP follow alpha
        # ---------------------------------------------------------------------
        # 기본은 전체 zmp_follow_alpha를 쓰고,
        # 발목 joint만 ankle_zmp_follow_alpha로 낮춰서 trajectory 강제 추종을 완화한다.
        self._zmp_follow_alpha_vec = torch.full(
            (1, self.cfg.action_space),
            float(self.cfg.zmp_follow_alpha),
            dtype=torch.float32,
            device=self.device,
        )

        for _ankle_name in self.cfg.ankle_follow_joint_names:
            if _ankle_name in self._joint_names:
                self._zmp_follow_alpha_vec[0, self._joint_names.index(_ankle_name)] = float(
                    self.cfg.ankle_zmp_follow_alpha
                )
            else:
                logger.warning(f"[LIKU-IMI] ankle joint name not found: {_ankle_name}")

        logger.info(
            "[LIKU-IMI] zmp_follow_alpha_vec = "
            f"{self._zmp_follow_alpha_vec[0].detach().cpu().tolist()}"
        )



        self._phase_offset = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self._last_q_zmp = torch.zeros(
            num_envs,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_q_zmp_vel = torch.zeros_like(self._last_q_zmp)
        self._last_q_target = self._default_leg_q.detach().clone()

        # ZMP trajectory table load
        self._load_zmp_table()


        try:
            stiff = self.robot.data.default_joint_stiffness[0, self._joint_dof_idx]
            damp = self.robot.data.default_joint_damping[0, self._joint_dof_idx]

            logger.info(
                "[LIKU-IMI][CHECK] applied stiffness = "
                f"{stiff.detach().cpu().tolist()}"
            )
            logger.info(
                "[LIKU-IMI][CHECK] applied damping = "
                f"{damp.detach().cpu().tolist()}"
            )
        except Exception as e:
            logger.warning(f"[LIKU-IMI][CHECK] stiffness/damping print failed: {e}")

        try:
            for actuator_name, actuator in self.robot.actuators.items():
                logger.info(f"[LIKU-IMI][ACTUATOR] name={actuator_name}")
                logger.info(f"[LIKU-IMI][ACTUATOR] joint_indices={actuator.joint_indices}")
        except Exception as e:
            logger.warning(f"[LIKU-IMI][ACTUATOR] print failed: {e}")




        logger.info(f"[LIKU-IMI] control_mode = {self.cfg.control_mode}")
        logger.info(f"[LIKU-IMI] controlled joints = {self._joint_names}")
        logger.info(f"[LIKU-IMI] controlled joint ids = {self._joint_dof_idx}")
        logger.info(f"[LIKU-IMI] action_space = {self.cfg.action_space}")
        logger.info(f"[LIKU-IMI] observation_space = {self.cfg.observation_space}")
        logger.info(f"[LIKU-IMI] action_scale(rad) = {self.cfg.action_scale}")
        logger.info(f"[LIKU-IMI] zmp_traj_path = '{self.cfg.zmp_traj_path}'")
        logger.info(f"[LIKU-IMI] zmp_cycle_time = {self.cfg.zmp_cycle_time}")
        logger.info(f"[LIKU-IMI] zmp_follow_alpha = {self.cfg.zmp_follow_alpha}")
        logger.info(f"[LIKU-IMI] ankle_zmp_follow_alpha = {self.cfg.ankle_zmp_follow_alpha}")
        logger.info(f"[LIKU-IMI] ankle_follow_joint_names = {self.cfg.ankle_follow_joint_names}")
        logger.info(f"[LIKU-IMI] foot body names = {self._foot_body_names}")
        logger.info(f"[LIKU-IMI] foot body ids = {self._foot_body_ids}")

        if hasattr(self.robot, "body_names"):
            logger.info(f"[LIKU-IMI] all body names = {self.robot.body_names}")

        logger.info(f"[LIKU-IMI] robot_up_axis_local = {self.cfg.robot_up_axis_local}")
        logger.info(f"[LIKU-IMI] robot_forward_axis_local = {self.cfg.robot_forward_axis_local}")
        logger.info(f"[LIKU-IMI] world_forward_axis(heading) = {self.cfg.world_forward_axis}")
        logger.info(f"[LIKU-IMI] world_move_axis(reward) = {self.cfg.world_move_axis}")

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _get_root_quat_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_quat_w"):
            return self.robot.data.root_quat_w
        return self.robot.data.root_state_w[:, 3:7]

    def _get_root_pos_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_pos_w"):
            return self.robot.data.root_pos_w
        return self.robot.data.root_state_w[:, 0:3]

    def _get_root_lin_vel_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_lin_vel_w"):
            return self.robot.data.root_lin_vel_w
        return self.robot.data.root_state_w[:, 7:10]

    def _get_root_ang_vel_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_ang_vel_w"):
            return self.robot.data.root_ang_vel_w
        return self.robot.data.root_state_w[:, 10:13]

    def _get_foot_quat_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "body_quat_w"):
            return self.robot.data.body_quat_w[:, self._foot_body_ids, :]
        return self.robot.data.body_state_w[:, self._foot_body_ids, 3:7]

    def _get_foot_pos_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "body_pos_w"):
            return self.robot.data.body_pos_w[:, self._foot_body_ids, :]
        return self.robot.data.body_state_w[:, self._foot_body_ids, 0:3]

    def _get_custom_axis_proj(self):
        root_quat = self._get_root_quat_w()
        num_envs = root_quat.shape[0]

        up_local = self._robot_up_axis_local.expand(num_envs, -1)
        fwd_local = self._robot_forward_axis_local.expand(num_envs, -1)
        world_up = self._world_up_axis.expand(num_envs, -1)
        world_fwd = self._world_forward_axis.expand(num_envs, -1)

        up_world = quat_rotate_wxyz(root_quat, up_local)
        fwd_world = quat_rotate_wxyz(root_quat, fwd_local)

        up_world = F.normalize(up_world, dim=-1, eps=1e-6)
        fwd_world = F.normalize(fwd_world, dim=-1, eps=1e-6)
        world_up = F.normalize(world_up, dim=-1, eps=1e-6)
        world_fwd = F.normalize(world_fwd, dim=-1, eps=1e-6)

        custom_up_proj = torch.sum(up_world * world_up, dim=-1)
        custom_heading_proj = torch.sum(fwd_world * world_fwd, dim=-1)

        return custom_up_proj, custom_heading_proj

    def _debug_axis_candidates(self):
        q = self._get_root_quat_w()[0:1]

        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        q6 = q.expand(6, -1)
        rotated = quat_rotate_wxyz(q6, candidates)

        world_up = self._world_up_axis.expand(6, -1)
        world_fwd = self._world_forward_axis.expand(6, -1)

        rotated = F.normalize(rotated, dim=-1, eps=1e-6)
        world_up = F.normalize(world_up, dim=-1, eps=1e-6)
        world_fwd = F.normalize(world_fwd, dim=-1, eps=1e-6)

        up_scores = torch.sum(rotated * world_up, dim=-1)
        fwd_scores = torch.sum(rotated * world_fwd, dim=-1)

        msg = []
        for i in range(6):
            msg.append(
                f"{names[i]}: up={up_scores[i].item():+.3f}, "
                f"fwd={fwd_scores[i].item():+.3f}"
            )

        logger.info("[AXIS_TEST] " + " | ".join(msg))

    def _apply_user_joint_value_edit(self, q_in: torch.Tensor) -> torch.Tensor:
        """
        trajectory table을 IsaacLab에 넣기 직전에 joint별로 직접 수정하는 곳.

        q_in/q shape:
          - [T, 21]      : trajectory table 로딩 시
          - [num_envs,21]: 필요 시 배치 target에도 사용 가능

        여기서는 CONTROLLED_JOINTS 순서 기준으로 값을 꺼내고 다시 넣는다.
        기본값은 전부 identity라서 "A3 = A3"처럼 아무 변경도 하지 않는다.

        사용 예:
          A3 = -A3
          A3 = A3 + math.radians(5.0)
          F19 = -F19
          B10 = B10 * 0.8

        주의:
          이 함수는 q_table 로딩 시 적용되고, 이후 qd_table은 수정된 q_table 기준으로 다시 계산된다.
          그래서 부호/scale/offset을 여기서 바꾸면 velocity target도 자동으로 일관되게 바뀐다.
        """
        if not bool(getattr(self.cfg, "enable_user_joint_value_edit", True)):
            return q_in

        q = q_in.clone()

        idx = {name: i for i, name in enumerate(self._joint_names)}

        # ------------------------------------------------------------------
        # 1) 현재 trajectory 값을 이름으로 꺼내기
        # ------------------------------------------------------------------
        D13 = q[:, idx["D13"]]

        F19 = q[:, idx["F19"]]
        E15 = q[:, idx["E15"]]
        F20 = q[:, idx["F20"]]
        E16 = q[:, idx["E16"]]
        E17 = q[:, idx["E17"]]
        F21 = q[:, idx["F21"]]
        F22 = q[:, idx["F22"]]
        E18 = q[:, idx["E18"]]

        A1 = q[:, idx["A1"]]
        B7 = q[:, idx["B7"]]
        B8 = q[:, idx["B8"]]
        A2 = q[:, idx["A2"]]
        B9 = q[:, idx["B9"]]
        A3 = q[:, idx["A3"]]
        A5 = q[:, idx["A5"]]
        A4 = q[:, idx["A4"]]
        B10 = q[:, idx["B10"]]
        B11 = q[:, idx["B11"]]
        B12 = q[:, idx["B12"]]
        A6 = q[:, idx["A6"]]

        # ------------------------------------------------------------------
        # 2) 사용자가 직접 수정하는 영역
        #    우선은 전부 identity로 둠.
        #
        #    예시:
        #      A3 = -A3
        #      A3 = A3 + math.radians(5.0)
        #      F19 = -F19
        # ------------------------------------------------------------------

        D13 = D13

        F19 = F19
        E15 = E15
        F20 = F20
        E16 = E16
        E17 = E17
        F21 = F21
        F22 = F22
        E18 = E18

        A1 = A1
        B7 = B7
        B8 = B8
        A2 = A2
        B9 = B9 + math.radians(-20.0)
        A3 = A3 + math.radians(-20.0)
        A5 = A5
        A4 = A4
        B10 = -B10
        B11 = B11
        B12 = B12
        A6 = A6

        # ------------------------------------------------------------------
        # 3) 수정된 값을 다시 q에 넣기
        # ------------------------------------------------------------------
        q[:, idx["D13"]] = D13

        q[:, idx["F19"]] = F19
        q[:, idx["E15"]] = E15
        q[:, idx["F20"]] = F20
        q[:, idx["E16"]] = E16
        q[:, idx["E17"]] = E17
        q[:, idx["F21"]] = F21
        q[:, idx["F22"]] = F22
        q[:, idx["E18"]] = E18

        q[:, idx["A1"]] = A1
        q[:, idx["B7"]] = B7
        q[:, idx["B8"]] = B8
        q[:, idx["A2"]] = A2
        q[:, idx["B9"]] = B9
        q[:, idx["A3"]] = A3
        q[:, idx["A5"]] = A5
        q[:, idx["A4"]] = A4
        q[:, idx["B10"]] = B10
        q[:, idx["B11"]] = B11
        q[:, idx["B12"]] = B12
        q[:, idx["A6"]] = A6

        return q

    def _load_zmp_table(self):
        """
        ZMP joint trajectory를 self._zmp_q_table / self._zmp_qd_table로 로딩한다.
        없으면 zero trajectory로 둔다.
        """
        self._zmp_q_table = None
        self._zmp_qd_table = None

        path_str = str(self.cfg.zmp_traj_path).strip()
        if path_str == "":
            logger.warning("[LIKU-IMI] zmp_traj_path is empty. Use zero ZMP target.")
            return

        path = Path(path_str)
        if not path.exists():
            logger.warning(f"[LIKU-IMI] zmp_traj_path not found: {path}. Use zero ZMP target.")
            return

        suffix = path.suffix.lower()

        if suffix in [".pt", ".pth"]:
            data = torch.load(path, map_location=self.device)

            traj_joint_names = None
            if isinstance(data, torch.Tensor):
                q_table = data
            elif isinstance(data, dict):
                traj_joint_names = data.get("joint_names", None)
                q_table = None
                for key in ["q", "joint_pos", "positions", "trajectory", "zmp_q"]:
                    if key in data:
                        q_table = data[key]
                        break
                if q_table is None:
                    raise RuntimeError(
                        "[LIKU-IMI] .pt file must be Tensor or dict with one of "
                        "['q', 'joint_pos', 'positions', 'trajectory', 'zmp_q']."
                    )
                if not isinstance(q_table, torch.Tensor):
                    q_table = torch.tensor(q_table, dtype=torch.float32, device=self.device)
            else:
                raise RuntimeError(f"[LIKU-IMI] unsupported .pt data type: {type(data)}")

            if traj_joint_names is not None:
                traj_joint_names = [str(x) for x in list(traj_joint_names)]
                if traj_joint_names != list(self._joint_names):
                    raise RuntimeError(
                        "[LIKU-IMI] trajectory joint order mismatch!\n"
                        f"trajectory: {traj_joint_names}\n"
                        f"env joints : {list(self._joint_names)}"
                    )

            q_table = q_table.to(device=self.device, dtype=torch.float32)

        elif suffix == ".csv":
            rows = []
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 0:
                        continue
                    try:
                        values = [float(x) for x in row]
                    except ValueError:
                        # header skip
                        continue
                    rows.append(values)

            if len(rows) == 0:
                raise RuntimeError(f"[LIKU-IMI] CSV has no numeric rows: {path}")

            q_table = torch.tensor(rows, dtype=torch.float32, device=self.device)

        else:
            raise RuntimeError(f"[LIKU-IMI] unsupported zmp trajectory format: {suffix}")

        if q_table.ndim != 2:
            raise RuntimeError(f"[LIKU-IMI] zmp q_table must be 2D [T, D], got shape={tuple(q_table.shape)}")

        # CSV 등에서 [T, action+1]이면 첫 column을 phase/time으로 보고 제거.
        # [T, action+2]이면 time_s, phase 두 column으로 보고 제거.
        if q_table.shape[1] == self.cfg.action_space + 2:
            q_table = q_table[:, 2:]
        elif q_table.shape[1] == self.cfg.action_space + 1:
            q_table = q_table[:, 1:]

        if q_table.shape[1] != self.cfg.action_space:
            raise RuntimeError(
                f"[LIKU-IMI] zmp q_table column count must be {self.cfg.action_space}, "
                f"{self.cfg.action_space + 1}, or {self.cfg.action_space + 2}, "
                f"got shape={tuple(q_table.shape)}"
            )

        if self.cfg.zmp_traj_in_degrees:
            q_table = q_table * (math.pi / 180.0)

        # ------------------------------------------------------------
        # User editable joint remap/sign/offset area.
        # _apply_user_joint_value_edit() 안에서 A3 = A3, A3 = -A3 같은 식으로 수정 가능.
        # qd_table은 아래에서 수정된 q_table 기준으로 다시 계산된다.
        # ------------------------------------------------------------
        q_table = self._apply_user_joint_value_edit(q_table)

        if q_table.shape[0] < 2:
            raise RuntimeError("[LIKU-IMI] zmp q_table must have at least 2 rows.")

        self._zmp_q_table = q_table.contiguous()

        # finite difference velocity table. cyclic trajectory라고 보고 마지막은 처음과 연결.
        dt_table = max(float(self.cfg.zmp_cycle_time) / float(q_table.shape[0]), 1.0e-6)
        q_next = torch.roll(q_table, shifts=-1, dims=0)
        self._zmp_qd_table = ((q_next - q_table) / dt_table).contiguous()

        logger.info(
            f"[LIKU-IMI] loaded ZMP trajectory: path={path}, "
            f"shape={tuple(self._zmp_q_table.shape)}, "
            f"degrees={self.cfg.zmp_traj_in_degrees}, "
            f"user_joint_edit={bool(getattr(self.cfg, 'enable_user_joint_value_edit', True))}"
        )

    def _get_phase(self) -> torch.Tensor:
        dt = float(getattr(self, "step_dt", self.cfg.sim.dt * self.cfg.decimation))
        cycle_time = max(float(self.cfg.zmp_cycle_time), 1.0e-6)

        # episode 시작 직후에는 q_zmp[0] 자세로 잠깐 버틴다.
        # 이 구간에서는 phase가 0이라 trajectory가 진행하지 않는다.
        delay_steps = max(int(getattr(self.cfg, "zmp_start_delay_steps", 0)), 0)
        effective_step = torch.clamp(
            self.episode_length_buf.float() - float(delay_steps),
            min=0.0,
        )

        phase = (
            effective_step
            * dt
            / cycle_time
            + self._phase_offset
        ) % 1.0

        return phase

    def _get_phase_obs(self) -> torch.Tensor:
        phase = self._get_phase()
        return torch.stack(
            [
                torch.sin(2.0 * math.pi * phase),
                torch.cos(2.0 * math.pi * phase),
            ],
            dim=-1,
        )

    def _get_zmp_joint_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return:
            q_zmp:     [num_envs, action_space], rad
            q_zmp_vel: [num_envs, action_space], rad/s
        """
        num_envs = self._get_root_pos_w().shape[0]

        if self._zmp_q_table is None:
            q_zmp = torch.zeros(
                num_envs,
                self.cfg.action_space,
                dtype=torch.float32,
                device=self.device,
            )
            q_zmp_vel = torch.zeros_like(q_zmp)
            return q_zmp, q_zmp_vel

        phase = self._get_phase()
        table = self._zmp_q_table
        vel_table = self._zmp_qd_table

        T = table.shape[0]
        idx_f = phase * float(T)
        idx0 = torch.floor(idx_f).long() % T
        idx1 = (idx0 + 1) % T
        alpha = (idx_f - torch.floor(idx_f)).unsqueeze(-1)

        q0 = table[idx0]
        q1 = table[idx1]
        q_zmp = (1.0 - alpha) * q0 + alpha * q1

        v0 = vel_table[idx0]
        v1 = vel_table[idx1]
        q_zmp_vel = (1.0 - alpha) * v0 + alpha * v1

        return q_zmp, q_zmp_vel

    def _get_visual_replay_joint_target(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        그래픽 확인용 trajectory target.
        episode_length_buf / reset / phase와 무관하게 common_step_counter로 trajectory index를 직접 증가시킨다.
        """
        num_envs = self._get_root_pos_w().shape[0]

        if self._zmp_q_table is None:
            q = torch.zeros(
                num_envs,
                self.cfg.action_space,
                dtype=torch.float32,
                device=self.device,
            )
            qd = torch.zeros_like(q)
            return q, qd, 0

        table = self._zmp_q_table
        vel_table = self._zmp_qd_table
        T = table.shape[0]

        stride = max(int(getattr(self.cfg, "visual_replay_stride", 1)), 1)
        idx = (int(self.common_step_counter) * stride) % T

        q_one = table[idx]
        qd_one = vel_table[idx]

        q = q_one.unsqueeze(0).repeat(num_envs, 1)
        qd = qd_one.unsqueeze(0).repeat(num_envs, 1)

        return q, qd, idx

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(
                self._get_root_pos_w().shape[0],
                device=self.device,
            )

        if hasattr(self, "_prev_root_pos_w"):
            self._prev_root_pos_w[env_ids] = self._get_root_pos_w()[env_ids].detach().clone()

        if hasattr(self, "_initial_root_pos_w"):
            self._initial_root_pos_w[env_ids] = self._get_root_pos_w()[env_ids].detach().clone()

        if hasattr(self, "_max_forward_disp"):
            self._max_forward_disp[env_ids] = 0.0

        if hasattr(self, "_prev_actions"):
            self._prev_actions[env_ids] = 0.0

        if hasattr(self, "_phase_offset"):
            if self.cfg.randomize_initial_phase:
                self._phase_offset[env_ids] = torch.rand(
                    len(env_ids),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                self._phase_offset[env_ids] = 0.0

        if hasattr(self, "_last_q_zmp"):
            self._last_q_zmp[env_ids] = 0.0
        if hasattr(self, "_last_q_zmp_vel"):
            self._last_q_zmp_vel[env_ids] = 0.0
        if hasattr(self, "_last_q_target"):
            self._last_q_target[env_ids] = self._default_leg_q[env_ids].detach().clone()


        # reset 때 joint를 trajectory 첫 프레임으로 맞춘다.
        # 시작하자마자 default pose -> q_zmp[0]로 확 튀는 문제 방지.
        if getattr(self.cfg, "reset_to_zmp_first_frame", True):
            if self._zmp_q_table is not None:
                q0 = self._zmp_q_table[0].unsqueeze(0).repeat(len(env_ids), 1)
                qd0 = torch.zeros_like(q0)

                self.robot.write_joint_state_to_sim(
                    position=q0,
                    velocity=qd0,
                    joint_ids=self._joint_dof_idx,
                    env_ids=env_ids,
                )

                self.robot.set_joint_position_target(
                    q0,
                    joint_ids=self._joint_dof_idx,
                    env_ids=env_ids,
                )

                self._last_q_zmp[env_ids] = q0.detach()
                self._last_q_zmp_vel[env_ids] = 0.0
                self._last_q_target[env_ids] = q0.detach()

    # -------------------------------------------------------------------------
    # RL hooks
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        if not self._printed_initial_axis:
            custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

            logger.info("==================================================")
            logger.info("[START_DIRECTION_CHECK][IMITATION]")
            logger.info(
                f"[AXIS_CUSTOM] up_proj={custom_up_proj[self._debug_env_id].item():.4f}, "
                f"heading_proj={custom_heading_proj[self._debug_env_id].item():.4f}"
            )
            self._debug_axis_candidates()
            logger.info("==================================================")

            self._printed_initial_axis = True

        super()._pre_physics_step(actions)

        if self.common_step_counter % self._debug_every == 0:
            a = self.actions[self._debug_env_id].detach().cpu()

            logger.info(
                f"[LIKU-IMI][step {int(self.common_step_counter)}] "
                f"action mean={a.mean().item():.4f}, "
                f"std={a.std().item():.4f}, "
                f"min={a.min().item():.4f}, "
                f"max={a.max().item():.4f}, "
                f"env0_action={a.tolist()}"
            )

            # 전체 env 기준 action 포화율을 함께 본다.
            # env0 action만 보면 특정 env 하나의 운에 판단이 흔들릴 수 있다.
            act_abs = torch.abs(self.actions)
            sat_ratio_all = (act_abs > 0.98).float().mean()
            mean_abs_action = act_abs.mean()

            sat_per_joint = (act_abs > 0.98).float().mean(dim=0)
            top_vals, top_ids = torch.topk(sat_per_joint, k=min(8, sat_per_joint.numel()))

            top_sat_msg = []
            for val, jid in zip(top_vals.detach().cpu().tolist(), top_ids.detach().cpu().tolist()):
                name = self._joint_names[jid] if jid < len(self._joint_names) else str(jid)
                top_sat_msg.append(f"{name}:{val:.2f}")

            logger.info(
                "[ACTION-STATS] "
                f"mean_abs_action={mean_abs_action.item():.4f}, "
                f"sat_ratio_all={sat_ratio_all.item():.4f}, "
                f"top_saturated_joints={top_sat_msg}"
            )

    def _apply_action(self):
        leg_q = self.robot.data.joint_pos[:, self._joint_dof_idx]

        # visual_replay는 episode phase가 아니라 common_step_counter로 trajectory index를 직접 증가시킨다.
        visual_idx = None
        if self.cfg.control_mode == "visual_replay":
            q_zmp, q_zmp_vel, visual_idx = self._get_visual_replay_joint_target()
        else:
            q_zmp, q_zmp_vel = self._get_zmp_joint_target()

        # episode 초반에는 residual action을 천천히 키운다.
        # 초기 random action 때문에 q_zmp가 바로 망가지는 문제 방지.
        warmup_steps = max(int(getattr(self.cfg, "residual_warmup_steps", 0)), 1)

        warmup_gate = torch.clamp(
            self.episode_length_buf.float().unsqueeze(-1) / float(warmup_steps),
            min=0.0,
            max=1.0,
        )

        q_residual = (
            float(self.cfg.action_scale)
            * warmup_gate
            * self.actions
            * self._residual_action_mask
        )



        if self.cfg.control_mode == "residual":
            # ------------------------------------------------------------
            # Per-joint soft ZMP following
            # ------------------------------------------------------------
            # 고관절/무릎은 zmp_follow_alpha를 쓰고,
            # 발목은 ankle_zmp_follow_alpha로 더 약하게 trajectory를 따라간다.
            #
            # alpha_vec = 0.0 -> default pose만 사용
            # alpha_vec = 1.0 -> q_zmp trajectory 100% 사용
            alpha_vec = self._zmp_follow_alpha_vec.expand(q_zmp.shape[0], -1)

            q_base = (
                (1.0 - alpha_vec) * self._default_leg_q
                + alpha_vec * q_zmp
            )

            q_target = q_base + q_residual

        elif self.cfg.control_mode == "replay":
            # ZMP trajectory 자체가 IsaacSim에서 말이 되는지 검증할 때 사용.
            q_target = q_zmp

        elif self.cfg.control_mode == "teleport_replay":
            # 디버그용:
            # actuator 힘으로 따라가는 게 아니라 joint state를 trajectory 값으로 직접 써버린다.
            # 다만 episode reset/phase는 기존 replay 방식을 따른다.
            q_target = q_zmp

        elif self.cfg.control_mode == "visual_replay":
            # 그래픽 확인용:
            # root 고정 + reset 방지 + trajectory index 직접 증가.
            # joint 순서/부호/offset이 맞는지 눈으로 확인하는 용도이며 학습용이 아니다.
            q_target = q_zmp

        elif self.cfg.control_mode == "direct":
            # ZMP 없이 position action 자체가 잘 먹는지 테스트할 때 사용.
            q_target = self._default_leg_q + q_residual

        elif self.cfg.control_mode == "hold_current":
            # 디버그용. 초기 default 자세를 target으로 유지한다.
            # q_target = leg_q.detach().clone()
            q_target = self._default_leg_q

        else:
            raise RuntimeError(
                f"[LIKU-IMI] unknown control_mode={self.cfg.control_mode}. "
                "Use 'residual', 'replay', 'teleport_replay', 'visual_replay', 'direct', or 'hold_current'."
            )

        q_target = torch.clamp(
            q_target,
            self._joint_lower_limits + self.cfg.joint_limit_margin,
            self._joint_upper_limits - self.cfg.joint_limit_margin,
        )

        self._last_q_zmp[:] = q_zmp.detach()
        self._last_q_zmp_vel[:] = q_zmp_vel.detach()
        self._last_q_target[:] = q_target.detach()

        # ------------------------------------------------------------
        # Debug log
        # ------------------------------------------------------------
        if bool(getattr(self.cfg, "debug_pos_ctrl", False)) and self.common_step_counter % self._debug_every == 0:
            env_id = self._debug_env_id
            q = leg_q[env_id].detach().cpu()
            qd = self.robot.data.joint_vel[env_id, self._joint_dof_idx].detach().cpu()
            qz = q_zmp[env_id].detach().cpu()
            qt = q_target[env_id].detach().cpu()
            qe = q - qt

            if visual_idx is not None:
                logger.info(
                    f"[VISUAL-REPLAY][step {int(self.common_step_counter)}] "
                    f"traj_idx={visual_idx}, "
                    f"stride={int(getattr(self.cfg, 'visual_replay_stride', 1))}, "
                    f"T={self._zmp_q_table.shape[0] if self._zmp_q_table is not None else 0}"
                )

            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"mode={self.cfg.control_mode}, "
                f"phase={self._get_phase()[env_id].item():.4f}"
            )
            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"env0_joint_pos={[round(x, 4) for x in q.tolist()]}"
            )
            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"env0_joint_vel={[round(x, 4) for x in qd.tolist()]}"
            )
            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"env0_q_zmp={[round(x, 4) for x in qz.tolist()]}"
            )
            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"env0_q_target={[round(x, 4) for x in qt.tolist()]}"
            )
            logger.info(
                f"[POS_CTRL][step {int(self.common_step_counter)}] "
                f"env0_q_error={[round(x, 4) for x in qe.tolist()]}"
            )

            # 가장 많이 틀어지는 joint만 따로 보기
            abs_err = qe.abs()
            top_vals, top_ids = torch.topk(abs_err, k=min(8, abs_err.numel()))

            logger.info(f"[TRACK-DETAIL][step {int(self.common_step_counter)}] worst joint errors:")
            for val, jid in zip(top_vals.tolist(), top_ids.tolist()):
                name = self._joint_names[jid]
                logger.info(
                    f"  {jid:02d} {name:>4s} | "
                    f"target={qt[jid].item():+.4f}, "
                    f"pos={q[jid].item():+.4f}, "
                    f"err={qe[jid].item():+.4f}, "
                    f"zmp={qz[jid].item():+.4f}"
                )

        # ------------------------------------------------------------
        # Apply action
        # ------------------------------------------------------------
        if self.cfg.control_mode == "visual_replay":
            # ------------------------------------------------------------
            # VISUAL DEBUG ONLY:
            # root를 초기 위치/자세에 완전히 고정하고 joint만 trajectory 값으로 강제 세팅한다.
            # 목적: trajectory joint 순서/부호/offset이 맞는지 그래픽으로 확인.
            # ------------------------------------------------------------
            if (not hasattr(self, "_visual_root_pose_w")) or (self._visual_root_pose_w.shape[0] != self._get_root_pos_w().shape[0]):
                root_pos = self._get_root_pos_w().detach().clone()
                root_quat = self._get_root_quat_w().detach().clone()
                # 그래픽 확인용으로 root를 위로 띄움
                root_pos[:, 2] += float(getattr(self.cfg, "visual_replay_height_offset", 0.0))

                self._visual_root_pose_w = torch.cat([root_pos, root_quat], dim=-1)

            root_pose = self._visual_root_pose_w.clone()
            root_vel = torch.zeros((self._get_root_pos_w().shape[0], 6), dtype=torch.float32, device=self.device)

            # root 고정
            self.robot.write_root_pose_to_sim(root_pose)
            self.robot.write_root_velocity_to_sim(root_vel)

            # joint trajectory 강제 적용. 그래픽 확인용이라 velocity는 0으로 둔다.
            self.robot.write_joint_state_to_sim(
                position=q_target,
                velocity=torch.zeros_like(q_target),
                joint_ids=self._joint_dof_idx,
            )

            # 다음 physics step target도 유지
            self.robot.set_joint_position_target(
                q_target,
                joint_ids=self._joint_dof_idx,
            )

            # 접촉/solver가 root를 밀어내도 다음 step 전에 다시 고정되도록 한 번 더 고정
            self.robot.write_root_pose_to_sim(root_pose)
            self.robot.write_root_velocity_to_sim(root_vel)

            return

        if self.cfg.control_mode == "teleport_replay":
            # ------------------------------------------------------------
            # DEBUG ONLY:
            # root를 초기 위치/자세에 고정하고,
            # joint만 trajectory 값으로 강제 세팅한다.
            # 기존 phase/reset 로직은 그대로 따른다.
            # ------------------------------------------------------------

            if not hasattr(self, "_teleport_root_pose_w"):
                root_pos = self._get_root_pos_w().detach().clone()
                root_quat = self._get_root_quat_w().detach().clone()
                self._teleport_root_pose_w = torch.cat([root_pos, root_quat], dim=-1)

            root_pose = self._teleport_root_pose_w.clone()

            # root velocity = [linear xyz, angular xyz]
            root_vel = torch.zeros((self._get_root_pos_w().shape[0], 6), dtype=torch.float32, device=self.device)

            # 1) root 고정
            self.robot.write_root_pose_to_sim(root_pose)
            self.robot.write_root_velocity_to_sim(root_vel)

            # 2) joint trajectory 강제 적용
            self.robot.write_joint_state_to_sim(
                position=q_target,
                velocity=q_zmp_vel,
                joint_ids=self._joint_dof_idx,
            )

            # 3) 다음 physics step target도 유지
            self.robot.set_joint_position_target(
                q_target,
                joint_ids=self._joint_dof_idx,
            )

            return

        self.robot.set_joint_position_target(
            q_target,
            joint_ids=self._joint_dof_idx,
        )

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

        root_pos_w = self._get_root_pos_w()
        num_envs = root_pos_w.shape[0]

        dt = float(getattr(self, "step_dt", self.cfg.sim.dt * self.cfg.decimation))

        delta_pos_w = root_pos_w - self._prev_root_pos_w
        root_vel_w = delta_pos_w / max(dt, 1.0e-6)

        delta_pos_xy = delta_pos_w.clone()
        delta_pos_xy[:, 2] = 0.0

        root_vel_xy = root_vel_w.clone()
        root_vel_xy[:, 2] = 0.0

        move_axis = self._world_move_axis.expand(num_envs, -1).clone()
        move_axis[:, 2] = 0.0
        move_axis = F.normalize(move_axis, dim=-1, eps=1e-6)

        forward_vel = torch.sum(root_vel_xy * move_axis, dim=-1)

        forward_vel_vec = forward_vel.unsqueeze(-1) * move_axis
        side_vel_vec = root_vel_xy - forward_vel_vec
        side_vel = torch.linalg.norm(side_vel_vec[:, 0:2], dim=-1)

        z_vel = root_vel_w[:, 2]
        torso_z = root_pos_w[:, 2]

        # --------------------------------------------------
        # reset 기준 전진/측면 누적 이동량
        # --------------------------------------------------
        # side_vel만 벌주면 천천히 옆으로 흘러가는 해법이 남을 수 있다.
        # 그래서 reset 위치 기준으로 lateral displacement도 같이 벌준다.
        root_delta_from_reset = root_pos_w - self._initial_root_pos_w
        root_delta_xy = root_delta_from_reset.clone()
        root_delta_xy[:, 2] = 0.0

        forward_disp = torch.sum(root_delta_xy * move_axis, dim=-1)
        forward_disp_vec = forward_disp.unsqueeze(-1) * move_axis
        side_disp_vec = root_delta_xy - forward_disp_vec
        side_disp = torch.linalg.norm(side_disp_vec[:, 0:2], dim=-1)

        # signed 값은 observation/debug용. 부호 기준은 move_axis의 좌측/우측 직교축이다.
        side_axis = torch.stack(
            [-move_axis[:, 1], move_axis[:, 0], torch.zeros_like(move_axis[:, 0])],
            dim=-1,
        )
        side_axis = F.normalize(side_axis, dim=-1, eps=1e-6)
        signed_side_disp = torch.sum(root_delta_xy * side_axis, dim=-1)

        side_pos_penalty = (
            self.cfg.side_pos_cost
            * torch.clamp(side_disp - self.cfg.side_pos_deadband, min=0.0) ** 2
        )

        # reset 이후 최고 전진거리에서 다시 뒤로 밀리는 정도.
        # detach된 buffer로 관리해서 reward 계산용 상태로만 사용한다.
        self._max_forward_disp[:] = torch.maximum(
            self._max_forward_disp,
            forward_disp.detach(),
        )
        max_forward_disp = self._max_forward_disp
        backtrack_dist = torch.clamp(
            max_forward_disp - forward_disp - self.cfg.backtrack_deadband,
            min=0.0,
        )

        # --------------------------------------------------
        # posture / health
        # --------------------------------------------------
        up_error = torch.clamp(1.0 - custom_up_proj, min=0.0)
        heading_error = torch.clamp(1.0 - custom_heading_proj, min=0.0)

        upright_reward = torch.exp(-((up_error / max(self.cfg.upright_sigma, 1.0e-6)) ** 2))
        heading_reward = torch.exp(-((heading_error / max(self.cfg.heading_sigma, 1.0e-6)) ** 2))

        # dot-product reward만으로는 yaw가 조금 틀어진 상태를 강하게 잡기 어렵다.
        # 실제 각도(rad)를 계산해서 heading penalty와 forward gate에 같이 사용한다.
        heading_angle_error = torch.acos(
            torch.clamp(custom_heading_proj, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
        )
        # currently used for logging / optional gating; not multiplied into forward_reward yet
        heading_forward_gate = torch.exp(
            -((heading_angle_error / max(self.cfg.heading_forward_gate_width, 1.0e-6)) ** 2)
        )

        height_reward = torch.exp(
            -(((torso_z - self.cfg.stand_target_height) / max(self.cfg.stand_height_sigma, 1.0e-6)) ** 2)
        )

        height_gate = torch.clamp(
            (torso_z - self.cfg.termination_height)
            / max(self.cfg.stand_target_height - self.cfg.termination_height, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        posture_gate = torch.clamp(
            (custom_up_proj - self.cfg.bad_posture_threshold)
            / max(1.0 - self.cfg.bad_posture_threshold, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        healthy_gate = height_gate * posture_gate

        # --------------------------------------------------
        # imitation
        # --------------------------------------------------
        leg_q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        leg_qd = self.robot.data.joint_vel[:, self._joint_dof_idx]

        # visual_replay에서는 _apply_action()에서 직접 증가시킨 trajectory target을 기준으로 reward/log를 맞춘다.
        if self.cfg.control_mode == "visual_replay":
            q_zmp = self._last_q_zmp
            q_zmp_vel = self._last_q_zmp_vel
        else:
            q_zmp, q_zmp_vel = self._get_zmp_joint_target()

        # residual 학습에서는 imitation 기준도 q_zmp 100%가 아니라,
        # 실제 target과 같은 방식으로 default pose와 q_zmp를 섞은 reference를 사용한다.
        # 발목은 alpha를 낮춰서 시뮬 접촉 안정성을 확보한다.
        if self.cfg.control_mode == "residual":
            alpha_vec = self._zmp_follow_alpha_vec.expand(q_zmp.shape[0], -1)
            q_imit_ref = (
                (1.0 - alpha_vec) * self._default_leg_q
                + alpha_vec * q_zmp
            )
            qd_imit_ref = alpha_vec * q_zmp_vel
            alpha_log = float(self.cfg.zmp_follow_alpha)
            ankle_alpha_log = float(self.cfg.ankle_zmp_follow_alpha)
        else:
            alpha_vec = None
            q_imit_ref = q_zmp
            qd_imit_ref = q_zmp_vel
            alpha_log = 1.0
            ankle_alpha_log = 1.0

        q_error = torch.mean((leg_q - q_imit_ref) ** 2, dim=-1)
        qd_error = torch.mean((leg_qd - qd_imit_ref) ** 2, dim=-1)

        imitation_pos_reward = torch.exp(
            -q_error / max(self.cfg.zmp_joint_sigma ** 2, 1.0e-6)
        )
        imitation_vel_reward = torch.exp(
            -qd_error / max(self.cfg.zmp_vel_sigma ** 2, 1.0e-6)
        )

        # 몸이 낮아지거나 기울어진 상태에서는 imitation reward를 강하게 주지 않는다.
        # 즉, "trajectory를 따라가는 것"보다 "안 넘어지는 것"을 먼저 배우게 한다.
        stable_imi_gate = torch.clamp(
            (custom_up_proj - self.cfg.stable_imi_up_threshold)
            / max(self.cfg.stable_imi_up_width, 1.0e-6),
            min=0.0,
            max=1.0,
        ) * torch.clamp(
            (torso_z - self.cfg.stable_imi_height_threshold)
            / max(self.cfg.stable_imi_height_width, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        imitation_reward = (
            self.cfg.imitation_pos_scale * imitation_pos_reward
            + self.cfg.imitation_vel_scale * imitation_vel_reward
        ) * stable_imi_gate

        # --------------------------------------------------
        # forward / movement shaping. 초기에는 scale 작게.
        # --------------------------------------------------
        forward_tracking_reward = torch.exp(
            -(
                (forward_vel - self.cfg.target_forward_vel)
                / max(self.cfg.forward_vel_sigma, 1.0e-6)
            ) ** 2
        )

        forward_alive_gate = torch.clamp(
            forward_vel / max(self.cfg.target_forward_vel, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        # 몸 방향이 틀어진 상태로 이동하면 전진 보상을 줄인다.
        forward_reward = forward_tracking_reward * forward_alive_gate * healthy_gate

        # reset 위치 기준으로 실제 전진거리가 누적될수록 보상한다.
        # forward_vel은 순간속도라 제자리 왕복/흔들림도 보상받을 수 있으므로,
        # forward_disp_reward로 "앞으로 간 상태를 유지하는 것"을 추가로 유도한다.
        forward_disp_raw_reward = (
            torch.clamp(
                forward_disp / max(self.cfg.forward_disp_target, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            * healthy_gate
        )

        # 옆으로 많이 새는 상태에서는 누적 전진 보상을 깎는다.
        # side_pos_penalty를 더 키우는 대신, 잘못된 전진 보상 자체를 줄이는 역할이다.
        side_forward_gate = torch.exp(
            -((side_disp / max(self.cfg.forward_disp_side_gate_width, 1.0e-6)) ** 2)
        )
        forward_disp_reward = forward_disp_raw_reward * side_forward_gate

        backward_vel = torch.clamp(-forward_vel, min=0.0, max=0.60)

        # healthy_gate가 0에 가까워지는 순간 penalty도 같이 사라지면,
        # 넘어지는 중의 나쁜 상태가 충분히 벌점 처리되지 않는다.
        penalty_gate = 0.25 + 0.75 * healthy_gate

        backward_penalty = self.cfg.backward_vel_cost * backward_vel * penalty_gate
        backtrack_penalty = self.cfg.backtrack_cost * backtrack_dist * penalty_gate
        heading_angle_penalty = (
            self.cfg.heading_angle_cost
            * torch.clamp(heading_angle_error - self.cfg.heading_angle_deadband, min=0.0) ** 2
            * penalty_gate
        )
        side_vel_penalty = self.cfg.side_vel_cost * side_vel * penalty_gate
        z_vel_penalty = self.cfg.z_vel_cost * torch.abs(z_vel) * penalty_gate

        low_height_penalty = (
            self.cfg.low_height_cost
            * torch.clamp(1.90 - torso_z, min=0.0) ** 2
        )

        tilt_penalty = (
            self.cfg.tilt_cost
            * torch.clamp(0.95 - custom_up_proj, min=0.0) ** 2
        )

        # --------------------------------------------------
        # regularization
        # --------------------------------------------------
        actions_cost = torch.mean(self.actions ** 2, dim=-1)
        action_rate_cost = torch.mean((self.actions - self._prev_actions) ** 2, dim=-1)
        action_saturation_cost = torch.mean(
            torch.clamp(torch.abs(self.actions) - self.cfg.action_soft_limit, min=0.0) ** 2,
            dim=-1,
        )

        # residual action이 너무 큰 target shift를 계속 만드는 것을 억제.
        residual_target_cost = torch.mean(
            ((self._last_q_target - q_imit_ref) / max(float(self.cfg.action_scale), 1.0e-6)) ** 2,
            dim=-1,
        )

        joint_vel_cost = torch.mean(leg_qd ** 2, dim=-1)

        joint_low_violation = torch.clamp(
            (self._joint_lower_limits + self.cfg.joint_limit_margin) - leg_q,
            min=0.0,
        )
        joint_high_violation = torch.clamp(
            leg_q - (self._joint_upper_limits - self.cfg.joint_limit_margin),
            min=0.0,
        )
        joint_limit_cost = torch.mean(joint_low_violation ** 2 + joint_high_violation ** 2, dim=-1)

        total_reward = (
            self.cfg.alive_scale * healthy_gate
            + imitation_reward
            + self.cfg.upright_scale * upright_reward * height_gate
            + self.cfg.height_scale * height_reward * posture_gate
            + self.cfg.heading_scale * heading_reward * healthy_gate
            + self.cfg.forward_scale * forward_reward
            + self.cfg.forward_disp_scale * forward_disp_reward
            - backward_penalty
            - backtrack_penalty
            - heading_angle_penalty
            - side_vel_penalty
            - z_vel_penalty
            - side_pos_penalty
            - low_height_penalty
            - tilt_penalty
            - self.cfg.action_cost_scale * actions_cost
            - self.cfg.action_rate_cost_scale * action_rate_cost
            - self.cfg.action_saturation_cost_scale * action_saturation_cost
            - self.cfg.residual_target_cost_scale * residual_target_cost
            - self.cfg.joint_vel_cost_scale * joint_vel_cost
            - self.cfg.joint_limit_cost_scale * joint_limit_cost
        )

        # --------------------------------------------------
        # done reward override
        # --------------------------------------------------
        bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
        died = (torso_z < self.cfg.termination_height) | bad_posture

        total_reward = torch.where(
            died,
            torch.ones_like(total_reward) * self.cfg.death_cost,
            total_reward,
        )

        # --------------------------------------------------
        # debug
        # --------------------------------------------------
        if self.common_step_counter % self._debug_every == 0:
            env_id = self._debug_env_id

            logger.info(
                f"[MOVE-IMI][env{env_id}] "
                f"root_pos={root_pos_w[env_id].detach().cpu().tolist()}, "
                f"delta_pos={delta_pos_w[env_id].detach().cpu().tolist()}, "
                f"move_axis={self._world_move_axis[0].detach().cpu().tolist()}, "
                f"forward_vel={forward_vel[env_id].item():.4f}, "
                f"side_vel={side_vel[env_id].item():.4f}, "
                f"z_vel={z_vel[env_id].item():.4f}, "
                f"forward_disp={forward_disp[env_id].item():.4f}, "
                f"max_forward_disp={max_forward_disp[env_id].item():.4f}, "
                f"backtrack_dist={backtrack_dist[env_id].item():.4f}, "
                f"side_disp={side_disp[env_id].item():.4f}, "
                f"signed_side_disp={signed_side_disp[env_id].item():+.4f}"
            )

            logger.info(
                f"[POSTURE-IMI][env{env_id}] "
                f"up_proj={custom_up_proj[env_id].item():.4f}, "
                f"heading_proj={custom_heading_proj[env_id].item():.4f}, "
                f"heading_angle_deg={torch.rad2deg(heading_angle_error[env_id]).item():.2f}, "
                f"heading_forward_gate={heading_forward_gate[env_id].item():.4f}, "
                f"torso_z={torso_z[env_id].item():.4f}, "
                f"height_gate={height_gate[env_id].item():.4f}, "
                f"posture_gate={posture_gate[env_id].item():.4f}, "
                f"healthy_gate={healthy_gate[env_id].item():.4f}"
            )

            # 전체 env 기준 이동 통계.
            # env0 하나만 보면 play 느낌과 로그 판단이 어긋날 수 있어서,
            # forward 성공률 / lateral drift 비율을 같이 본다.
            forward_disp_pos_ratio = (forward_disp > 0.0).float().mean()
            forward_good_ratio = ((forward_disp > 0.15) & (side_disp < 0.15)).float().mean()
            backward_episode_ratio = (forward_disp < 0.0).float().mean()
            lateral_bad_ratio = (side_disp > 0.25).float().mean()
            side_bigger_ratio = (side_disp > torch.abs(forward_disp)).float().mean()
            forward_vel_pos_ratio = (forward_vel > 0.0).float().mean()
            backtrack_ratio = (backtrack_dist > 0.0).float().mean()
            heading_bad_ratio = (heading_angle_error > self.cfg.heading_angle_deadband).float().mean()

            logger.info(
                "[MOVE-STATS] "
                f"forward_vel_mean={forward_vel.mean().item():+.4f}, "
                f"forward_vel_p50={torch.quantile(forward_vel, 0.50).item():+.4f}, "
                f"forward_vel_pos_ratio={forward_vel_pos_ratio.item():.3f}, "
                f"forward_disp_mean={forward_disp.mean().item():+.4f}, "
                f"forward_disp_p50={torch.quantile(forward_disp, 0.50).item():+.4f}, "
                f"forward_disp_max={forward_disp.max().item():+.4f}, "
                f"forward_disp_pos_ratio={forward_disp_pos_ratio.item():.3f}, "
                f"forward_good_ratio={forward_good_ratio.item():.3f}, "
                f"backward_episode_ratio={backward_episode_ratio.item():.3f}, "
                f"backtrack_mean={backtrack_dist.mean().item():.4f}, "
                f"backtrack_ratio={backtrack_ratio.item():.3f}, "
                f"heading_angle_deg_mean={torch.rad2deg(heading_angle_error).mean().item():.2f}, "
                f"heading_angle_deg_p50={torch.rad2deg(torch.quantile(heading_angle_error, 0.50)).item():.2f}, "
                f"heading_bad_ratio={heading_bad_ratio.item():.3f}, "
                f"heading_forward_gate_mean={heading_forward_gate.mean().item():.4f}, "
                f"side_gate_mean={side_forward_gate.mean().item():.4f}, "
                f"side_disp_mean={side_disp.mean().item():.4f}, "
                f"side_disp_p50={torch.quantile(side_disp, 0.50).item():.4f}, "
                f"side_disp_max={side_disp.max().item():.4f}, "
                f"lateral_bad_ratio={lateral_bad_ratio.item():.3f}, "
                f"side_bigger_than_forward_ratio={side_bigger_ratio.item():.3f}"
            )

            logger.info(
                f"[REWARD-IMI] "
                f"total={total_reward.mean().item():.4f}, "
                f"imi_pos={imitation_pos_reward.mean().item():.4f}, "
                f"imi_vel={imitation_vel_reward.mean().item():.4f}, "
                f"zmp_follow_alpha={alpha_log:.3f}, "
                f"ankle_alpha={ankle_alpha_log:.3f}, "
                f"stable_imi_gate={stable_imi_gate.mean().item():.4f}, "
                f"imitation_reward={imitation_reward.mean().item():.4f}, "
                f"upright={upright_reward.mean().item():.4f}, "
                f"height={height_reward.mean().item():.4f}, "
                f"heading={heading_reward.mean().item():.4f}, "
                f"heading_angle_penalty={heading_angle_penalty.mean().item():.4f}, "
                f"heading_forward_gate={heading_forward_gate.mean().item():.4f}, "
                f"forward={forward_reward.mean().item():.4f}, "
                f"forward_disp_raw={forward_disp_raw_reward.mean().item():.4f}, "
                f"side_forward_gate={side_forward_gate.mean().item():.4f}, "
                f"forward_disp_reward={forward_disp_reward.mean().item():.4f}, "
                f"backward_penalty={backward_penalty.mean().item():.4f}, "
                f"backtrack_penalty={backtrack_penalty.mean().item():.4f}, "
                f"side_vel_penalty={side_vel_penalty.mean().item():.4f}, "
                f"z_vel_penalty={z_vel_penalty.mean().item():.4f}, "
                f"side_pos_penalty={side_pos_penalty.mean().item():.4f}, "
                f"low_height_penalty={low_height_penalty.mean().item():.4f}, "
                f"tilt_penalty={tilt_penalty.mean().item():.4f}, "
                f"action_cost={actions_cost.mean().item():.4f}, "
                f"action_rate_cost={action_rate_cost.mean().item():.4f}, "
                f"action_saturation_cost={action_saturation_cost.mean().item():.4f}, "
                f"residual_target_cost={residual_target_cost.mean().item():.4f}, "
                f"joint_vel_cost={joint_vel_cost.mean().item():.4f}, "
                f"joint_limit_cost={joint_limit_cost.mean().item():.4f}, "
                f"q_error={q_error.mean().item():.4f}, "
                f"qd_error={qd_error.mean().item():.4f}, "
                f"died_ratio={died.float().mean().item():.4f}"
            )

        self._prev_root_pos_w[:] = root_pos_w.detach().clone()
        self._prev_actions[:] = self.actions.detach().clone()

        return total_reward

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        custom_up_proj, _ = self._get_custom_axis_proj()

        root_z = self._get_root_pos_w()[:, 2]

        # visual_replay는 trajectory를 그래픽으로 확인하는 모드라 reset을 막는다.
        # common_step_counter 기반으로 trajectory index를 계속 증가시키기 때문에,
        # died/time_out이 걸리면 모션이 계속 초반으로 돌아가 보인다.
        if self.cfg.control_mode == "visual_replay":
            terminated = torch.zeros(root_z.shape[0], dtype=torch.bool, device=self.device)
            time_out = torch.zeros_like(terminated)

            if self.common_step_counter % self._debug_every == 0:
                logger.info(
                    f"[DONE-IMI][VISUAL] reset disabled | "
                    f"root_z_mean={root_z.mean().item():.4f}, "
                    f"up_proj_mean={custom_up_proj.mean().item():.4f}"
                )

            return terminated, time_out

        bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
        died = (root_z < self.cfg.termination_height) | bad_posture

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.common_step_counter % self._debug_every == 0:
            logger.info(
                f"[DONE-IMI] "
                f"root_z_mean={root_z.mean().item():.4f}, "
                f"up_proj_mean={custom_up_proj.mean().item():.4f}, "
                f"died_ratio={died.float().mean().item():.4f}, "
                f"timeout_ratio={time_out.float().mean().item():.4f}"
            )

        return died, time_out

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """
        22축 imitation용 custom observation.

        기존 parent LocomotionEnv observation은 action_space / joint 개수 변경 시
        shape가 헷갈릴 수 있어서 여기서는 직접 policy observation을 만든다.

        obs 구성, 총 97차원:
          - torso_z: 1
          - reset 기준 forward_disp, signed_side_disp: 2
          - root_lin_vel_w: 3
          - root_ang_vel_w: 3
          - custom_up_proj, custom_heading_proj: 2
          - controlled joint pos: 21
          - controlled joint vel: 21
          - blended q reference: 21
          - previous action: 21
          - phase sin/cos: 2
        """
        root_pos_w = self._get_root_pos_w()
        root_lin_vel_w = self._get_root_lin_vel_w()
        root_ang_vel_w = self._get_root_ang_vel_w()
        custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

        num_envs = root_pos_w.shape[0]
        move_axis = self._world_move_axis.expand(num_envs, -1).clone()
        move_axis[:, 2] = 0.0
        move_axis = F.normalize(move_axis, dim=-1, eps=1e-6)

        root_delta_from_reset = root_pos_w - self._initial_root_pos_w
        root_delta_xy = root_delta_from_reset.clone()
        root_delta_xy[:, 2] = 0.0
        forward_disp = torch.sum(root_delta_xy * move_axis, dim=-1)

        side_axis = torch.stack(
            [-move_axis[:, 1], move_axis[:, 0], torch.zeros_like(move_axis[:, 0])],
            dim=-1,
        )
        side_axis = F.normalize(side_axis, dim=-1, eps=1e-6)
        signed_side_disp = torch.sum(root_delta_xy * side_axis, dim=-1)

        q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        qd = self.robot.data.joint_vel[:, self._joint_dof_idx]
        if self.cfg.control_mode == "visual_replay":
            q_zmp = self._last_q_zmp
        else:
            q_zmp, _ = self._get_zmp_joint_target()

        # policy observation에도 실제로 따라갈 blended reference를 넣는다.
        # full q_zmp를 넣으면 제어 target은 약한데 observation은 강한 trajectory라 헷갈릴 수 있다.
        if self.cfg.control_mode == "residual":
            alpha = max(0.0, min(1.0, float(self.cfg.zmp_follow_alpha)))
            q_ref_obs = (
                (1.0 - alpha) * self._default_leg_q
                + alpha * q_zmp
            )
        else:
            q_ref_obs = q_zmp

        phase_obs = self._get_phase_obs()

        obs = torch.cat(
            [
                root_pos_w[:, 2:3],
                forward_disp.unsqueeze(-1),
                signed_side_disp.unsqueeze(-1),
                root_lin_vel_w,
                root_ang_vel_w * self.cfg.angular_velocity_scale,
                custom_up_proj.unsqueeze(-1),
                custom_heading_proj.unsqueeze(-1),
                q,
                qd * self.cfg.dof_vel_scale,
                q_ref_obs,
                self._prev_actions,
                phase_obs,
            ],
            dim=-1,
        )

        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"[LIKU-IMI] observation dim mismatch: obs={obs.shape[-1]}, "
                f"cfg.observation_space={self.cfg.observation_space}"
            )

        if self.common_step_counter == 0:
            logger.info(f"[LIKU-IMI][OBS] policy: shape={tuple(obs.shape)}")

        return {"policy": obs}