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
            effort_limit_sim=400000.0,
            velocity_limit_sim=300.0,
            # stiffness=60000.0,
            # damping=6000.0,
            stiffness=600000.0,
            damping=60000.0,


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
            effort_limit_sim=3000.0,
            velocity_limit_sim=60.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------

@configclass
class LikuImitationEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2

    # action 의미:
    #   최종 joint position target = q_zmp + action_scale * policy_action
    # 단위: rad
    # 처음에는 0.05~0.15 정도가 안전함.
    action_scale = 0.08

    action_space = 21

    # custom observation: root(8) + q(21) + qd(21) + q_zmp(21) + prev_action(21) + phase(2) + height(1) = 95
    observation_space = 95
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
    # control_mode: str = "residual"
    # control_mode: str = "replay"
    # control_mode: str = "hold_current"
    control_mode: str = "teleport_replay"

    # ZMP trajectory 파일 경로.
    # 비워두면 q_zmp = 0으로 동작함.
    # 지원 형식:
    #   1) .pt / .pth: Tensor [T, 22] 또는 dict 안의 "q", "joint_pos", "positions", "trajectory"
    #   2) .csv: header 있어도 됨. [T, 22] 또는 첫 column phase/time + [T, 22]
    zmp_traj_path: str = "D:/IsaacLab/traj/traj_headfix_60hz.pt"
    zmp_traj_in_degrees: bool = False
    zmp_cycle_time: float = 160
    randomize_initial_phase: bool = False

    # phase를 observation에 sin/cos로 추가할지 여부.
    # True면 custom observation에 phase sin/cos 2개가 포함됨.
    use_phase_observation: bool = True

    # -------------------------------------------------------------------------
    # termination / posture
    # -------------------------------------------------------------------------
    death_cost: float = -6.0
    termination_height: float = 1.80
    bad_posture_threshold: float = 0.85

    stand_target_height: float = 2.01
    stand_height_sigma: float = 0.05
    upright_sigma: float = 0.20
    heading_sigma: float = 0.45

    # -------------------------------------------------------------------------
    # reward scales
    # -------------------------------------------------------------------------
    alive_scale: float = 0.10
    imitation_pos_scale: float = 2.00
    imitation_vel_scale: float = 0.15
    upright_scale: float = 0.80
    height_scale: float = 0.80
    heading_scale: float = 0.10

    # 처음에는 forward_scale을 작게 두는 게 좋음.
    # ZMP replay/residual이 안 넘어지는 걸 먼저 확인한 뒤 키우기.
    forward_scale: float = 0.10
    target_forward_vel: float = 0.10
    forward_vel_sigma: float = 0.08
    backward_vel_cost: float = 1.00
    side_vel_cost: float = 0.30
    z_vel_cost: float = 0.30

    # imitation reward sigma
    # pos는 rad 기준, vel은 rad/s 기준.
    zmp_joint_sigma: float = 0.35
    zmp_vel_sigma: float = 1.50

    # regularization
    action_cost_scale: float = 0.020
    action_rate_cost_scale: float = 0.080
    residual_target_cost_scale: float = 0.020
    joint_vel_cost_scale: float = 0.002
    joint_limit_cost_scale: float = 0.50
    joint_limit_margin: float = 0.05

    # debug
    debug_every: int = 60

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

        self._prev_actions = torch.zeros(
            num_envs,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )

        # D13은 head hold joint라 policy residual을 적용하지 않는다.
        self._residual_action_mask = torch.ones(
            1,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )
        for _fixed_name in RESIDUAL_FIXED_JOINTS:
            if _fixed_name in self._joint_names:
                self._residual_action_mask[0, self._joint_names.index(_fixed_name)] = 0.0

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
            f"degrees={self.cfg.zmp_traj_in_degrees}"
        )

    def _get_phase(self) -> torch.Tensor:
        dt = float(getattr(self, "step_dt", self.cfg.sim.dt * self.cfg.decimation))
        cycle_time = max(float(self.cfg.zmp_cycle_time), 1.0e-6)

        phase = (
            self.episode_length_buf.float()
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

    def _apply_action(self):
        leg_q = self.robot.data.joint_pos[:, self._joint_dof_idx]

        q_zmp, q_zmp_vel = self._get_zmp_joint_target()
        q_residual = float(self.cfg.action_scale) * self.actions * self._residual_action_mask

        if self.cfg.control_mode == "residual":
            # 핵심 모드: ZMP trajectory를 기본값으로 하고 policy는 보정값만 출력.
            q_target = q_zmp + q_residual

        elif self.cfg.control_mode == "replay":
            # ZMP trajectory 자체가 IsaacSim에서 말이 되는지 검증할 때 사용.
            q_target = q_zmp

        elif self.cfg.control_mode == "teleport_replay":
            # 디버그용:
            # actuator 힘으로 따라가는 게 아니라 joint state를 trajectory 값으로 직접 써버린다.
            # 이 모드는 "trajectory 매칭/부호/순서가 맞는지" 확인하는 용도.
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
                "Use 'residual', 'replay', 'teleport_replay', 'direct', or 'hold_current'."
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
        if self.common_step_counter % self._debug_every == 0:
            env_id = self._debug_env_id
            q = leg_q[env_id].detach().cpu()
            qd = self.robot.data.joint_vel[env_id, self._joint_dof_idx].detach().cpu()
            qz = q_zmp[env_id].detach().cpu()
            qt = q_target[env_id].detach().cpu()
            qe = q - qt

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
        if self.cfg.control_mode == "teleport_replay":
            # 처음 root pose를 저장해두고 계속 그 위치/자세로 고정
            if not hasattr(self, "_teleport_root_pose_w"):
                root_pos = self.robot.data.root_pos_w.detach().clone()
                root_quat = self.robot.data.root_quat_w.detach().clone()
                self._teleport_root_pose_w = torch.cat([root_pos, root_quat], dim=-1)

            root_pose = self._teleport_root_pose_w.clone()
            root_vel = torch.zeros_like(self.robot.data.root_vel_w)

            # 1) root 고정
            self.robot.write_root_pose_to_sim(root_pose)
            self.robot.write_root_velocity_to_sim(root_vel)

            # 2) joint trajectory 강제 적용
            self.robot.write_joint_state_to_sim(
                position=q_target,
                velocity=q_zmp_vel,
                joint_ids=self._joint_dof_idx,
            )

            # 3) 다음 physics step에도 target 유지
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
        # posture / health
        # --------------------------------------------------
        up_error = torch.clamp(1.0 - custom_up_proj, min=0.0)
        heading_error = torch.clamp(1.0 - custom_heading_proj, min=0.0)

        upright_reward = torch.exp(-((up_error / max(self.cfg.upright_sigma, 1.0e-6)) ** 2))
        heading_reward = torch.exp(-((heading_error / max(self.cfg.heading_sigma, 1.0e-6)) ** 2))

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

        q_zmp, q_zmp_vel = self._get_zmp_joint_target()

        q_error = torch.mean((leg_q - q_zmp) ** 2, dim=-1)
        qd_error = torch.mean((leg_qd - q_zmp_vel) ** 2, dim=-1)

        imitation_pos_reward = torch.exp(
            -q_error / max(self.cfg.zmp_joint_sigma ** 2, 1.0e-6)
        )
        imitation_vel_reward = torch.exp(
            -qd_error / max(self.cfg.zmp_vel_sigma ** 2, 1.0e-6)
        )

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

        forward_reward = forward_tracking_reward * forward_alive_gate * healthy_gate

        backward_vel = torch.clamp(-forward_vel, min=0.0, max=0.60)
        backward_penalty = self.cfg.backward_vel_cost * backward_vel * healthy_gate
        side_vel_penalty = self.cfg.side_vel_cost * side_vel * healthy_gate
        z_vel_penalty = self.cfg.z_vel_cost * torch.abs(z_vel) * healthy_gate

        # --------------------------------------------------
        # regularization
        # --------------------------------------------------
        actions_cost = torch.mean(self.actions ** 2, dim=-1)
        action_rate_cost = torch.mean((self.actions - self._prev_actions) ** 2, dim=-1)

        # residual action이 너무 큰 target shift를 계속 만드는 것을 억제.
        residual_target_cost = torch.mean(
            ((self._last_q_target - q_zmp) / max(float(self.cfg.action_scale), 1.0e-6)) ** 2,
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
            + self.cfg.imitation_pos_scale * imitation_pos_reward * healthy_gate
            + self.cfg.imitation_vel_scale * imitation_vel_reward * healthy_gate
            + self.cfg.upright_scale * upright_reward * height_gate
            + self.cfg.height_scale * height_reward * posture_gate
            + self.cfg.heading_scale * heading_reward * healthy_gate
            + self.cfg.forward_scale * forward_reward
            - backward_penalty
            - side_vel_penalty
            - z_vel_penalty
            - self.cfg.action_cost_scale * actions_cost
            - self.cfg.action_rate_cost_scale * action_rate_cost
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
                f"z_vel={z_vel[env_id].item():.4f}"
            )

            logger.info(
                f"[POSTURE-IMI][env{env_id}] "
                f"up_proj={custom_up_proj[env_id].item():.4f}, "
                f"heading_proj={custom_heading_proj[env_id].item():.4f}, "
                f"torso_z={torso_z[env_id].item():.4f}, "
                f"height_gate={height_gate[env_id].item():.4f}, "
                f"posture_gate={posture_gate[env_id].item():.4f}, "
                f"healthy_gate={healthy_gate[env_id].item():.4f}"
            )

            logger.info(
                f"[REWARD-IMI] "
                f"total={total_reward.mean().item():.4f}, "
                f"imi_pos={imitation_pos_reward.mean().item():.4f}, "
                f"imi_vel={imitation_vel_reward.mean().item():.4f}, "
                f"upright={upright_reward.mean().item():.4f}, "
                f"height={height_reward.mean().item():.4f}, "
                f"heading={heading_reward.mean().item():.4f}, "
                f"forward={forward_reward.mean().item():.4f}, "
                f"backward_penalty={backward_penalty.mean().item():.4f}, "
                f"side_vel_penalty={side_vel_penalty.mean().item():.4f}, "
                f"z_vel_penalty={z_vel_penalty.mean().item():.4f}, "
                f"action_cost={actions_cost.mean().item():.4f}, "
                f"action_rate_cost={action_rate_cost.mean().item():.4f}, "
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

        obs 구성, 총 95차원:
          - torso_z: 1
          - root_lin_vel_w: 3
          - root_ang_vel_w: 3
          - custom_up_proj, custom_heading_proj: 2
          - controlled joint pos: 21
          - controlled joint vel: 21
          - q_zmp reference: 21
          - previous action: 21
          - phase sin/cos: 2
        """
        root_pos_w = self._get_root_pos_w()
        root_lin_vel_w = self._get_root_lin_vel_w()
        root_ang_vel_w = self._get_root_ang_vel_w()
        custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

        q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        qd = self.robot.data.joint_vel[:, self._joint_dof_idx]
        q_zmp, _ = self._get_zmp_joint_target()
        phase_obs = self._get_phase_obs()

        obs = torch.cat(
            [
                root_pos_w[:, 2:3],
                root_lin_vel_w,
                root_ang_vel_w * self.cfg.angular_velocity_scale,
                custom_up_proj.unsqueeze(-1),
                custom_heading_proj.unsqueeze(-1),
                q,
                qd * self.cfg.dof_vel_scale,
                q_zmp,
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
