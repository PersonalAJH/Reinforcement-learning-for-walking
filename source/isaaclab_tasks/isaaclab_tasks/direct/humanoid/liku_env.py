from __future__ import annotations

import logging
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

LIKU_USD_PATH = "D:/isaacsim/jhusd/new_test/test2_6joints.usd"

CONTROLLED_LEG_JOINTS = ["A3", "A4", "A5", "B9", "B10", "B11"]


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


LIKU_CFG = ArticulationCfg(
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
            "A3": 0.001,
            "A4": -0.001,
            "A5": 0.001,
            "B9": 0.001,
            "B10": -0.001,
            "B11": 0.001,

            "D13": 0.001,
            "E15": 0.001,
            "E16": -0.001,
            "E17": 0.001,
            "E18": -0.001,
            "F19": 0.001,
            "F20": 0.001,
            "F21": 0.001,
            "F22": -0.001,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=CONTROLLED_LEG_JOINTS,
            effort_limit_sim=20000.0,
            velocity_limit_sim=200.0,

            # torque control처럼 사용
            stiffness=0.0,
            damping=3.0,
        ),
    },
)


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------

@configclass
class LikuEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    action_space = 6
    observation_space = 48
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

    robot: ArticulationCfg = LIKU_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -------------------------------------------------------------------------
    # training stage
    # -------------------------------------------------------------------------
    # 1단계: "stand"   - 제자리 서기 안정화
    # 2단계: "onestep" - 한 발 전진 + 안정적으로 버티기
    # 3단계: "walk"    - 연속 전진
    training_stage: str = "stand"

    # -------------------------------------------------------------------------
    # action / actuator
    # -------------------------------------------------------------------------
    joint_gears: list = [10000.0] * 6

    # -------------------------------------------------------------------------
    # reward / penalty design
    # -------------------------------------------------------------------------
    death_cost: float = -6.0

    # stage별 termination
    # stand에서는 낮게 주저앉는 동작을 빨리 끊고,
    # walk에서는 탐색을 위해 조금 더 관대하게 둔다.
    stand_termination_height: float = 1.82
    onestep_termination_height: float = 1.80
    walk_termination_height: float = 1.80
    bad_posture_threshold: float = 0.85

    # standing target
    stand_target_height: float = 2.01
    stand_height_sigma: float = 0.05
    stand_height_deadband: float = 0.03

    # posture shaping
    upright_sigma: float = 0.20
    heading_sigma: float = 0.45

    # stand reward scales
    stand_alive_scale: float = 0.20
    stand_upright_scale: float = 1.20
    stand_height_scale: float = 1.50
    stand_heading_scale: float = 0.20
    stand_xy_vel_cost: float = 1.00
    stand_z_vel_cost: float = 1.00
    stand_xy_pos_cost: float = 0.80
    stand_height_pos_cost: float = 12.00
    stand_joint_pos_cost: float = 0.08

    # walk reward scales
    walk_alive_scale: float = 0.04
    walk_upright_scale: float = 0.15
    walk_height_scale: float = 0.08
    walk_heading_scale: float = 0.05

    # target velocity tracking 방식이므로 scale 의미가 기존 선형 velocity reward와 다름
    walk_forward_vel_scale: float = 2.50
    walk_target_forward_vel: float = 0.12
    walk_forward_vel_sigma: float = 0.08
    walk_forward_speed_limit: float = 0.22
    walk_forward_speed_spike_cost: float = 1.50
    walk_backward_vel_cost: float = 3.50
    walk_side_vel_cost: float = 0.80
    walk_z_vel_cost: float = 0.80

    # onestep stage: 한 발 전진을 명시적으로 학습시키는 중간 커리큘럼
    onestep_target_forward_dist: float = 0.08
    onestep_success_forward_dist: float = 0.06
    onestep_dist_sigma: float = 0.04
    onestep_record_reward_scale: float = 500.0
    onestep_target_reward_scale: float = 0.80
    onestep_success_bonus_scale: float = 0.35
    onestep_hold_vel_cost: float = 0.50
    onestep_overshoot_cost: float = 2.00

    # walk stage: 한 발 후 멈추는 local optimum 방지
    forward_record_reward_scale: float = 1000.0
    forward_record_delta_clip: float = 0.004
    min_forward_vel: float = 0.04
    no_progress_cost_scale: float = 0.20
    episode_side_cost_scale: float = 0.35

    # regularization
    action_cost_scale: float = 0.050
    action_rate_cost_scale: float = 0.22
    action_saturation_cost_scale: float = 0.40
    action_soft_limit: float = 0.70
    joint_vel_cost_scale: float = 0.055
    joint_limit_cost_scale: float = 1.50
    # controlled leg joint angle limit: +/-30 deg = +/-0.5236 rad
    joint_limit_soft: float = 0.7853981634
    joint_limit_hard: float = 0.7853981634

    # walk stage warmup
    walk_warmup_steps: float = 60.0

    # LHIP/RHIP 명칭이어도 실제로는 foot body로 취급
    foot_body_names = ["Hip6", "LHip6"]

    # foot-step auxiliary reward
    # root 기준으로 한쪽 foot body가 reset 시점보다 앞으로 나가면 보조 reward를 준다.
    # 단, 발만 과하게 뻗는 꼼수를 막기 위해 overstep penalty를 같이 둔다.
    foot_step_target: float = 0.04
    foot_step_sigma: float = 0.035
    foot_step_reward_scale: float = 0.4
    foot_overstep_limit: float = 0.12
    foot_overstep_cost: float = 1.00

    # walk 단계용 step-separation auxiliary reward
    # reset 기준 foot 전진량이 아니라,
    # 현재 root 기준으로 양발이 앞뒤로 적당히 벌어졌는지를 본다.
    # 즉, 제자리에서 양발이 같이 흔들리는 해법보다 한 발 앞/한 발 뒤 패턴을 유도한다.
    step_separation_target: float = 0.10
    step_separation_sigma: float = 0.04
    step_separation_reward_scale: float = 0.25
    step_separation_over_limit: float = 0.18
    step_separation_over_cost: float = 1.00

    # -------------------------------------------------------------------------
    # CUSTOM AXIS
    # -------------------------------------------------------------------------
    robot_up_axis_local = (0.0, 1.0, 0.0)
    robot_forward_axis_local = (-1.0, 0.0, 0.0)

    world_up_axis = (0.0, 0.0, 1.0)

    # 보는 방향 기준
    world_forward_axis = (1.0, 0.0, 0.0)

    # 실제 이동 보상 기준
    # root가 world -X로 이동하면 forward reward
    world_move_axis = (-1.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------

class LikuEnv(LocomotionEnv):
    cfg: LikuEnvCfg

    def __init__(self, cfg: LikuEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug_every = 60
        self._debug_env_id = 0
        self._printed_initial_axis = False

        # ---------------------------------------------------------------------
        # controlled joints
        # ---------------------------------------------------------------------
        joint_ids = []
        joint_names = []

        for name in CONTROLLED_LEG_JOINTS:
            ids, names = self.robot.find_joints(name)

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(names, tuple):
                names = list(names)

            if len(ids) == 0:
                raise RuntimeError(f"[LIKU] joint '{name}' not found in robot articulation.")

            joint_ids.extend(ids)
            joint_names.extend(names)

        self._joint_dof_idx = joint_ids
        self._joint_names = joint_names

        if len(self._joint_dof_idx) != self.cfg.action_space:
            raise RuntimeError(
                f"[LIKU] controlled joint count ({len(self._joint_dof_idx)}) "
                f"!= action_space ({self.cfg.action_space})"
            )

        # ---------------------------------------------------------------------
        # foot bodies
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
                raise RuntimeError(f"[LIKU] foot body '{name}' not found in robot bodies.")

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

        num_envs = self._get_root_pos_w().shape[0]

        self._prev_root_pos_w = self._get_root_pos_w().detach().clone()

        # stand 단계에서 천천히 밀려나거나 주저앉는 꼼수를 막기 위해
        # reset 시점의 root 위치를 저장한다.
        self._initial_root_pos_w = self._get_root_pos_w().detach().clone()

        self._prev_actions = torch.zeros(
            num_envs,
            self.cfg.action_space,
            dtype=torch.float32,
            device=self.device,
        )

        # episode 누적 이동거리 로그용
        self._episode_forward_dist = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_side_dist = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # episode 내 최고 전진거리 기록
        self._episode_forward_best = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # reset 시점의 발 방향 저장
        self._initial_foot_quat_w = self._get_foot_quat_w().detach().clone()

        # reset 시점의 root 기준 발 위치 저장
        self._initial_foot_rel_w = (
            self._get_foot_pos_w()
            - self._get_root_pos_w().unsqueeze(1)
        ).detach().clone()

        # reset 시점의 발 world z 저장
        self._initial_foot_z_w = self._get_foot_pos_w()[:, :, 2].detach().clone()

        # reset 시점의 root-foot z 거리 저장
        self._initial_root_to_foot_z = (
            self._get_root_pos_w()[:, 2].unsqueeze(1)
            - self._get_foot_pos_w()[:, :, 2]
        ).detach().clone()

        logger.info(f"[LIKU] training_stage = {self.cfg.training_stage}")
        logger.info(f"[LIKU] controlled joints = {self._joint_names}")
        logger.info(f"[LIKU] controlled joint ids = {self._joint_dof_idx}")
        logger.info(f"[LIKU] action_space = {self.cfg.action_space}")
        logger.info(f"[LIKU] observation_space = {self.cfg.observation_space}")
        logger.info(f"[LIKU] joint_gears = {self.joint_gears}")

        logger.info(f"[LIKU] foot body names = {self._foot_body_names}")
        logger.info(f"[LIKU] foot body ids = {self._foot_body_ids}")

        if hasattr(self.robot, "body_names"):
            logger.info(f"[LIKU] all body names = {self.robot.body_names}")

        logger.info(f"[LIKU] robot_up_axis_local = {self.cfg.robot_up_axis_local}")
        logger.info(f"[LIKU] robot_forward_axis_local = {self.cfg.robot_forward_axis_local}")
        logger.info(f"[LIKU] world_forward_axis(heading) = {self.cfg.world_forward_axis}")
        logger.info(f"[LIKU] world_move_axis(reward) = {self.cfg.world_move_axis}")

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

    def _get_foot_quat_w(self) -> torch.Tensor:
        """
        return: [num_envs, num_feet, 4]
        foot body quaternion in world frame, wxyz
        """
        if hasattr(self.robot.data, "body_quat_w"):
            return self.robot.data.body_quat_w[:, self._foot_body_ids, :]

        return self.robot.data.body_state_w[:, self._foot_body_ids, 3:7]

    def _get_foot_pos_w(self) -> torch.Tensor:
        """
        return: [num_envs, num_feet, 3]
        foot body position in world frame
        """
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

        if hasattr(self, "_episode_forward_dist"):
            self._episode_forward_dist[env_ids] = 0.0

        if hasattr(self, "_episode_side_dist"):
            self._episode_side_dist[env_ids] = 0.0

        if hasattr(self, "_episode_forward_best"):
            self._episode_forward_best[env_ids] = 0.0

        if hasattr(self, "_initial_foot_quat_w"):
            self._initial_foot_quat_w[env_ids] = self._get_foot_quat_w()[env_ids].detach().clone()

        if hasattr(self, "_initial_foot_rel_w"):
            self._initial_foot_rel_w[env_ids] = (
                self._get_foot_pos_w()[env_ids]
                - self._get_root_pos_w()[env_ids].unsqueeze(1)
            ).detach().clone()

        if hasattr(self, "_initial_foot_z_w"):
            self._initial_foot_z_w[env_ids] = (
                self._get_foot_pos_w()[env_ids, :, 2]
            ).detach().clone()

        if hasattr(self, "_initial_root_to_foot_z"):
            self._initial_root_to_foot_z[env_ids] = (
                self._get_root_pos_w()[env_ids, 2].unsqueeze(1)
                - self._get_foot_pos_w()[env_ids, :, 2]
            ).detach().clone()

    # -------------------------------------------------------------------------
    # RL hooks
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        if not self._printed_initial_axis:
            custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

            logger.info("==================================================")
            logger.info("[START_DIRECTION_CHECK]")
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
                f"[LIKU][step {int(self.common_step_counter)}] "
                f"action mean={a.mean().item():.4f}, "
                f"std={a.std().item():.4f}, "
                f"min={a.min().item():.4f}, "
                f"max={a.max().item():.4f}, "
                f"env0_action={a.tolist()}"
            )

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions

        if self.common_step_counter % self._debug_every == 0:
            env_id = self._debug_env_id

            q = self.robot.data.joint_pos[env_id].detach().cpu()
            qd = self.robot.data.joint_vel[env_id].detach().cpu()
            tau = forces[env_id].detach().cpu()

            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}] "
                f"env0_joint_pos={[round(x, 4) for x in q.tolist()]}"
            )
            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}] "
                f"env0_joint_vel={[round(x, 4) for x in qd.tolist()]}"
            )
            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}] "
                f"env0_forces(legs_only)={[round(x, 4) for x in tau.tolist()]}"
            )

        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """
        Refreshed reward design.

        핵심 의도:
        - stand: 자세 안정화만 학습한다.
        - onestep: 한 발 전진 + 안정적인 착지/유지.
        - walk: onestep 이후 한 발에서 멈추지 않고 계속 진행.
        """
        custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

        root_pos_w = self._get_root_pos_w()
        num_envs = root_pos_w.shape[0]

        # --------------------------------------------------
        # dt / root movement
        # --------------------------------------------------
        dt = float(getattr(self, "step_dt", self.cfg.sim.dt * self.cfg.decimation))

        delta_pos_w = root_pos_w - self._prev_root_pos_w
        root_vel_w = delta_pos_w / max(dt, 1.0e-6)

        delta_pos_xy = delta_pos_w.clone()
        delta_pos_xy[:, 2] = 0.0

        root_vel_xy = root_vel_w.clone()
        root_vel_xy[:, 2] = 0.0

        # 실제 이동 보상 기준 축
        move_axis = self._world_move_axis.expand(num_envs, -1).clone()
        move_axis[:, 2] = 0.0
        move_axis = F.normalize(move_axis, dim=-1, eps=1e-6)

        forward_dist = torch.sum(delta_pos_xy * move_axis, dim=-1)
        forward_vel = torch.sum(root_vel_xy * move_axis, dim=-1)

        # side 성분은 전방축에 수직인 xy 이동량
        forward_vel_vec = forward_vel.unsqueeze(-1) * move_axis
        side_vel_vec = root_vel_xy - forward_vel_vec
        side_vel = torch.linalg.norm(side_vel_vec[:, 0:2], dim=-1)

        z_vel = root_vel_w[:, 2]
        torso_z = root_pos_w[:, 2]

        if self.cfg.training_stage == "stand":
            active_termination_height = self.cfg.stand_termination_height
        elif self.cfg.training_stage == "onestep":
            active_termination_height = self.cfg.onestep_termination_height
        else:
            active_termination_height = self.cfg.walk_termination_height

        # --------------------------------------------------
        # posture / health shaping
        # --------------------------------------------------
        up_error = torch.clamp(1.0 - custom_up_proj, min=0.0)
        heading_error = torch.clamp(1.0 - custom_heading_proj, min=0.0)

        upright_reward = torch.exp(-((up_error / self.cfg.upright_sigma) ** 2))
        heading_reward = torch.exp(-((heading_error / self.cfg.heading_sigma) ** 2))

        height_reward = torch.exp(
            -(((torso_z - self.cfg.stand_target_height) / self.cfg.stand_height_sigma) ** 2)
        )

        height_gate = torch.clamp(
            (torso_z - active_termination_height)
            / max(self.cfg.stand_target_height - active_termination_height, 1.0e-6),
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

        warmup_gate = torch.clamp(
            (self.episode_length_buf.float() - self.cfg.walk_warmup_steps)
            / max(self.cfg.walk_warmup_steps, 1.0),
            min=0.0,
            max=1.0,
        )

        # --------------------------------------------------
        # regularization
        # --------------------------------------------------
        leg_q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        leg_qd = self.robot.data.joint_vel[:, self._joint_dof_idx]

        actions_cost = torch.mean(self.actions ** 2, dim=-1)

        action_rate_cost = torch.mean(
            (self.actions - self._prev_actions) ** 2,
            dim=-1,
        )

        joint_vel_cost = torch.mean(leg_qd ** 2, dim=-1)

        joint_limit_cost = torch.mean(
            torch.clamp(torch.abs(leg_q) - self.cfg.joint_limit_soft, min=0.0) ** 2,
            dim=-1,
        )

        # +/-30 deg hard limit for controlled leg joints.
        # If any controlled joint exceeds this range, the episode terminates.
        joint_angle_limit_violation = torch.any(
            torch.abs(leg_q) > self.cfg.joint_limit_hard,
            dim=-1,
        )

        action_saturation_cost = torch.mean(
            torch.clamp(torch.abs(self.actions) - self.cfg.action_soft_limit, min=0.0) ** 2,
            dim=-1,
        )

        common_regularization = (
            self.cfg.action_cost_scale * actions_cost
            + self.cfg.action_rate_cost_scale * action_rate_cost
            + self.cfg.action_saturation_cost_scale * action_saturation_cost
            + self.cfg.joint_vel_cost_scale * joint_vel_cost
            + self.cfg.joint_limit_cost_scale * joint_limit_cost
        )



        record_height_gate = torch.clamp(
            (torso_z - 1.95) / 0.08,
            min=0.0,
            max=1.0,
        )

        record_drop_gate = torch.clamp(
            (z_vel + 0.12) / 0.12,
            min=0.0,
            max=1.0,
        )

        record_side_gate = torch.clamp(
            (0.18 - side_vel) / 0.18,
            min=0.0,
            max=1.0,
        )

        record_stable_gate = (
            healthy_gate
            * record_height_gate
            * record_drop_gate
            * record_side_gate
            * warmup_gate
        )




        # --------------------------------------------------
        # stage 1: stand
        # --------------------------------------------------
        xy_vel_cost = torch.sum(root_vel_xy[:, 0:2] ** 2, dim=-1)
        z_vel_square_cost = z_vel ** 2

        root_delta_from_reset = root_pos_w - self._initial_root_pos_w
        xy_pos_cost = torch.sum(root_delta_from_reset[:, 0:2] ** 2, dim=-1)

        low_height_error = torch.clamp(
            (self.cfg.stand_target_height - torso_z) - self.cfg.stand_height_deadband,
            min=0.0,
        )
        high_height_error = torch.clamp(
            (torso_z - self.cfg.stand_target_height) - self.cfg.stand_height_deadband,
            min=0.0,
        )
        height_pos_cost = low_height_error ** 2 + 0.25 * high_height_error ** 2

        stand_joint_pos_cost = torch.mean(leg_q ** 2, dim=-1)

        stand_reward = (
            self.cfg.stand_alive_scale * healthy_gate
            + self.cfg.stand_upright_scale * upright_reward * height_gate
            + self.cfg.stand_height_scale * height_reward * posture_gate
            + self.cfg.stand_heading_scale * heading_reward * healthy_gate
            - self.cfg.stand_xy_vel_cost * xy_vel_cost
            - self.cfg.stand_z_vel_cost * z_vel_square_cost
            - self.cfg.stand_xy_pos_cost * xy_pos_cost
            - self.cfg.stand_height_pos_cost * height_pos_cost
            - self.cfg.stand_joint_pos_cost * stand_joint_pos_cost
            - common_regularization
        )



        # --------------------------------------------------
        # stage 2: onestep / stage 3: walk
        # --------------------------------------------------
        # 공통 gate:
        # - 높이가 너무 낮거나 빠르게 떨어지는 중이면 전진 reward를 거의 주지 않는다.
        # - "발만 앞으로 던지고 몸통이 무너지는" 해법을 막기 위한 gate다.
        backward_vel = torch.clamp(-forward_vel, min=0.0, max=0.60)

        height_strict_gate = torch.clamp(
            (torso_z - 1.90) / 0.12,
            min=0.0,
            max=1.0,
        )

        no_drop_gate = torch.clamp(
            (z_vel + 0.25) / 0.25,
            min=0.0,
            max=1.0,
        )

        stable_forward_gate = healthy_gate * height_strict_gate * no_drop_gate * warmup_gate

        # episode 누적 진행량을 reward 계산 전에 미리 계산한다.
        # 실제 상태 변수 업데이트는 reward 계산 마지막에서 한 번만 한다.
        episode_forward_next = self._episode_forward_dist + forward_dist.detach()
        episode_forward_positive = torch.clamp(episode_forward_next, min=0.0)

        episode_side_next = self._episode_side_dist + side_vel.detach() * dt

        forward_record_delta = torch.clamp(
            episode_forward_positive - self._episode_forward_best,
            min=0.0,
            max=self.cfg.forward_record_delta_clip,
        )

        # --------------------------------------------------
        # foot-step auxiliary reward
        # --------------------------------------------------
        # root 기준으로 reset 시점 대비 한쪽 발이 앞으로 나갔는지 본다.
        # 이 reward는 보행을 직접 대체하는 reward가 아니라,
        # 발을 앞으로 내딛는 동작을 유도하는 보조 reward다.
        foot_pos_w = self._get_foot_pos_w()
        foot_rel_w = foot_pos_w - root_pos_w.unsqueeze(1)

        foot_forward_rel = torch.sum(
            foot_rel_w * move_axis.unsqueeze(1),
            dim=-1,
        )

        initial_foot_forward_rel = torch.sum(
            self._initial_foot_rel_w * move_axis.unsqueeze(1),
            dim=-1,
        )

        foot_forward_delta = foot_forward_rel - initial_foot_forward_rel
        max_foot_forward_delta = torch.max(foot_forward_delta, dim=1).values

        foot_step_reward = (
            self.cfg.foot_step_reward_scale
            * torch.exp(
                -(
                    (
                        max_foot_forward_delta - self.cfg.foot_step_target
                    )
                    / max(self.cfg.foot_step_sigma, 1.0e-6)
                ) ** 2
            )
            * healthy_gate
            * height_strict_gate
            * no_drop_gate
        )

        foot_overstep_penalty = (
            self.cfg.foot_overstep_cost
            * torch.clamp(
                max_foot_forward_delta - self.cfg.foot_overstep_limit,
                min=0.0,
            ) ** 2
            * healthy_gate
        )

        # --------------------------------------------------
        # walk 단계용 step-separation auxiliary reward
        # --------------------------------------------------
        # reset 기준 foot 전진량은 walk가 진행될수록 의미가 약해질 수 있다.
        # 그래서 walk에서는 현재 root 기준으로 양발이 앞뒤로 적당히 벌어졌는지를 추가로 본다.
        # 단, 제자리에서 다리만 벌리는 꼼수를 막기 위해 forward_vel 기반 moving gate를 곱한다.
        left_foot_forward = foot_forward_rel[:, 0]
        right_foot_forward = foot_forward_rel[:, 1]

        step_separation = torch.abs(left_foot_forward - right_foot_forward)

        step_separation_motion_gate = torch.clamp(
            forward_vel / max(self.cfg.walk_target_forward_vel, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        step_separation_reward = (
            self.cfg.step_separation_reward_scale
            * torch.exp(
                -(
                    (
                        step_separation - self.cfg.step_separation_target
                    )
                    / max(self.cfg.step_separation_sigma, 1.0e-6)
                ) ** 2
            )
            * healthy_gate
            * height_strict_gate
            * no_drop_gate
            * warmup_gate
            * step_separation_motion_gate
        )

        step_separation_over_penalty = (
            self.cfg.step_separation_over_cost
            * torch.clamp(
                step_separation - self.cfg.step_separation_over_limit,
                min=0.0,
            ) ** 2
            * healthy_gate
        )

        # --------------------------------------------------
        # onestep reward
        # --------------------------------------------------
        # 목표:
        # - 한 발 정도 전진한다.
        # - 전진 후 주저앉거나 옆으로 무너지지 않고 버틴다.
        # - 너무 크게 던져서 넘어지는 런지 동작은 억제한다.
        onestep_record_reward = (
            self.cfg.onestep_record_reward_scale
            * forward_record_delta
            * record_stable_gate
        )

        onestep_dist_error = (
            episode_forward_positive - self.cfg.onestep_target_forward_dist
        )

        onestep_target_reward = (
            self.cfg.onestep_target_reward_scale
            * torch.exp(
                -(
                    onestep_dist_error
                    / max(self.cfg.onestep_dist_sigma, 1.0e-6)
                ) ** 2
            )
            * healthy_gate
            * height_strict_gate
            * no_drop_gate
        )

        onestep_success = (
            episode_forward_positive > self.cfg.onestep_success_forward_dist
        ).float()

        onestep_success_bonus = (
            self.cfg.onestep_success_bonus_scale
            * onestep_success
            * healthy_gate
            * height_strict_gate
            * no_drop_gate
        )

        onestep_hold_vel_penalty = (
            self.cfg.onestep_hold_vel_cost
            * onestep_success
            * (
                torch.sum(root_vel_xy[:, 0:2] ** 2, dim=-1)
                + 0.5 * torch.abs(z_vel)
            )
            * healthy_gate
        )

        onestep_overshoot_penalty = (
            self.cfg.onestep_overshoot_cost
            * torch.clamp(
                episode_forward_positive
                - 1.7 * self.cfg.onestep_target_forward_dist,
                min=0.0,
            ) ** 2
            * healthy_gate
        )

        # onestep에서도 옆으로 많이 흘러가면 안 된다.
        onestep_episode_side_excess = torch.clamp(
            episode_side_next
            - 3.0 * episode_forward_positive
            - 0.04,
            min=0.0,
        )

        onestep_episode_side_penalty = (
            0.40
            * onestep_episode_side_excess
            * healthy_gate
        )

        onestep_reward = (
            self.cfg.walk_alive_scale * healthy_gate
            + self.cfg.walk_upright_scale * upright_reward * height_gate
            + self.cfg.walk_height_scale * height_reward * posture_gate
            + self.cfg.walk_heading_scale * heading_reward * healthy_gate
            + onestep_record_reward
            + onestep_target_reward
            + onestep_success_bonus
            + foot_step_reward
            - foot_overstep_penalty
            - onestep_hold_vel_penalty
            - onestep_overshoot_penalty
            - onestep_episode_side_penalty
            - self.cfg.walk_backward_vel_cost * backward_vel * healthy_gate
            - self.cfg.walk_side_vel_cost * side_vel * healthy_gate
            - self.cfg.walk_z_vel_cost * torch.abs(z_vel) * healthy_gate
            - common_regularization
        )

        # --------------------------------------------------
        # walk reward
        # --------------------------------------------------
        # 낮은 목표 속도를 꾸준히 추종하도록 만든다.
        # 여기에 "episode 최고 전진거리 갱신 reward"를 추가해서
        # 한 발 후 멈추는 local optimum을 줄인다.
        forward_tracking_reward = torch.exp(
            -(
                (
                    forward_vel - self.cfg.walk_target_forward_vel
                )
                / max(self.cfg.walk_forward_vel_sigma, 1.0e-6)
            ) ** 2
        )

        # backward / 정지 상태에서 tracking reward가 들어가는 것을 방지한다.
        forward_alive_gate = torch.clamp(
            forward_vel / max(self.cfg.walk_target_forward_vel, 1.0e-6),
            min=0.0,
            max=1.0,
        )

        forward_reward = (
            self.cfg.walk_forward_vel_scale
            * forward_tracking_reward
            * forward_alive_gate
            * stable_forward_gate
        )



        forward_record_reward = (
            self.cfg.forward_record_reward_scale
            * forward_record_delta
            * record_stable_gate
        )






        backward_penalty = (
            self.cfg.walk_backward_vel_cost
            * backward_vel
            * healthy_gate
            * warmup_gate
        )

        side_vel_penalty = self.cfg.walk_side_vel_cost * side_vel * healthy_gate
        z_vel_penalty = self.cfg.walk_z_vel_cost * torch.abs(z_vel) * healthy_gate

        side_spike_penalty = 2.0 * torch.clamp(side_vel - 0.20, min=0.0) ** 2
        drop_spike_penalty = 2.5 * torch.clamp(-z_vel - 0.30, min=0.0) ** 2

        # forward_vel이 너무 크면 한 발 크게 던지는 런지 동작이 강화되기 쉬워서 억제한다.
        forward_speed_spike_penalty = (
            self.cfg.walk_forward_speed_spike_cost
            * torch.clamp(forward_vel - self.cfg.walk_forward_speed_limit, min=0.0) ** 2
        )

        no_progress_penalty = (
            self.cfg.no_progress_cost_scale
            * torch.clamp(self.cfg.min_forward_vel - forward_vel, min=0.0)
            * stable_forward_gate
        )

        episode_side_excess = torch.clamp(
            episode_side_next
            - 0.6 * episode_forward_positive
            - 0.05,
            min=0.0,
        )

        episode_side_penalty = (
            self.cfg.episode_side_cost_scale
            * episode_side_excess
            * healthy_gate
        )

        low_height_penalty = 6.0 * torch.clamp(1.95 - torso_z, min=0.0) ** 2

        walk_reward = (
            self.cfg.walk_alive_scale * healthy_gate
            + self.cfg.walk_upright_scale * upright_reward * height_gate
            + self.cfg.walk_height_scale * height_reward * posture_gate
            + self.cfg.walk_heading_scale * heading_reward * healthy_gate
            + forward_reward
            + forward_record_reward
            + step_separation_reward
            + 0.1 * foot_step_reward
            - step_separation_over_penalty
            - foot_overstep_penalty
            - backward_penalty
            - side_vel_penalty
            - z_vel_penalty
            - side_spike_penalty
            - drop_spike_penalty
            - forward_speed_spike_penalty
            - no_progress_penalty
            - episode_side_penalty
            - common_regularization
            - low_height_penalty
        )

        if self.cfg.training_stage == "stand":
            total_reward = stand_reward
        elif self.cfg.training_stage == "onestep":
            total_reward = onestep_reward
        elif self.cfg.training_stage == "walk":
            total_reward = walk_reward
        else:
            raise RuntimeError(
                f"[LIKU] unknown training_stage={self.cfg.training_stage}. "
                "Use 'stand', 'onestep', or 'walk'."
            )

        # --------------------------------------------------
        # done reward override
        # --------------------------------------------------
        bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
        died = (
            (torso_z < active_termination_height)
            | bad_posture
            | joint_angle_limit_violation
        )

        total_reward = torch.where(
            died,
            torch.ones_like(total_reward) * self.cfg.death_cost,
            total_reward,
        )

        # --------------------------------------------------
        # episode stats
        # --------------------------------------------------
        self._episode_forward_dist[:] = episode_forward_next
        self._episode_side_dist[:] = episode_side_next
        self._episode_forward_best[:] = torch.maximum(
            self._episode_forward_best,
            episode_forward_positive,
        )

        # --------------------------------------------------
        # debug
        # --------------------------------------------------
        if self.common_step_counter % self._debug_every == 0:
            env_id = self._debug_env_id

            logger.info(
                f"[MOVE][env{env_id}] "
                f"stage={self.cfg.training_stage}, "
                f"root_pos={root_pos_w[env_id].detach().cpu().tolist()}, "
                f"delta_pos={delta_pos_w[env_id].detach().cpu().tolist()}, "
                f"move_axis={self._world_move_axis[0].detach().cpu().tolist()}, "
                f"forward_dist={forward_dist[env_id].item():.5f}, "
                f"forward_vel={forward_vel[env_id].item():.4f}, "
                f"side_vel={side_vel[env_id].item():.4f}, "
                f"z_vel={z_vel[env_id].item():.4f}, "
                f"episode_forward_dist={self._episode_forward_dist[env_id].item():.4f}, "
                f"episode_side_dist={self._episode_side_dist[env_id].item():.4f}"
            )

            logger.info(
                f"[POSTURE][env{env_id}] "
                f"up_proj={custom_up_proj[env_id].item():.4f}, "
                f"heading_proj={custom_heading_proj[env_id].item():.4f}, "
                f"torso_z={torso_z[env_id].item():.4f}, "
                f"height_gate={height_gate[env_id].item():.4f}, "
                f"posture_gate={posture_gate[env_id].item():.4f}, "
                f"healthy_gate={healthy_gate[env_id].item():.4f}, "
                f"warmup_gate={warmup_gate[env_id].item():.4f}, "
                f"active_termination_height={active_termination_height:.3f}"
            )

            logger.info(
                f"[FOOT_STEP][env{env_id}] "
                f"foot_forward_delta={foot_forward_delta[env_id].detach().cpu().tolist()}, "
                f"max_foot_forward_delta={max_foot_forward_delta[env_id].item():.4f}, "
                f"foot_step_reward={foot_step_reward[env_id].item():.4f}, "
                f"foot_overstep_penalty={foot_overstep_penalty[env_id].item():.4f}"
            )

            logger.info(
                f"[STEP_SEPARATION][env{env_id}] "
                f"foot_forward_rel={foot_forward_rel[env_id].detach().cpu().tolist()}, "
                f"step_separation={step_separation[env_id].item():.4f}, "
                f"step_separation_motion_gate={step_separation_motion_gate[env_id].item():.4f}, "
                f"step_separation_reward={step_separation_reward[env_id].item():.4f}, "
                f"step_separation_over_penalty={step_separation_over_penalty[env_id].item():.4f}"
            )

            logger.info(
                f"[REWARD] "
                f"stage={self.cfg.training_stage}, "
                f"total={total_reward.mean().item():.4f}, "
                f"stand={stand_reward.mean().item():.4f}, "
                f"onestep={onestep_reward.mean().item():.4f}, "
                f"walk={walk_reward.mean().item():.4f}, "
                f"forward_reward={forward_reward.mean().item():.4f}, "
                f"forward_record_reward={forward_record_reward.mean().item():.4f}, "
                f"onestep_record_reward={onestep_record_reward.mean().item():.4f}, "
                f"onestep_target_reward={onestep_target_reward.mean().item():.4f}, "
                f"foot_step_reward={foot_step_reward.mean().item():.4f}, "
                f"foot_overstep_penalty={foot_overstep_penalty.mean().item():.4f}, "
                f"step_separation_reward={step_separation_reward.mean().item():.4f}, "
                f"step_separation_over_penalty={step_separation_over_penalty.mean().item():.4f}, "
                f"step_separation={step_separation.mean().item():.4f}, "
                f"no_progress_penalty={no_progress_penalty.mean().item():.4f}, "
                f"episode_side_penalty={episode_side_penalty.mean().item():.4f}, "
                f"backward_penalty={backward_penalty.mean().item():.4f}, "
                f"side_vel_penalty={side_vel_penalty.mean().item():.4f}, "
                f"z_vel_penalty={z_vel_penalty.mean().item():.4f}, "
                f"upright_reward={upright_reward.mean().item():.4f}, "
                f"height_reward={height_reward.mean().item():.4f}, "
                f"heading_reward={heading_reward.mean().item():.4f}, "
                f"action_cost={actions_cost.mean().item():.4f}, "
                f"action_rate_cost={action_rate_cost.mean().item():.4f}, "
                f"joint_vel_cost={joint_vel_cost.mean().item():.4f}, "
                f"joint_limit_cost={joint_limit_cost.mean().item():.4f}, "
                f"joint_angle_limit_ratio={joint_angle_limit_violation.float().mean().item():.4f}, "
                f"action_saturation_cost={action_saturation_cost.mean().item():.4f}, "
                f"xy_pos_cost={xy_pos_cost.mean().item():.4f}, "
                f"height_pos_cost={height_pos_cost.mean().item():.4f}, "
                f"stand_joint_pos_cost={stand_joint_pos_cost.mean().item():.4f}, "
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

        if self.cfg.training_stage == "stand":
            active_termination_height = self.cfg.stand_termination_height
        elif self.cfg.training_stage == "onestep":
            active_termination_height = self.cfg.onestep_termination_height
        else:
            active_termination_height = self.cfg.walk_termination_height

        leg_q = self.robot.data.joint_pos[:, self._joint_dof_idx]
        joint_angle_limit_violation = torch.any(
            torch.abs(leg_q) > self.cfg.joint_limit_hard,
            dim=-1,
        )

        bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
        died = (
            (root_z < active_termination_height)
            | bad_posture
            | joint_angle_limit_violation
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.common_step_counter % self._debug_every == 0:
            logger.info(
                f"[DONE_CUSTOM] "
                f"stage={self.cfg.training_stage}, "
                f"root_z_mean={root_z.mean().item():.4f}, "
                f"up_proj_mean={custom_up_proj.mean().item():.4f}, "
                f"joint_angle_limit_ratio={joint_angle_limit_violation.float().mean().item():.4f}, "
                f"died_ratio={died.float().mean().item():.4f}, "
                f"timeout_ratio={time_out.float().mean().item():.4f}"
            )

        return died, time_out

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        obs = super()._get_observations()

        if self.common_step_counter == 0:
            if isinstance(obs, dict):
                for k, v in obs.items():
                    logger.info(f"[LIKU][OBS] {k}: shape={tuple(v.shape)}")
            else:
                logger.info(f"[LIKU][OBS] shape={tuple(obs.shape)}")

        return obs