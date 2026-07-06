from future import annotations

import loggingimport torchimport torch.nn.functional as F

import isaaclab.sim as sim_utilsfrom isaaclab.assets import ArticulationCfgfrom isaaclab.envs import DirectRLEnvCfgfrom isaaclab.scene import InteractiveSceneCfgfrom isaaclab.sim import SimulationCfgfrom isaaclab.terrains import TerrainImporterCfgfrom isaaclab.utils import configclassfrom isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

logger = logging.getLogger(name)

-----------------------------------------------------------------------------

Robot asset

-----------------------------------------------------------------------------

LIKU_USD_PATH = "D:/isaacsim/jhusd/new_test/test2_6joints.usd"

CONTROLLED_LEG_JOINTS = ["A3", "A4", "A5", "B9", "B10", "B11"]

def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:"""q: [N, 4] quaternion in (w, x, y, z)v: [N, 3] vector"""q_xyz = q[:, 1:4]q_w = q[:, 0:1]

uv = torch.cross(q_xyz, v, dim=-1)
uuv = torch.cross(q_xyz, uv, dim=-1)

return v + 2.0 * (q_w * uv + uuv)

def quat_angle_diff_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:"""q1, q2: [N, 4] quaternion in (w, x, y, z)return: angle difference [N] in radians"""q1 = F.normalize(q1, dim=-1, eps=1e-6)q2 = F.normalize(q2, dim=-1, eps=1e-6)

dot = torch.sum(q1 * q2, dim=-1).abs()
dot = torch.clamp(dot, 0.0, 1.0)

return 2.0 * torch.acos(dot)

LIKU_CFG = ArticulationCfg(spawn=sim_utils.UsdFileCfg(usd_path=LIKU_USD_PATH,rigid_props=sim_utils.RigidBodyPropertiesCfg(),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,),),init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.015),

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

-----------------------------------------------------------------------------

Env config

-----------------------------------------------------------------------------

@configclassclass LikuEnvCfg(DirectRLEnvCfg):episode_length_s = 15.0decimation = 2action_scale = 1.0

action_space = 6
observation_space = 48
state_space = 0

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
# 1단계: "stand"
# 2단계: "walk"
training_stage: str = "stand"

# -------------------------------------------------------------------------
# action / actuator
# -------------------------------------------------------------------------
# stand 단계에서는 6000으로 버티는 힘을 확보
# walk 단계에서도 일단 6000 유지, saturation penalty로 폭주 억제
joint_gears: list = [6000.0] * 6

# -------------------------------------------------------------------------
# reward scales
# -------------------------------------------------------------------------
heading_weight: float = 0.15
up_weight: float = 0.10
alive_reward_scale: float = 0.15

energy_cost_scale: float = 0.02
actions_cost_scale: float = 0.04
dof_vel_scale: float = 0.1

death_cost: float = -1.0
torso_tilt_penalty_scale: float = 1.0

# stand 단계에서는 코드에서 사실상 사용 안 함
# walk 단계에서는 약하게 사용
no_progress_penalty_scale: float = 0.15

termination_height: float = 1.90
side_motion_penalty_scale: float = 40.0

angular_velocity_scale: float = 0.25
contact_force_scale: float = 0.01
forward_reward_scale: float = 4.5

vertical_motion_penalty_cost_scale: float = 0.20

# -------------------------------------------------------------------------
# standing pretrain target
# -------------------------------------------------------------------------
stand_target_height: float = 2.01
stand_height_sigma: float = 0.04
stand_height_reward_weight: float = 1.2

# -------------------------------------------------------------------------
# foot settings
# -------------------------------------------------------------------------
foot_body_names = ["Hip6", "LHip6"]

foot_orientation_reward_weight: float = 0.03
foot_orientation_sigma: float = 0.35
foot_orientation_cutoff: float = 0.60

# 발 방향 penalty
foot_orientation_allowance: float = 0.18
foot_orientation_penalty_scale: float = 6.0

# 발이 위로 뜨는 것 방지
foot_lift_allowance: float = 0.025
foot_lift_penalty_scale: float = 120.0

# 양발 높이 차이 방지
foot_height_diff_allowance: float = 0.035
foot_height_diff_penalty_scale: float = 80.0

# 발 z가 시작 위치에서 위/아래로 많이 벗어나는 것 방지
foot_z_deviation_allowance: float = 0.015
foot_z_deviation_penalty_scale: float = 120.0

# 발이 root 기준 뒤쪽으로 빠지는 것 방지
foot_backward_allowance: float = 0.10
foot_backward_penalty_scale: float = 40.0
foot_backward_penalty_cost: float = 1.0

# 다리 접힘 방지
leg_collapse_allowance: float = 0.06
leg_collapse_penalty_scale: float = 80.0

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

bad_posture_threshold = 0.85

-----------------------------------------------------------------------------

Env

-----------------------------------------------------------------------------

class LikuEnv(LocomotionEnv):cfg: LikuEnvCfg

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

    if hasattr(self, "_prev_actions"):
        self._prev_actions[env_ids] = 0.0

    if hasattr(self, "_episode_forward_dist"):
        self._episode_forward_dist[env_ids] = 0.0

    if hasattr(self, "_episode_side_dist"):
        self._episode_side_dist[env_ids] = 0.0

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
    custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

    old_progress_reward = self.potentials - self.prev_potentials

    root_quat = self._get_root_quat_w()
    root_pos_w = self._get_root_pos_w()
    num_envs = root_pos_w.shape[0]

    # --------------------------------------------------
    # movement
    # --------------------------------------------------
    move_fwd_xy = self._world_move_axis.expand(num_envs, -1).clone()
    move_fwd_xy[:, 2] = 0.0
    move_fwd_xy = F.normalize(move_fwd_xy, dim=-1, eps=1e-6)

    delta_pos_w = root_pos_w - self._prev_root_pos_w

    delta_pos_xy = delta_pos_w.clone()
    delta_pos_xy[:, 2] = 0.0

    # 실제 root 이동거리. 단위는 USD world unit.
    forward_dist = torch.sum(delta_pos_xy * move_fwd_xy, dim=-1)

    forward_dist_clipped = torch.clamp(forward_dist, min=-0.03, max=0.03)

    forward_dist_pos = torch.clamp(forward_dist_clipped, min=0.0)
    backward_dist = torch.clamp(-forward_dist_clipped, min=0.0)

    torso_z = root_pos_w[:, 2]

    # --------------------------------------------------
    # gates
    # --------------------------------------------------
    height_gate = torch.clamp(
        (torso_z - 1.88) / 0.12,
        min=0.0,
        max=1.0,
    )

    height_keep_gate = torch.clamp(
        (torso_z - 1.96) / 0.04,
        min=0.0,
        max=1.0,
    )

    posture_gate = torch.clamp(
        (custom_up_proj - 0.94) / 0.04,
        min=0.0,
        max=1.0,
    )

    heading_gate = torch.clamp(
        (custom_heading_proj - 0.80) / 0.15,
        min=0.0,
        max=1.0,
    )

    # --------------------------------------------------
    # joint limit
    # --------------------------------------------------
    q_all = self.robot.data.joint_pos

    joint_limit_start = 1.35

    joint_limit_penalty = torch.sum(
        torch.clamp(torch.abs(q_all) - joint_limit_start, min=0.0) ** 2,
        dim=-1,
    )

    joint_gate = torch.exp(-2.0 * joint_limit_penalty)

    stand_gate = height_gate * posture_gate * heading_gate * joint_gate
    strict_forward_gate = height_keep_gate * posture_gate * heading_gate * joint_gate

    # --------------------------------------------------
    # no drop gate
    # --------------------------------------------------
    downward_drop_for_gate = torch.clamp(
        -delta_pos_w[:, 2],
        min=0.0,
    )

    no_drop_gate = torch.clamp(
        (0.020 - downward_drop_for_gate) / 0.020,
        min=0.0,
        max=1.0,
    )

    # --------------------------------------------------
    # forward rewards for walk stage
    # --------------------------------------------------
    # forward_dist_limited = torch.clamp(
    #     forward_dist_pos,
    #     max=0.003,
    # )

    # forward_reward_raw = 150.0 * forward_dist_limited
    # forward_reward_raw = forward_reward_raw * stand_gate * no_drop_gate

    # forward_reward_healthy = 600.0 * forward_dist_limited
    # forward_reward_healthy = forward_reward_healthy * strict_forward_gate * no_drop_gate

    # forward_dist_reward = forward_reward_raw + forward_reward_healthy

    # backward_dist_penalty = 120.0 * backward_dist

    forward_dist_limited = torch.clamp(
        forward_dist_pos,
        max=0.0030,
    )

    # root가 실제로 앞으로 이동한 보상
    forward_dist_reward = (
        800.0
        * forward_dist_limited
        * strict_forward_gate
        * no_drop_gate
    )

    # 뒤로 가는 이동 penalty
    backward_dist_penalty = (
        160.0
        * backward_dist
        * stand_gate
    )

    # 앞뒤 흔들기 방지용 signed progress
    # 앞으로 가면 +, 뒤로 가면 -
    forward_dist_signed = torch.clamp(
        forward_dist_clipped,
        min=-0.0030,
        max=0.0030,
    )

    net_forward_reward = (
        200.0
        * forward_dist_signed
        * strict_forward_gate
        * no_drop_gate
    )

    # --------------------------------------------------
    # posture rewards
    # --------------------------------------------------
    heading_reward = torch.where(
        custom_heading_proj > 0.8,
        torch.ones_like(custom_heading_proj) * self.cfg.heading_weight,
        self.cfg.heading_weight * custom_heading_proj / 0.8,
    )

    up_reward = torch.where(
        custom_up_proj > 0.93,
        torch.ones_like(custom_up_proj) * self.cfg.up_weight,
        torch.zeros_like(custom_up_proj),
    )

    stable_gate = height_gate * posture_gate

    heading_reward = heading_reward * stable_gate
    up_reward = up_reward * stable_gate

    alive_reward = (
        torch.ones_like(forward_dist_reward)
        * self.cfg.alive_reward_scale
        * stable_gate
    )

    # --------------------------------------------------
    # foot data
    # --------------------------------------------------
    foot_pos_w = self._get_foot_pos_w()
    foot_rel_w = foot_pos_w - root_pos_w.unsqueeze(1)
    foot_z_w = foot_pos_w[:, :, 2]

    # --------------------------------------------------
    # foot backward penalty
    # --------------------------------------------------
    initial_foot_rel_w = self._initial_foot_rel_w

    back_axis = -move_fwd_xy
    back_axis = back_axis.unsqueeze(1)

    foot_rel_delta = foot_rel_w - initial_foot_rel_w

    foot_backward_amount = torch.sum(
        foot_rel_delta * back_axis,
        dim=-1,
    )

    foot_backward_error = torch.clamp(
        foot_backward_amount - self.cfg.foot_backward_allowance,
        min=0.0,
    )

    foot_backward_penalty = self.cfg.foot_backward_penalty_scale * torch.sum(
        foot_backward_error ** 2,
        dim=-1,
    )

    # --------------------------------------------------
    # leg collapse penalty
    # --------------------------------------------------
    root_to_foot_z = (
        root_pos_w[:, 2].unsqueeze(1)
        - foot_z_w
    )

    leg_collapse_amount = self._initial_root_to_foot_z - root_to_foot_z

    leg_collapse_error = torch.clamp(
        leg_collapse_amount - self.cfg.leg_collapse_allowance,
        min=0.0,
    )

    leg_collapse_penalty = self.cfg.leg_collapse_penalty_scale * torch.sum(
        leg_collapse_error ** 2,
        dim=-1,
    )

    # --------------------------------------------------
    # foot orientation reward / penalty
    # --------------------------------------------------
    foot_quat_w = self._get_foot_quat_w()
    initial_foot_quat_w = self._initial_foot_quat_w

    num_feet = foot_quat_w.shape[1]

    foot_angle_error = quat_angle_diff_wxyz(
        foot_quat_w.reshape(-1, 4),
        initial_foot_quat_w.reshape(-1, 4),
    ).reshape(num_envs, num_feet)

    foot_angle_error_mean = foot_angle_error.mean(dim=-1)
    foot_angle_error_max = foot_angle_error.max(dim=-1).values

    foot_orientation_reward = torch.exp(
        -((foot_angle_error_mean / self.cfg.foot_orientation_sigma) ** 2)
    )

    foot_orientation_reward = torch.where(
        foot_angle_error_max < self.cfg.foot_orientation_cutoff,
        foot_orientation_reward,
        torch.zeros_like(foot_orientation_reward),
    )

    foot_orientation_reward = foot_orientation_reward * stand_gate

    foot_orientation_error = torch.clamp(
        foot_angle_error - self.cfg.foot_orientation_allowance,
        min=0.0,
    )

    foot_orientation_penalty = (
        self.cfg.foot_orientation_penalty_scale
        * torch.sum(foot_orientation_error ** 2, dim=-1)
        * stand_gate
    )

    # --------------------------------------------------
    # foot step reward
    # --------------------------------------------------
    # 기존 절대 foot 위치 기준은 USD/foot body 구조에 따라 보상이 포화될 수 있음.
    # 그래서 reset 시점의 root 기준 발 위치 대비 변화량만 보상한다.
    root_xy = root_pos_w[:, 0:2].unsqueeze(1)
    foot_xy = foot_pos_w[:, :, 0:2]
    foot_rel_xy = foot_xy - root_xy

    initial_foot_rel_xy = self._initial_foot_rel_w[:, :, 0:2]
    foot_rel_delta_xy = foot_rel_xy - initial_foot_rel_xy

    move_fwd_xy_2d = self._world_move_axis[:, 0:2].expand(num_envs, -1)
    move_fwd_xy_2d = F.normalize(move_fwd_xy_2d, dim=-1, eps=1e-6)

    # 각 발이 reset 대비 전방으로 얼마나 움직였는지
    foot_forward_delta = torch.sum(
        foot_rel_delta_xy * move_fwd_xy_2d.unsqueeze(1),
        dim=-1,
    )

    max_foot_forward_delta = torch.max(foot_forward_delta, dim=1).values

    # root가 실제로 앞으로 갈 때만 foot-step reward를 살림
    target_forward_dist = 0.0015
    root_progress_gate = torch.clamp(
        forward_dist_pos / target_forward_dist,
        min=0.0,
        max=1.0,
    )

    foot_step_amount = torch.clamp(
        max_foot_forward_delta - 0.003,
        min=0.0,
        max=0.035,
    )

    foot_step_reward = (
        10.0
        * foot_step_amount
        * height_gate
        * posture_gate
        * root_progress_gate
    )

    foot_forward_diff = torch.abs(
        foot_forward_delta[:, 0] - foot_forward_delta[:, 1]
    )

    step_separation_reward = (
        4.0
        * torch.clamp(foot_forward_diff - 0.005, min=0.0, max=0.04)
        * height_gate
        * posture_gate
        * root_progress_gate
    )

    # --------------------------------------------------
    # foot lift / one-leg standing penalty
    # --------------------------------------------------
    foot_lift_amount = foot_z_w - self._initial_foot_z_w

    foot_lift_error = torch.clamp(
        foot_lift_amount - self.cfg.foot_lift_allowance,
        min=0.0,
    )

    foot_lift_penalty = (
        self.cfg.foot_lift_penalty_scale
        * torch.sum(foot_lift_error ** 2, dim=-1)
        * stand_gate
    )

    foot_height_diff = torch.abs(foot_z_w[:, 0] - foot_z_w[:, 1])

    foot_height_diff_error = torch.clamp(
        foot_height_diff - self.cfg.foot_height_diff_allowance,
        min=0.0,
    )

    foot_height_diff_penalty = (
        self.cfg.foot_height_diff_penalty_scale
        * foot_height_diff_error ** 2
        * stand_gate
    )

    # 발이 시작 z에서 위/아래로 많이 벗어나는 것 방지
    foot_z_deviation = torch.abs(foot_z_w - self._initial_foot_z_w)

    foot_z_deviation_error = torch.clamp(
        foot_z_deviation - self.cfg.foot_z_deviation_allowance,
        min=0.0,
    )

    foot_z_deviation_penalty = (
        self.cfg.foot_z_deviation_penalty_scale
        * torch.sum(foot_z_deviation_error ** 2, dim=-1)
        * stand_gate
    )

    # --------------------------------------------------
    # penalties
    # --------------------------------------------------
    upward_jump = torch.clamp(delta_pos_w[:, 2], min=0.0)
    downward_drop = torch.clamp(-delta_pos_w[:, 2], min=0.0)

    vertical_motion_penalty = 40.0 * upward_jump + 60.0 * downward_drop

    side_dist = torch.abs(delta_pos_xy[:, 1])
    side_motion_penalty = self.cfg.side_motion_penalty_scale * side_dist

    # episode 누적 이동거리 로그용
    self._episode_forward_dist += forward_dist.detach()
    self._episode_side_dist += side_dist.detach()

    actions_cost = torch.sum(self.actions ** 2, dim=-1)

    action_rate_cost = torch.sum(
        (self.actions - self._prev_actions) ** 2,
        dim=-1,
    )

    leg_dof_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]

    electricity_cost = torch.sum(
        torch.abs(self.actions * leg_dof_vel * self.cfg.dof_vel_scale),
        dim=-1,
    )

    action_saturation_penalty = torch.sum(
        torch.clamp(torch.abs(self.actions) - 0.85, min=0.0) ** 2,
        dim=-1,
    )

    leg_velocity_spike_penalty = torch.sum(
        torch.clamp(torch.abs(leg_dof_vel) - 14.0, min=0.0) ** 2,
        dim=-1,
    )

    leg_vel_square_penalty = torch.mean(leg_dof_vel ** 2, dim=-1)

    fwd_local = self._robot_forward_axis_local.expand(num_envs, -1)
    fwd_world = quat_rotate_wxyz(root_quat, fwd_local)
    fwd_world = F.normalize(fwd_world, dim=-1, eps=1e-6)

    body_pitch_penalty = torch.square(fwd_world[:, 2])

    torso_tilt_angle = torch.acos(
        torch.clamp(custom_up_proj, -1.0, 1.0)
    )

    torso_tilt_penalty = 20.0 * torch.clamp(
        torso_tilt_angle - 0.14,
        min=0.0,
    ) ** 2

    low_height_penalty = torch.clamp(2.00 - torso_z, min=0.0) ** 2

    # --------------------------------------------------
    # no progress penalty for walk stage
    # --------------------------------------------------
    target_forward_dist = 0.0015

    no_progress_error = torch.clamp(
        target_forward_dist - forward_dist_pos,
        min=0.0,
        max=target_forward_dist,
    )

    no_progress_warmup_gate = torch.clamp(
        (self.episode_length_buf.float() - 15.0) / 30.0,
        min=0.0,
        max=1.0,
    )

    no_progress_penalty = (
        self.cfg.no_progress_penalty_scale
        * (no_progress_error / target_forward_dist)
        * stand_gate
        * no_progress_warmup_gate
    )

    # --------------------------------------------------
    # stand pretrain reward
    # --------------------------------------------------
    height_target_reward = torch.exp(
        -((torso_z - self.cfg.stand_target_height) / self.cfg.stand_height_sigma) ** 2
    )

    horizontal_drift_penalty = 50.0 * torch.sum(delta_pos_xy ** 2, dim=-1)

    vertical_drift_penalty = 30.0 * torch.square(delta_pos_w[:, 2])

    stand_reward = (
        self.cfg.alive_reward_scale * stand_gate
        + self.cfg.up_weight * posture_gate
        + self.cfg.heading_weight * heading_gate
        + self.cfg.stand_height_reward_weight * height_target_reward * posture_gate

        + self.cfg.foot_orientation_reward_weight * foot_orientation_reward

        - horizontal_drift_penalty
        - vertical_drift_penalty

        - 0.02 * actions_cost
        - 0.02 * action_rate_cost
        - 0.002 * leg_vel_square_penalty

        - 80.0 * low_height_penalty

        - foot_orientation_penalty
        - foot_lift_penalty
        - foot_height_diff_penalty
        - foot_z_deviation_penalty
        - leg_collapse_penalty
    )

    # --------------------------------------------------
    # walk reward
    # --------------------------------------------------
    walk_alive_reward = 0.03 * stable_gate
    walk_up_reward = 0.03 * posture_gate * height_gate
    walk_heading_reward = 0.03 * heading_gate * height_gate

    walk_reward = (
        forward_dist_reward
        + net_forward_reward
        + walk_alive_reward
        + walk_up_reward
        + walk_heading_reward
        + 0.03 * foot_orientation_reward
        + foot_step_reward
        + step_separation_reward

        - backward_dist_penalty
        - no_progress_penalty
        - side_motion_penalty
        - self.cfg.vertical_motion_penalty_cost_scale * vertical_motion_penalty

        - self.cfg.actions_cost_scale * actions_cost
        - self.cfg.energy_cost_scale * electricity_cost
        - 0.12 * action_rate_cost

        - 1.6 * action_saturation_penalty
        - 0.004 * leg_velocity_spike_penalty

        - 4.0 * body_pitch_penalty
        - 2.5 * joint_limit_penalty
        - 100.0 * low_height_penalty
        - torso_tilt_penalty

        - foot_backward_penalty * self.cfg.foot_backward_penalty_cost
        - leg_collapse_penalty

        # walk에서는 발을 완전히 고정하면 못 걸으니까 약하게만 적용
        # 다만 발 보상 꼼수 방지를 위해 0.2 정도는 남김
        - 0.1 * foot_orientation_penalty
        - 0.15 * foot_lift_penalty
        - 0.15 * foot_height_diff_penalty
        - 0.15 * foot_z_deviation_penalty   
    )

    # walk_reward = (
    #     forward_dist_reward
    #     + alive_reward
    #     + up_reward
    #     + heading_reward
    #     + self.cfg.foot_orientation_reward_weight * foot_orientation_reward

    #     - backward_dist_penalty
    #     - no_progress_penalty
    #     - side_motion_penalty
    #     - self.cfg.vertical_motion_penalty_cost_scale * vertical_motion_penalty

    #     - self.cfg.actions_cost_scale * actions_cost
    #     - self.cfg.energy_cost_scale * electricity_cost
    #     - 0.1 * action_rate_cost

    #     - 0.6 * action_saturation_penalty
    #     - 0.002 * leg_velocity_spike_penalty

    #     - 4.0 * body_pitch_penalty
    #     - 2.5 * joint_limit_penalty
    #     - 80.0 * low_height_penalty
    #     - torso_tilt_penalty

    #     - foot_backward_penalty * self.cfg.foot_backward_penalty_cost
    #     - leg_collapse_penalty
    #     - foot_orientation_penalty
    #     - foot_lift_penalty
    #     - foot_height_diff_penalty
    #     - foot_z_deviation_penalty
    # )

    if self.cfg.training_stage == "stand":
        total_reward = stand_reward
    elif self.cfg.training_stage == "walk":
        total_reward = walk_reward
    else:
        raise RuntimeError(
            f"[LIKU] unknown training_stage={self.cfg.training_stage}. "
            "Use 'stand' or 'walk'."
        )

    # --------------------------------------------------
    # done reward override
    # --------------------------------------------------
    bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
    root_z = root_pos_w[:, 2]

    died = (root_z < self.cfg.termination_height) | bad_posture

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
            f"[AXIS_CUSTOM] "
            f"up_proj={custom_up_proj[env_id].item():.4f}, "
            f"heading_proj={custom_heading_proj[env_id].item():.4f}"
        )

        logger.info(
            f"[AXIS] root_quat_w="
            f"{self._get_root_quat_w()[env_id].detach().cpu().tolist()}"
        )

        self._debug_axis_candidates()

        logger.info(
            f"[MOVE_DIST][env{env_id}] "
            f"stage={self.cfg.training_stage}, "
            f"root_pos={root_pos_w[env_id].detach().cpu().tolist()}, "
            f"delta_pos={delta_pos_w[env_id].detach().cpu().tolist()}, "
            f"move_axis={self._world_move_axis[0].detach().cpu().tolist()}, "
            f"forward_dist={forward_dist[env_id].item():.5f}, "
            f"episode_forward_dist={self._episode_forward_dist[env_id].item():.4f}, "
            f"episode_side_dist={self._episode_side_dist[env_id].item():.4f}, "
            f"height_gate={height_gate[env_id].item():.4f}, "
            f"height_keep_gate={height_keep_gate[env_id].item():.4f}, "
            f"posture_gate={posture_gate[env_id].item():.4f}, "
            f"heading_gate={heading_gate[env_id].item():.4f}, "
            f"joint_gate={joint_gate[env_id].item():.4f}, "
            f"stand_gate={stand_gate[env_id].item():.4f}, "
            f"strict_forward_gate={strict_forward_gate[env_id].item():.4f}, "
            f"no_drop_gate={no_drop_gate[env_id].item():.4f}, "
            f"height_target_reward={height_target_reward[env_id].item():.4f}, "
            f"foot_angle_error={foot_angle_error_mean[env_id].item():.4f}, "
            f"foot_orientation_reward={foot_orientation_reward[env_id].item():.4f}, "
            f"foot_orientation_penalty={foot_orientation_penalty[env_id].item():.4f}, "
            f"foot_lift_amount_env0={foot_lift_amount[env_id].detach().cpu().tolist()}, "
            f"foot_lift_penalty_env0={foot_lift_penalty[env_id].item():.4f}, "
            f"foot_z_deviation_env0={foot_z_deviation[env_id].detach().cpu().tolist()}, "
            f"foot_z_deviation_penalty_env0={foot_z_deviation_penalty[env_id].item():.4f}, "
            f"foot_height_diff_env0={foot_height_diff[env_id].item():.4f}, "
            f"foot_height_diff_penalty_env0={foot_height_diff_penalty[env_id].item():.4f}, "
            f"foot_backward_amount={foot_backward_amount.mean().item():.4f}, "
            f"foot_backward_penalty={foot_backward_penalty.mean().item():.4f}, "
            f"leg_collapse_amount={leg_collapse_amount.mean().item():.4f}, "
            f"leg_collapse_penalty={leg_collapse_penalty.mean().item():.4f}, "
        )

        logger.info(
            f"[LIKU][step {int(self.common_step_counter)}][reward] "
            f"stage={self.cfg.training_stage}, "
            f"total={total_reward.mean().item():.4f}, "
            f"stand_reward={stand_reward.mean().item():.4f}, "
            f"walk_reward={walk_reward.mean().item():.4f}, "
            f"old_progress={old_progress_reward.mean().item():.4f}, "

            f"forward_dist={forward_dist.mean().item():.5f}, "
            f"forward_dist_reward={forward_dist_reward.mean().item():.4f}, "
            f"net_forward_reward={net_forward_reward.mean().item():.4f}, "
            f"backward_dist_penalty={backward_dist_penalty.mean().item():.4f}, "

            f"alive={alive_reward.mean().item():.4f}, "
            f"up={up_reward.mean().item():.4f}, "
            f"heading={heading_reward.mean().item():.4f}, "
            f"height_target_reward={height_target_reward.mean().item():.4f}, "

            f"foot_angle_error_mean={foot_angle_error_mean.mean().item():.4f}, "
            f"foot_angle_error_max={foot_angle_error_max.mean().item():.4f}, "
            f"foot_orientation_reward={foot_orientation_reward.mean().item():.4f}, "
            f"foot_orientation_penalty={foot_orientation_penalty.mean().item():.4f}, "
            f"foot_lift_penalty={foot_lift_penalty.mean().item():.4f}, "
            f"foot_height_diff_penalty={foot_height_diff_penalty.mean().item():.4f}, "
            f"foot_z_deviation_penalty={foot_z_deviation_penalty.mean().item():.4f}, "

            f"side_motion_penalty={side_motion_penalty.mean().item():.4f}, "
            f"vertical_motion_penalty={vertical_motion_penalty.mean().item():.4f}, "
            f"horizontal_drift_penalty={horizontal_drift_penalty.mean().item():.4f}, "
            f"vertical_drift_penalty={vertical_drift_penalty.mean().item():.4f}, "

            f"joint_limit_penalty={joint_limit_penalty.mean().item():.4f}, "
            f"low_height_penalty={low_height_penalty.mean().item():.4f}, "
            f"body_pitch_penalty={body_pitch_penalty.mean().item():.4f}, "

            f"action_rate_cost={action_rate_cost.mean().item():.4f}, "
            f"act_cost={actions_cost.mean().item():.4f}, "
            f"elec_cost={electricity_cost.mean().item():.4f}, "
            f"action_saturation_penalty={action_saturation_penalty.mean().item():.4f}, "
            f"leg_velocity_spike_penalty={leg_velocity_spike_penalty.mean().item():.4f}, "
            f"leg_vel_square_penalty={leg_vel_square_penalty.mean().item():.4f}, "
            f"no_progress_penalty={no_progress_penalty.mean().item():.4f}, "
            f"no_drop_gate={no_drop_gate.mean().item():.4f}, "
            f"root_progress_gate={root_progress_gate.mean().item():.4f}, "

            f"foot_backward_amount_env0={foot_backward_amount[env_id].detach().cpu().tolist()}, "
            f"foot_backward_penalty_env0={foot_backward_penalty[env_id].item():.4f}, "
            f"leg_collapse_amount_env0={leg_collapse_amount[env_id].detach().cpu().tolist()}, "
            f"leg_collapse_penalty_env0={leg_collapse_penalty[env_id].item():.4f}, "

            f"foot_step_reward={foot_step_reward.mean().item():.4f}, "
            f"step_separation_reward={step_separation_reward.mean().item():.4f}, "
            f"foot_step_amount={foot_step_amount.mean().item():.4f}, "
            f"foot_forward_delta_mean={foot_forward_delta.mean().item():.4f}, "
            f"foot_forward_diff={foot_forward_diff.mean().item():.4f}, "
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
            f"[DONE_CUSTOM] "
            f"stage={self.cfg.training_stage}, "
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
    obs = super()._get_observations()

    if self.common_step_counter == 0:
        if isinstance(obs, dict):
            for k, v in obs.items():
                logger.info(f"[LIKU][OBS] {k}: shape={tuple(v.shape)}")
        else:
            logger.info(f"[LIKU][OBS] shape={tuple(obs.shape)}")

    return obs