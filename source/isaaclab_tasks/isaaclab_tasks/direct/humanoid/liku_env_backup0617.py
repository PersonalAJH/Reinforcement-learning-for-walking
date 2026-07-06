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

    # q와 -q는 같은 자세라서 abs 사용
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
        pos=(0.0, 0.0, 2.05),

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

            # 너무 크면 움직임이 둔해질 수 있음
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

    # action_scale=1.0 기준 최대 force = 6000
    joint_gears: list = [6000.0] * 6

    # reward scales
    heading_weight: float = 0.5
    up_weight: float = 0.3
    energy_cost_scale: float = 0.1
    actions_cost_scale: float = 0.15
    alive_reward_scale: float = 0.45
    dof_vel_scale: float = 0.1
    death_cost: float = -1.0
    torso_tilt_penalty_scale: float = 1.0

    # 지금은 전진 안 함 벌점 끄기
    # 전진은 forward reward로만 유도
    no_progress_penalty_scale: float = 0.0

    termination_height: float = 1.90

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
    forward_reward_scale: float = 4.0

    vertical_motion_penalty_cost_scale: float = 0.5
    side_motion_penalty_scale: float = 120.0

    # 발 방향 유지 reward
    # Hip6, LHip6의 world orientation이 reset 시점과 비슷할수록 reward
    foot_body_names = ["Hip6", "LHip6"]
    foot_orientation_reward_weight: float = 0.6
    foot_orientation_sigma: float = 0.35  # rad, 약 20도
    foot_orientation_reward_weight: float = 0.3
    foot_orientation_sigma: float = 0.35
    foot_orientation_cutoff: float = 0.60
    foot_backward_penalty_cost: float = 1.00

    # CUSTOM AXIS
    robot_up_axis_local = (0.0, 1.0, 0.0)
    robot_forward_axis_local = (-1.0, 0.0, 0.0)

    world_up_axis = (0.0, 0.0, 1.0)

    # 보는 방향 기준
    world_forward_axis = (1.0, 0.0, 0.0)

    # 실제 이동 보상 기준
    # root가 world -X로 이동하면 forward reward
    world_move_axis = (-1.0, 0.0, 0.0)

    bad_posture_threshold = 0.85


    foot_backward_allowance: float = 0.10   # 10cm까지는 허용
    foot_backward_penalty_scale: float = 40.0


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

        # 발 body id 찾기
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

        # reset 시점의 발 방향 저장
        self._initial_foot_quat_w = self._get_foot_quat_w().detach().clone()


        # reset 시점의 root 기준 발 위치 저장
        self._initial_foot_rel_w = (self._get_foot_pos_w() - self._get_root_pos_w().unsqueeze(1)).detach().clone()


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

        # fallback: body_state_w = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        return self.robot.data.body_state_w[:, self._foot_body_ids, 3:7]

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
    # find foot pos
    # -------------------------------------------------------------------------

    def _get_foot_pos_w(self) -> torch.Tensor:
        """
        return: [num_envs, num_feet, 3]
        foot body position in world frame
        """
        if hasattr(self.robot.data, "body_pos_w"):
            return self.robot.data.body_pos_w[:, self._foot_body_ids, :]

        return self.robot.data.body_state_w[:, self._foot_body_ids, 0:3]






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

        if hasattr(self, "_initial_foot_quat_w"):
            self._initial_foot_quat_w[env_ids] = self._get_foot_quat_w()[env_ids].detach().clone()

        if hasattr(self, "_initial_foot_rel_w"):
            self._initial_foot_rel_w[env_ids] = (self._get_foot_pos_w()[env_ids] - self._get_root_pos_w()[env_ids].unsqueeze(1)).detach().clone()


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

        forward_dist = torch.sum(delta_pos_xy * move_fwd_xy, dim=-1)
        forward_dist_clipped = torch.clamp(forward_dist, min=-0.03, max=0.03)

        forward_dist_pos = torch.clamp(forward_dist_clipped, min=0.0)
        backward_dist = torch.clamp(-forward_dist_clipped, min=0.0)

        torso_z = root_pos_w[:, 2]

        # --------------------------------------------------
        # gates
        # --------------------------------------------------
        height_gate = torch.clamp(
            (torso_z - 1.90) / 0.15,
            min=0.0,
            max=1.0,
        )

        # 현재 네 코드 기준 유지
        # 1.80 기준이라 사실상 높이 gate가 거의 항상 켜짐
        height_keep_gate = torch.clamp(
            (torso_z - 1.80) / 0.05,
            min=0.0,
            max=1.0,
        )

        posture_gate = torch.clamp(
            (custom_up_proj - 0.94) / 0.08,
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
        # forward rewards
        # --------------------------------------------------
        forward_reward_raw = 40.0 * forward_dist_pos
        forward_reward_raw = forward_reward_raw * posture_gate * heading_gate

        forward_reward_healthy = 160.0 * forward_dist_pos
        forward_reward_healthy = forward_reward_healthy * strict_forward_gate

        forward_dist_reward = forward_reward_raw + forward_reward_healthy

        backward_dist_penalty = 160.0 * backward_dist

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
        # foot orientation reward
        # --------------------------------------------------
        foot_quat_w = self._get_foot_quat_w()
        initial_foot_quat_w = self._initial_foot_quat_w

        num_feet = foot_quat_w.shape[1]

        foot_angle_error = quat_angle_diff_wxyz(
            foot_quat_w.reshape(-1, 4),
            initial_foot_quat_w.reshape(-1, 4),
        ).reshape(num_envs, num_feet)

        foot_angle_error_mean = foot_angle_error.mean(dim=-1)


        # --------------------------------------------------
        # foot backward penalty
        # --------------------------------------------------
        foot_pos_w = self._get_foot_pos_w()
        foot_rel_w = foot_pos_w - root_pos_w.unsqueeze(1)

        initial_foot_rel_w = self._initial_foot_rel_w

        # 이동 방향 반대가 "뒤로 빼는 방향"
        # world_move_axis = (-1, 0, 0)이면 back_axis = (+1, 0, 0)
        back_axis = -move_fwd_xy
        back_axis = back_axis.unsqueeze(1)  # [num_envs, 1, 3]

        foot_rel_delta = foot_rel_w - initial_foot_rel_w

        # 양수면 발이 초기 위치보다 뒤로 빠진 것
        foot_backward_amount = torch.sum(
            foot_rel_delta * back_axis,
            dim=-1,
        )  # [num_envs, num_feet]

        # allowance 이상 뒤로 빠진 것만 penalty
        foot_backward_error = torch.clamp(
            foot_backward_amount - self.cfg.foot_backward_allowance,
            min=0.0,
        )

        foot_backward_penalty = self.cfg.foot_backward_penalty_scale * torch.sum(
            foot_backward_error ** 2,
            dim=-1,
        )





        # # 시작 방향과 비슷하면 1, 많이 틀어지면 0
        # foot_orientation_reward = torch.exp(
        #     -((foot_angle_error_mean / self.cfg.foot_orientation_sigma) ** 2)
        # )

        # # 몸이 완전히 무너진 상태에서 발 방향만 맞추는 꼼수 방지
        # foot_orientation_reward = foot_orientation_reward * posture_gate



        foot_quat_w = self._get_foot_quat_w()
        initial_foot_quat_w = self._initial_foot_quat_w

        num_feet = foot_quat_w.shape[1]

        foot_angle_error = quat_angle_diff_wxyz(
            foot_quat_w.reshape(-1, 4),
            initial_foot_quat_w.reshape(-1, 4),
        ).reshape(num_envs, num_feet)

        foot_angle_error_mean = foot_angle_error.mean(dim=-1)
        foot_angle_error_max = foot_angle_error.max(dim=-1).values

        # 시작 방향과 비슷하면 1, 많이 틀어지면 0에 가까워짐
        foot_orientation_reward = torch.exp(
            -((foot_angle_error_mean / self.cfg.foot_orientation_sigma) ** 2)
        )

        # 한쪽 발이라도 너무 많이 틀어지면 reward를 아예 0으로 만듦
        foot_orientation_reward = torch.where(
            foot_angle_error_max < self.cfg.foot_orientation_cutoff,
            foot_orientation_reward,
            torch.zeros_like(foot_orientation_reward),
        )

        # 몸이 무너진 상태에서 발 방향만 맞추는 꼼수 방지
        foot_orientation_reward = foot_orientation_reward * posture_gate






        # --------------------------------------------------
        # penalties
        # --------------------------------------------------
        upward_jump = torch.clamp(delta_pos_w[:, 2], min=0.0)
        downward_drop = torch.clamp(-delta_pos_w[:, 2], min=0.0)

        vertical_motion_penalty = 80.0 * upward_jump + 120.0 * downward_drop

        side_dist = torch.abs(delta_pos_xy[:, 1])
        side_motion_penalty = self.cfg.side_motion_penalty_scale * side_dist

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





        # 2.00 아래로 내려가기 시작하면 손해
        low_height_penalty = torch.clamp(2.00 - torso_z, min=0.0) ** 2

        # no_progress는 꺼져 있음
        target_forward_dist = 0.0035
        no_progress_error = torch.clamp(
            target_forward_dist - forward_dist,
            min=0.0,
        )

        no_progress_penalty = (
            self.cfg.no_progress_penalty_scale
            * (no_progress_error / target_forward_dist)
            * stand_gate
        )

        # --------------------------------------------------
        # total reward
        # --------------------------------------------------
        total_reward = (
            forward_dist_reward
            + alive_reward
            + up_reward
            + heading_reward
            + self.cfg.foot_orientation_reward_weight * foot_orientation_reward

            - backward_dist_penalty
            - no_progress_penalty
            - side_motion_penalty
            - self.cfg.vertical_motion_penalty_cost_scale * vertical_motion_penalty

            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * electricity_cost
            - 0.1 * action_rate_cost

            - 4.0 * body_pitch_penalty
            - 2.5 * joint_limit_penalty
            - 80.0 * low_height_penalty
            - torso_tilt_penalty
            - foot_backward_penalty * self.cfg.foot_backward_penalty_cost
        )

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
                f"root_pos={root_pos_w[env_id].detach().cpu().tolist()}, "
                f"delta_pos={delta_pos_w[env_id].detach().cpu().tolist()}, "
                f"move_axis={self._world_move_axis[0].detach().cpu().tolist()}, "
                f"forward_dist={forward_dist[env_id].item():.5f}, "
                f"height_gate={height_gate[env_id].item():.4f}, "
                f"height_keep_gate={height_keep_gate[env_id].item():.4f}, "
                f"posture_gate={posture_gate[env_id].item():.4f}, "
                f"heading_gate={heading_gate[env_id].item():.4f}, "
                f"joint_gate={joint_gate[env_id].item():.4f}, "
                f"stand_gate={stand_gate[env_id].item():.4f}, "
                f"strict_forward_gate={strict_forward_gate[env_id].item():.4f}, "
                f"foot_angle_error={foot_angle_error_mean[env_id].item():.4f}, "
                f"foot_orientation_reward={foot_orientation_reward[env_id].item():.4f}"
                f"foot_backward_amount={foot_backward_amount.mean().item():.4f}, "
                f"foot_backward_penalty={foot_backward_penalty.mean().item():.4f}, "
            )

            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}][reward] "
                f"total={total_reward.mean().item():.4f}, "
                f"old_progress={old_progress_reward.mean().item():.4f}, "

                f"forward_dist={forward_dist.mean().item():.5f}, "
                f"forward_dist_reward={forward_dist_reward.mean().item():.4f}, "
                f"backward_dist_penalty={backward_dist_penalty.mean().item():.4f}, "

                f"alive={alive_reward.mean().item():.4f}, "
                f"up={up_reward.mean().item():.4f}, "
                f"heading={heading_reward.mean().item():.4f}, "

                f"foot_angle_error_mean={foot_angle_error_mean.mean().item():.4f}, "
                f"foot_angle_error_max={foot_angle_error_max.mean().item():.4f}, "
                f"foot_orientation_reward={foot_orientation_reward.mean().item():.4f}, "

                f"side_motion_penalty={side_motion_penalty.mean().item():.4f}, "
                f"vertical_motion_penalty={vertical_motion_penalty.mean().item():.4f}, "

                f"joint_limit_penalty={joint_limit_penalty.mean().item():.4f}, "
                f"low_height_penalty={low_height_penalty.mean().item():.4f}, "
                f"body_pitch_penalty={body_pitch_penalty.mean().item():.4f}, "

                f"action_rate_cost={action_rate_cost.mean().item():.4f}, "
                f"act_cost={actions_cost.mean().item():.4f}, "
                f"elec_cost={electricity_cost.mean().item():.4f}, "
                f"no_progress_penalty={no_progress_penalty.mean().item():.4f}"
                f"foot_backward_amount_env0={foot_backward_amount[env_id].detach().cpu().tolist()}, "
                f"foot_backward_penalty_env0={foot_backward_penalty[env_id].item():.4f}, "


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