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

# 실제 policy/action으로 제어할 관절 6개
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


LIKU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIKU_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 필요하면 z는 다시 조정
        pos=(0.0, 0.0, 2.55),

        # USD 축 자체를 안 믿고 reward 축을 따로 쓰는 방식이므로
        # 일단 identity로 둔다.
        # rot=(0.0, 0.0, 0.0, 1.0),
        rot = (0.0871557, 0.0, 0.7044160, 0.7044160),

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
            effort_limit_sim=200.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------

@configclass
class LikuEnvCfg(DirectRLEnvCfg):
    # Episode settings
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # 다리 6개만 제어
    action_space = 6
    observation_space = 48
    state_space = 0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

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
        num_envs=128, env_spacing=4.0, replicate_physics=True
    )

    robot: ArticulationCfg = LIKU_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 다리 6개용 기어
    joint_gears: list = [15.0] * 6

    # reward
    heading_weight: float = 0.5
    up_weight: float = 0.3
    energy_cost_scale: float = 0.01
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.2
    dof_vel_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 1.6
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    # -------------------------------------------------------------------------
    # CUSTOM AXIS
    # 아래 2줄이 핵심. 실행 후 [AXIS_TEST] 로그 보고 바꿔라.
    # 예: +X가 up=+1에 가깝게 나오면 robot_up_axis_local = (1,0,0)
    # -------------------------------------------------------------------------
    robot_up_axis_local = (0.0, 1.0, 0.0)
    robot_forward_axis_local = (-1.0, 0.0, 0.0)

    # 월드 기준
    world_up_axis = (0.0, 0.0, 1.0)
    world_forward_axis = (1.0, 0.0, 0.0)

    # 자세가 많이 틀어지면 죽이기
    bad_posture_threshold = 0.7


# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------

class LikuEnv(LocomotionEnv):
    cfg: LikuEnvCfg

    def __init__(self, cfg: LikuEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug_every = 60
        self._debug_env_id = 0

        # 다리 6개 관절 인덱스만 추출
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

        # custom axis tensors
        self._robot_up_axis_local = torch.tensor(
            self.cfg.robot_up_axis_local, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self._robot_forward_axis_local = torch.tensor(
            self.cfg.robot_forward_axis_local, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self._world_up_axis = torch.tensor(
            self.cfg.world_up_axis, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self._world_forward_axis = torch.tensor(
            self.cfg.world_forward_axis, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        logger.info(f"[LIKU] controlled joints = {self._joint_names}")
        logger.info(f"[LIKU] controlled joint ids = {self._joint_dof_idx}")
        logger.info(f"[LIKU] action_space = {self.cfg.action_space}")
        logger.info(f"[LIKU] observation_space = {self.cfg.observation_space}")
        logger.info(f"[LIKU] joint_gears = {self.joint_gears}")
        logger.info(f"[LIKU] robot_up_axis_local = {self.cfg.robot_up_axis_local}")
        logger.info(f"[LIKU] robot_forward_axis_local = {self.cfg.robot_forward_axis_local}")

    # -------------------------------------------------------------------------
    # custom axis helpers
    # -------------------------------------------------------------------------

    def _get_root_quat_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_quat_w"):
            return self.robot.data.root_quat_w
        return self.robot.data.root_state_w[:, 3:7]

    def _get_custom_axis_proj(self):
        root_quat = self._get_root_quat_w()
        num_envs = root_quat.shape[0]

        up_local = self._robot_up_axis_local.expand(num_envs, -1)
        fwd_local = self._robot_forward_axis_local.expand(num_envs, -1)
        world_up = self._world_up_axis.expand(num_envs, -1)
        world_fwd = self._world_forward_axis.expand(num_envs, -1)

        up_world = quat_rotate_wxyz(root_quat, up_local)
        fwd_world = quat_rotate_wxyz(root_quat, fwd_local)

        up_world = F.normalize(up_world, dim=-1)
        fwd_world = F.normalize(fwd_world, dim=-1)
        world_up = F.normalize(world_up, dim=-1)
        world_fwd = F.normalize(world_fwd, dim=-1)

        custom_up_proj = torch.sum(up_world * world_up, dim=-1)
        custom_heading_proj = torch.sum(fwd_world * world_fwd, dim=-1)

        return custom_up_proj, custom_heading_proj

    def _debug_axis_candidates(self):
        q = self._get_root_quat_w()[0:1]

        candidates = torch.tensor(
            [
                [ 1.0,  0.0,  0.0],   # +X
                [-1.0,  0.0,  0.0],   # -X
                [ 0.0,  1.0,  0.0],   # +Y
                [ 0.0, -1.0,  0.0],   # -Y
                [ 0.0,  0.0,  1.0],   # +Z
                [ 0.0,  0.0, -1.0],   # -Z
            ],
            dtype=torch.float32,
            device=self.device,
        )

        names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        q6 = q.expand(6, -1)
        rotated = quat_rotate_wxyz(q6, candidates)

        world_up = self._world_up_axis.expand(6, -1)
        world_fwd = self._world_forward_axis.expand(6, -1)

        rotated = F.normalize(rotated, dim=-1)
        world_up = F.normalize(world_up, dim=-1)
        world_fwd = F.normalize(world_fwd, dim=-1)

        up_scores = torch.sum(rotated * world_up, dim=-1)
        fwd_scores = torch.sum(rotated * world_fwd, dim=-1)

        msg = []
        for i in range(6):
            msg.append(
                f"{names[i]}: up={up_scores[i].item():+.3f}, fwd={fwd_scores[i].item():+.3f}"
            )

        logger.info("[AXIS_TEST] " + " | ".join(msg))

    # -------------------------------------------------------------------------
    # RL step hooks
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)

        if self.common_step_counter % self._debug_every == 0:
            a = self.actions[self._debug_env_id].detach().cpu()
            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}] "
                f"action mean={a.mean().item():.4f}, std={a.std().item():.4f}, "
                f"min={a.min().item():.4f}, max={a.max().item():.4f}, "
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








    # 원래라면 이걸 써야함(토탈 reward)


    # def _get_rewards(self) -> torch.Tensor:
    #     custom_up_proj, custom_heading_proj = self._get_custom_axis_proj()

    #     progress_reward = self.potentials - self.prev_potentials

    #     heading_reward = torch.where(
    #         custom_heading_proj > 0.8,
    #         torch.ones_like(custom_heading_proj) * self.cfg.heading_weight,
    #         self.cfg.heading_weight * custom_heading_proj / 0.8,
    #     )

    #     up_reward = torch.where(
    #         custom_up_proj > 0.93,
    #         torch.ones_like(custom_up_proj) * self.cfg.up_weight,
    #         torch.zeros_like(custom_up_proj),
    #     )

    #     actions_cost = torch.sum(self.actions ** 2, dim=-1)
    #     leg_dof_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]

    #     electricity_cost = torch.sum(
    #         torch.abs(self.actions * leg_dof_vel * self.cfg.dof_vel_scale),
    #         dim=-1,
    #     )

    #     alive_reward = torch.ones_like(progress_reward) * self.cfg.alive_reward_scale

    #     total_reward = (
    #         progress_reward
    #         + alive_reward
    #         + up_reward
    #         + heading_reward
    #         - self.cfg.actions_cost_scale * actions_cost
    #         - self.cfg.energy_cost_scale * electricity_cost
    #     )

    #     died = self.torso_position[:, 2] < self.cfg.termination_height
    #     total_reward = torch.where(
    #         died,
    #         torch.ones_like(total_reward) * self.cfg.death_cost,
    #         total_reward,
    #     )

    #     if self.common_step_counter % self._debug_every == 0:
    #         logger.info(
    #             f"[AXIS_CUSTOM] up_proj={custom_up_proj[0].item():.4f}, "
    #             f"heading_proj={custom_heading_proj[0].item():.4f}"
    #         )
    #         logger.info(
    #             f"[AXIS] root_quat_w={self._get_root_quat_w()[0].detach().cpu().tolist()}"
    #         )
    #         self._debug_axis_candidates()

    #         logger.info(
    #             f"[LIKU][step {int(self.common_step_counter)}][reward] "
    #             f"total={total_reward.mean().item():.4f}, "
    #             f"progress={progress_reward.mean().item():.4f}, "
    #             f"alive={alive_reward.mean().item():.4f}, "
    #             f"up={up_reward.mean().item():.4f}, "
    #             f"heading={heading_reward.mean().item():.4f}, "
    #             f"act_cost={actions_cost.mean().item():.4f}, "
    #             f"elec_cost={electricity_cost.mean().item():.4f}"
    #         )

    #     return total_reward






    def _get_rewards(self) -> torch.Tensor:
        custom_up_proj, _ = self._get_custom_axis_proj()

        # custom_up_proj 의미:
        # +1.0 = 로봇 머리/상체 방향이 월드 위쪽(+Z)과 완전히 일치
        #  0.0 = 옆으로 누움
        # -1.0 = 거꾸로 뒤집힘
        up_reward = torch.clamp(custom_up_proj, min=0.0, max=1.0)

        total_reward = up_reward

        if self.common_step_counter % self._debug_every == 0:
            up_angle_deg = torch.acos(
                torch.clamp(custom_up_proj, -1.0, 1.0)
            ) * 180.0 / torch.pi

            logger.info(
                f"[LIKU][step {int(self.common_step_counter)}][stand_test_reward] "
                f"total={total_reward.mean().item():.4f}, "
                f"up_proj={custom_up_proj.mean().item():.4f}, "
                f"up_angle_deg={up_angle_deg.mean().item():.2f}"
            )

            logger.info(
                f"[AXIS] root_quat_w={self._get_root_quat_w()[0].detach().cpu().tolist()}"
            )
            self._debug_axis_candidates()

        return total_reward












    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        custom_up_proj, _ = self._get_custom_axis_proj()

        bad_posture = custom_up_proj < self.cfg.bad_posture_threshold
        died = (self.torso_position[:, 2] < self.cfg.termination_height) | bad_posture
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.common_step_counter % self._debug_every == 0:
            logger.info(
                f"[DONE_CUSTOM] torso_z_mean={self.torso_position[:, 2].mean().item():.4f}, "
                f"up_proj_mean={custom_up_proj.mean().item():.4f}, "
                f"died_ratio={died.float().mean().item():.4f}, "
                f"timeout_ratio={time_out.float().mean().item():.4f}"
            )

        return died, time_out

    def _get_observations(self) -> dict:
        obs = super()._get_observations()

        if self.common_step_counter == 0:
            if isinstance(obs, dict):
                for k, v in obs.items():
                    logger.info(f"[LIKU][OBS] {k}: shape={tuple(v.shape)}")
            else:
                logger.info(f"[LIKU][OBS] shape={tuple(obs.shape)}")

        return obs