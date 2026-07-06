# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 기존 Humanoid 기본 예제는 그대로 주석 유지
# gym.register(
#     id="Isaac-Humanoid-Direct-v0",
#     entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.humanoid_env:HumanoidEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )


# -----------------------------------------------------------------------------
# 기존 torque / effort 방식 LIKU env
# -----------------------------------------------------------------------------

from .liku_env import LikuEnv, LikuEnvCfg

gym.register(
    id="Isaac-Liku-Direct-v0",
    entry_point=f"{__name__}:LikuEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LikuEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# -----------------------------------------------------------------------------
# 새 position target / ZMP imitation 방식 LIKU env
# -----------------------------------------------------------------------------

from .liku_imitation_env import LikuImitationEnv, LikuImitationEnvCfg

gym.register(
    id="Isaac-Liku-Imitation-Direct-v0",
    entry_point=f"{__name__}:LikuImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LikuImitationEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)