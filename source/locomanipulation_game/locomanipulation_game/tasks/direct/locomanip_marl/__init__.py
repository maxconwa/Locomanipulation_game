"""Two-agent loco-manipulation: legs track a velocity command, arms track wrist poses. IPPO / MAPPO via skrl."""

import gymnasium as gym

from . import agents

# IPPO or MAPPO only: without --algorithm IPPO|MAPPO, skrl's train.py would look
# up skrl_cfg_entry_point (PPO) and merge both agents into one.
_AGENT_CFGS = {
    "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
}

gym.register(
    id="LocoManip-Marl-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlEnvCfg", **_AGENT_CFGS},
)

# Navigation paused: stand and reach, for an emergent crouch.
gym.register(
    id="LocoManip-Marl-Stand-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlStandEnvCfg", **_AGENT_CFGS},
)

# Flat world, fresh policies: walk first, then alternate walking with crouch-reaching.
gym.register(
    id="LocoManip-Marl-Flat-Curriculum-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlFlatCurriculumEnvCfg",
        **_AGENT_CFGS,
    },
)

# The flat curriculum with IK-residual arms that observe their error, and the drop axis promoted on accuracy.
gym.register(
    id="LocoManip-Marl-Flat-IK-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlFlatIKEnvCfg",
        **_AGENT_CFGS,
    },
)

# The IK task with arm goals in front, spread evenly, low goals built in a feet-flat squat, and held until they end.
gym.register(
    id="LocoManip-Marl-Flat-IK2-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlFlatIK2EnvCfg",
        **_AGENT_CFGS,
    },
)

gym.register(
    id="LocoManip-Marl-Flat-IK3-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlFlatIK3EnvCfg",
        **_AGENT_CFGS,
    },
)

gym.register(
    id="LocoManip-Marl-Flat-Golem-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlFlatGolemEnvCfg",
        **_AGENT_CFGS,
    },
)
