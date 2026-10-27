"""Two-agent loco-manipulation on the H1-2: the legs walk on a velocity command as Unitree's sport mode does, the arms
track wrist poses. MAPPO via skrl."""

import gymnasium as gym

from . import agents

gym.register(
    id="LocoManip-Marl-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-AlwaysStep-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.always_step_env_cfg:AlwaysStepEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
