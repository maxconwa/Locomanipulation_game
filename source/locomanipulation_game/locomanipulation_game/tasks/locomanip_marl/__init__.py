"""Two-agent loco-manipulation: the legs track a velocity command, the arms wrist poses. MAPPO via skrl.
LocoManip-WholeBody-Direct-v0 is the same task with one agent for the whole body."""

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
    id="LocoManip-WholeBody-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipWholeBodyEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_wholebody_cfg.yaml",
    },
)
