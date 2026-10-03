"""Two-agent loco-manipulation: legs track a velocity command, arms track wrist poses. IPPO via skrl."""

import gymnasium as gym

from . import agents

gym.register(
    id="LocoManip-Marl-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanip_marl_env_cfg:LocoManipMarlEnvCfg",
        # IPPO only: without --algorithm IPPO, skrl's train.py would look up
        # skrl_cfg_entry_point (PPO) and merge both agents into one.
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)
