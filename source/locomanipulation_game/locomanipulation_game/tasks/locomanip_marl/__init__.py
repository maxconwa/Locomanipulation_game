"""Two-agent loco-manipulation on the H1-2: the legs walk on a velocity command as Unitree's sport mode does, the arms
track wrist poses. MAPPO via skrl. LocoManip-LegsOnly-v0 and LocoManip-ArmsOnly-v0 train one agent's task each;
LocoManip-Marl-Direct-v0 trains both at once, each on its own task (marl_direct_env_cfg.py); LocoManip-Marl-Pulse-v0
is that task with each agent acting in the latent of its frozen stage-1 decoder from pretraining/
(marl_pulse_env_cfg.py)."""

import gymnasium as gym

from . import agents

gym.register(
    id="LocoManip-Marl-Direct-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_direct_env_cfg:MarlDirectEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-LegsOnly-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_only_env_cfg:LegsOnlyEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-ArmsOnly-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arms_only_env_cfg:ArmsOnlyEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-Marl-Pulse-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_pulse_env_cfg:MarlPulseEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_pulse_cfg",
    },
)
