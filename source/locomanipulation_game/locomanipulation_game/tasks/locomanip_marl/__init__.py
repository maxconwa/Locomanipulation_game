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
    id="LocoManip-StandEnergy-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_energy_env_cfg:StandEnergyEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-AmpWalk-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_walk_env_cfg:AmpWalkEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-AmpStyleOnly-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_walk_env_cfg:AmpStyleOnlyEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-AmpRsi-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_walk_env_cfg:AmpRsiEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-AmpAirTime-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_walk_env_cfg:AmpAirTimeEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-LegsPassAmp-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_walk_env_cfg:LegsPassAmpEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-LegsSquare-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_square_env_cfg:LegsSquareEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-LegsSquareAmp-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_square_env_cfg:LegsSquareAmpEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="LocoManip-LegsBalanceAmp-v0",
    entry_point=f"{__name__}.locomanip_marl_env:LocoManipMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_square_env_cfg:LegsBalanceAmpEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
