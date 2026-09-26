import gymnasium as gym

from . import agents

gym.register(
    id="Legs-R0-v0",
    entry_point=f"{__name__}.reward_clip:PositiveRewardRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_r0_env_cfg:LocoManipulationLegsR0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocoManipulationLegsPPORunnerCfg",
    },
)
gym.register(
    id="Upper-Adv-Ri-v0",
    entry_point=f"{__name__}.reward_clip:NegativeRewardRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upper_adv_ri_env_cfg:LocoManipulationUpperAdvRiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocoManipulationUpperPPORunnerCfg",
    },
)

gym.register(
    id="Legs-Ri-v0",
    entry_point=f"{__name__}.reward_clip:PositiveRewardRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_ri_env_cfg:LocoManipulationLegsRiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocoManipulationLegsPPORunnerCfg",
    },
)
