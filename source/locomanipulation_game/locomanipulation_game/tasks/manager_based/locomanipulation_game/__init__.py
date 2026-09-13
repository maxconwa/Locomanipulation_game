import gymnasium as gym

from . import agents

gym.register(
    id="Legs-R0-v0",
    entry_point=f"{__name__}.legs_r0_env_cfg:PositiveRewardRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legs_r0_env_cfg:LocoManipulationLegsR0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocoManipulationLegsR0PPORunnerCfg",
    },
)

gym.register(
    id="Upper-R0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upper_r0_env_cfg:LocoManipulationUpperR0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocoManipulationUpperR0PPORunnerCfg",
    },
)