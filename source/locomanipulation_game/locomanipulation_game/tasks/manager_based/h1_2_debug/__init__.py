"""H1-2 debug task: spawns the robot with no locomotion rewards, for zero/random agent checks."""

import gymnasium as gym

gym.register(
    id="LocoManip-H12-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_2_debug_env_cfg:H12DebugEnvCfg",
    },
)