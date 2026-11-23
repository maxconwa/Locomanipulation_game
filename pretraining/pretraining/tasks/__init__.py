"""The Isaac Lab tracking env. Importing this package registers its gym ids (``import pretraining`` does it too);
its modules need the running app."""

import gymnasium as gym

# The tracking env amass_s1 trained in (track_env_cfg); "LocoManip-" keeps it in scripts/list_envs.py.
gym.register(
    id="LocoManip-Pretrain-Track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "pretraining.tasks.track_env_cfg:TrackEnvCfg"},
)
gym.register(
    id="LocoManip-Pretrain-Track-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "pretraining.tasks.track_env_cfg:TrackEnvCfg_PLAY"},
)
