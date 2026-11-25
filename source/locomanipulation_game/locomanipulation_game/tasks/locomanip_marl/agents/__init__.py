"""skrl agent configs: skrl_mappo_cfg.yaml, and skrl_mappo_pulse_cfg (LocoManip-Marl-Pulse-v0's) built from it."""

import os

import yaml

_MAPPO_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skrl_mappo_cfg.yaml")


def skrl_mappo_pulse_cfg() -> dict:
    """skrl_mappo_cfg.yaml as it is: the Pulse task runs at main's 100 Hz (its stage-1 runs are on main's plant), so
    main's discount and lambda hold; PULSE's own settings (initial std, std bounds, output init scale) are the env
    cfg's (MarlPulseEnvCfg). Kept as the registry's entry point, so the task can diverge from the yaml in one place."""
    with open(_MAPPO_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)
