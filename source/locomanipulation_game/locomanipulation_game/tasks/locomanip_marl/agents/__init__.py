"""skrl agent configs: skrl_mappo_cfg.yaml, and skrl_mappo_pulse_cfg (LocoManip-Marl-Pulse-v0's) built from it."""

import os

import yaml

_MAPPO_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skrl_mappo_cfg.yaml")
PULSE_DISCOUNT_FACTOR = 0.99  # the yaml's own comment's value at 50 Hz (the Pulse task's rate; 0.995 at 100 Hz)


def skrl_mappo_pulse_cfg() -> dict:
    """skrl_mappo_cfg.yaml at discount_factor PULSE_DISCOUNT_FACTOR: the same horizon in seconds at 50 Hz."""
    with open(_MAPPO_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["agent"]["discount_factor"] = PULSE_DISCOUNT_FACTOR
    return cfg
