"""skrl agent configs: skrl_mappo_cfg.yaml, and skrl_mappo_pulse_cfg (LocoManip-Marl-Pulse-v0's) built from it."""

import os

import yaml

_MAPPO_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skrl_mappo_cfg.yaml")
# the yaml's own comment's values at 50 Hz (the Pulse task's rate; 0.995 and 0.975 at 100 Hz): the same horizons in s
PULSE_DISCOUNT_FACTOR = 0.99
PULSE_GAE_LAMBDA = 0.95


def skrl_mappo_pulse_cfg() -> dict:
    """skrl_mappo_cfg.yaml at PULSE_DISCOUNT_FACTOR and PULSE_GAE_LAMBDA: the same horizons in seconds at 50 Hz."""
    with open(_MAPPO_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["agent"]["discount_factor"] = PULSE_DISCOUNT_FACTOR
    cfg["agent"]["gae_lambda"] = PULSE_GAE_LAMBDA
    return cfg
