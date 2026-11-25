"""Exported policies in MuJoCo: the actors (TorchScript, LSTM state carried), an export per checkpoint (cached), and
the settings of two exports run together (the legs of one, the arms of another).

An export (scripts/skrl/export_marl.py) always holds both legs.pt and arms.pt; which actors run is the caller's
choice. Two exports together: the legs' settings from the legs export, the arms' from the arms export (ARMS_KEYS;
kp, kd and target limits per joint from the export that owns the joint); the settings both bodies live by
(SHARED_KEYS) must be equal, or the combination is refused with the difference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import common as C

ARMS_KEYS = ("arms", "arm_goals", "arm_ik", "arm_residual_scales", "arm_seed_table", "arms_obs_terms", "ik_damping",
             "max_ik_step", "residual_cutoff_hz", "rest_pose", "torso_action_scale")
SHARED_KEYS = ("control_dt", "imu", "gait_period", "gait_phase_cos", "default_joint_pos", "command_filter_tau",
               "zero_cmd_eps", "clip_actions")
LEG_JOINTS, ARM_JOINTS = range(12), range(13, 27)


class Recurrent:
    """An exported actor called as actor(obs[None]) -> action[None], carrying its LSTM state."""

    def __init__(self, net):
        self.net = net
        self.recurrent = hasattr(net, "hidden_size")
        if self.recurrent:
            self.h = torch.zeros(net.num_layers, 1, net.hidden_size)
            self.c = torch.zeros_like(self.h)

    def __call__(self, x):
        if not self.recurrent:
            return self.net(x)
        a, self.h, self.c = self.net(x, self.h, self.c)
        return a

    def reset_state(self):
        if self.recurrent:
            self.h.zero_()
            self.c.zero_()

    def state_norm(self) -> float:
        return float(self.c.abs().mean()) if self.recurrent else 0.0


def actor(export_dir: str | Path, agent: str) -> Recurrent:
    return Recurrent(torch.jit.load(os.path.join(str(export_dir), f"{agent}.pt"), map_location="cpu").eval())


def export_for(checkpoint: str | Path, cache_root: Path) -> Path:
    """The checkpoint's export, made once (scripts/skrl/export_marl.py, torch and yaml only) into
    <cache_root>/export_<step>."""
    from evals.common import runs
    out = cache_root / f"export_{runs.step_of(checkpoint)}"
    if not (out / "export.yaml").is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(C.REPO / "scripts" / "skrl" / "export_marl.py"), "--checkpoint", str(checkpoint),
               "--out", str(out)]
        print(f"[evals] exporting: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    return out


def combined_settings(legs_export: str | Path | None, arms_export: str | Path | None) -> dict:
    """One settings dict for the sim: one export's, or the legs' and the arms' merged (module docstring)."""
    if legs_export is None or arms_export is None or Path(legs_export).resolve() == Path(arms_export).resolve():
        return C.load_deploy_cfg(str(legs_export or arms_export))
    legs, arms = C.load_deploy_cfg(str(legs_export)), C.load_deploy_cfg(str(arms_export))
    diff = [f"{k}: {legs.get(k)!r} vs {arms.get(k)!r}" for k in SHARED_KEYS if legs.get(k) != arms.get(k)]
    if diff:
        raise SystemExit(f"[evals] the legs export {legs_export} and the arms export {arms_export} disagree on settings"
                         " both bodies live by:\n  " + "\n  ".join(diff))
    out = dict(legs)
    for k in ARMS_KEYS:
        if k in arms:
            out[k] = arms[k]
        else:
            out.pop(k, None)
    for k in ("kp", "kd"):
        v = np.array(legs[k], float)
        v[list(ARM_JOINTS)] = np.array(arms[k], float)[list(ARM_JOINTS)]
        out[k] = v.tolist()
    limits = dict(legs.get("target_limits") or {})
    for name, rng in (arms.get("target_limits") or {}).items():
        if any(s in name for s in ("shoulder", "elbow", "wrist")):
            limits[name] = rng
    out["target_limits"] = limits
    return out
