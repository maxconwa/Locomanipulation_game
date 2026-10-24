"""Export a MAPPO checkpoint of the legs+arms task for GOLEM's lmg_deploy package (locomotion_game_deploy before its rename).

Writes into --out (a policies/<name> directory of the deploy package):
  legs.pt, arms.pt   TorchScript: skrl's observation scaler (RunningStandardScaler, eps 1e-8, clip +-5), the
                     policy MLP's mean, and the env's +-clip_actions clamp. Raw observation in, applied raw action out.
                     An LSTM actor (recurrent.LSTMGaussianPolicy) exports as forward(obs, h, c) -> (action, h, c)
                     with hidden_size and num_layers attributes; the deploy node carries h, c and zeroes them at reset.
  estimator.pt       TorchScript: the pelvis-motion estimator (input norm + MLP) saved with the checkpoint.
  export.yaml        per-checkpoint values the node loads over its template: the navigation rest pose (the run's
                     arm target table default_poses), target_limits (joint -> [low, high], the run's clip of every
                     joint target into GOLEM's position e-stop range; null for runs before it; the node must clip
                     targets into it as the env did), odometry (mode learned, the estimator's input window in
                     frames, and the run's vertical_from_feet and tilt_from_gravity: the node replaces the estimate's
                     vertical motion with the feet's and its tilt with gravity's, as the env does), height_drop_obs
                     (whether the actors take the height_drop input runs before golemlim had), gait_phase_cos
                     (whether they observe each leg's phase cos after its sin, from Tune3Track2 on), imu (the body the
                     run's IMU sat on and its offset: imu_link on the torso, or livox_link at the MID-360's chip),
                     the deploy template's per-run keys as the run trained them (common.read_deploy_settings:
                     control_dt, kp, kd, gait_period, zero_cmd_eps, command_ranges; and command_filter_tau, the command
                     low-pass the deploy does not apply yet) and the source files.
Needs only torch; no simulator.

    python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
        --out <GOLEM>/core_ws/src/lmg_deploy/policies/<policy>
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mujoco_evals.common import (  # noqa: E402  (standard library and yaml only)
    _yaml_load_any,
    read_deploy_settings,
    read_imu,
    read_target_limits,
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0", help="Recorded in export.yaml.")
parser.add_argument("--clip_actions", type=float, default=10.0, help="The env's cfg.clip_actions.")
args = parser.parse_args()


class Actor(nn.Module):
    def __init__(self, mean, var, layers: nn.Sequential, epsilon: float, clip_obs: float, clip_actions: float):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", torch.sqrt(var.float()))
        self.net = layers
        self.epsilon, self.clip_obs, self.clip_actions = epsilon, clip_obs, clip_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((obs - self.mean) / (self.std + self.epsilon), -self.clip_obs, self.clip_obs)
        return torch.clamp(self.net(z), -self.clip_actions, self.clip_actions)


class RecurrentActor(nn.Module):
    hidden_size: int
    num_layers: int

    def __init__(self, mean, var, lstm: nn.LSTM, layers: nn.Sequential, skip_observations: bool, epsilon: float,
                 clip_obs: float, clip_actions: float):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", torch.sqrt(var.float()))
        self.lstm, self.net = lstm, layers
        self.hidden_size, self.num_layers = lstm.hidden_size, lstm.num_layers
        self.skip_observations = skip_observations
        self.epsilon, self.clip_obs, self.clip_actions = epsilon, clip_obs, clip_actions

    def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.clamp((obs - self.mean) / (self.std + self.epsilon), -self.clip_obs, self.clip_obs)
        out, (h, c) = self.lstm(z.unsqueeze(1), (h, c))
        features = out.squeeze(1)
        if self.skip_observations:
            features = torch.cat([features, z], dim=-1)
        return torch.clamp(self.net(features), -self.clip_actions, self.clip_actions), h, c


class Estimator(nn.Module):
    def __init__(self, mean, var, layers: nn.Sequential):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("var", var.float())
        self.net = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / torch.sqrt(self.var + 1e-8))


def mlp(state: dict, prefix: str) -> nn.Sequential:
    """Linear layers prefix.0, .2, ... with ELU between them (skrl's net_container, the estimator's net)."""
    idx = sorted({int(k[len(prefix) + 1:].split(".")[0]) for k in state if k.startswith(prefix + ".")})
    layers = []
    for i, j in enumerate(idx):
        w, b = state[f"{prefix}.{j}.weight"], state[f"{prefix}.{j}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


ckpt_path = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ckpt_path))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
os.makedirs(args.out, exist_ok=True)
num_obs = {}
for agent in ("legs", "arms"):
    pre, policy = ckpt[agent]["observation_preprocessor"], ckpt[agent]["policy"]
    n = num_obs[agent] = pre["running_mean"].shape[0]
    if "lstm.weight_ih_l0" in policy:
        layers = sum(1 for k in policy if k.startswith("lstm.weight_ih_l"))
        hidden = policy["lstm.weight_hh_l0"].shape[1]
        lstm = nn.LSTM(policy["lstm.weight_ih_l0"].shape[1], hidden, num_layers=layers, batch_first=True)
        lstm.load_state_dict({k[len("lstm."):]: v for k, v in policy.items() if k.startswith("lstm.")})
        net = mlp(policy, "net")
        actor = RecurrentActor(pre["running_mean"], pre["running_variance"], lstm, net,
                               net[0].in_features == hidden + n, 1e-8, 5.0, args.clip_actions).eval()
        kind = f"LSTM {layers}x{hidden}{' + skip' if actor.skip_observations else ''} -> "
    else:
        actor = Actor(pre["running_mean"], pre["running_variance"], mlp(policy, "net_container"),
                      1e-8, 5.0, args.clip_actions).eval()
        kind = ""
    torch.jit.script(actor).save(os.path.join(args.out, f"{agent}.pt"))
    print(f"{agent}: {n} obs -> {kind}{actor.net[-1].out_features} actions")
# the legs' 95 inputs, + 1 with the height_drop input (always 0) that runs before golemlim had, + 2 with each leg's
# gait phase cos (the run's observations name gait_phase_cos)
legs_terms = (_yaml_load_any(Path(run_dir, "params", "env.yaml").read_text()).get("observations") or {}).get("legs") or {}
gait_phase_cos = "gait_phase_cos" in legs_terms
height_drop_obs = num_obs["legs"] - 2 * gait_phase_cos == 96
if num_obs["legs"] - 2 * gait_phase_cos not in (95, 96) or num_obs["arms"] - num_obs["legs"] != 28:
    raise ValueError(f"legs {num_obs['legs']}, arms {num_obs['arms']} observations (gait_phase_cos {gait_phase_cos}); "
                     "the deploy layout knows legs 95 or 96, + 2 with gait_phase_cos, and arms 28 more")

step = re.search(r"_(\d+)\.pt$", ckpt_path)
est_path = os.path.join(run_dir, "estimator", f"estimator_{step.group(1)}.pt") if step else None
if not est_path or not os.path.isfile(est_path):
    est_path = sorted(glob.glob(os.path.join(run_dir, "estimator", "estimator_*.pt")), key=os.path.getmtime)[-1]
est = torch.load(est_path, map_location="cpu", weights_only=False)["model"]
estimator = Estimator(est["norm.mean"], est["norm.var"], mlp(est, "net")).eval()
torch.jit.script(estimator).save(os.path.join(args.out, "estimator.pt"))
history_frames, rest = divmod(est["norm.mean"].shape[0] - 26, 75)  # 75 values per frame, then 26 actions
print(f"estimator: {est['norm.mean'].shape[0]} inputs, {history_frames} frames ({est_path})")

# the tables the run trained with: its env.yaml table_file (absolute, or in the run dir), else the newest in the run dir
m = re.search(r"^\s*table_file: (\S+)$", Path(run_dir, "params", "env.yaml").read_text(), re.M)
named = os.path.join(run_dir, m.group(1)) if m else None
tables = [named] if named and os.path.isfile(named) else sorted(glob.glob(os.path.join(run_dir, "arm_target_tables*.pt")))
rest = None
if tables:
    data = torch.load(tables[-1], map_location="cpu", weights_only=False)
    rest = [[round(float(v), 6) for v in pose] for pose in data["default_poses"]]
limits = read_target_limits(Path(run_dir))
env_yaml = Path(run_dir, "params", "env.yaml").read_text()


def env_flag(name: str) -> bool:
    m = re.search(rf"^\s*{name}: (true|false)$", env_yaml, re.M)
    return bool(m) and m.group(1) == "true"


export = {
    "rest_pose": rest,
    "target_limits": {name: [round(low, 6), round(high, 6)] for name, (low, high) in limits.items()} if limits else None,
    "odometry": {"mode": (re.search(r"^\s*odometry: (learned|legs)$", env_yaml, re.M) or [None, "learned"])[1],
                 "history_frames": history_frames,
                 "vertical_from_feet": env_flag("vertical_from_feet"), "tilt_from_gravity": env_flag("tilt_from_gravity")},
    "height_drop_obs": height_drop_obs,
    # the actors observe each leg's gait phase cos after its sin (lmg_deploy controller: legs 97, arms 125)
    "gait_phase_cos": gait_phase_cos,
    # the IMU the policy and estimator read: the torso's imu_link (LowState.imu_state) or the head's MID-360
    # (livox_link, or from the 10-19 update its chip frame livox_imu_link), and gyro_lowpass_hz when the gyro was
    # low-passed (the deploy filters only when the key is present)
    "imu": read_imu(Path(run_dir)),
    # control rate, PD gains, gait clock, stand gate, command ranges and filter: the template's keys, as trained
    **read_deploy_settings(Path(run_dir)),
    "source": {"checkpoint": ckpt_path, "estimator": est_path, "target_tables": tables[-1] if tables else None,
               "task": args.task, "exported": datetime.now().isoformat(timespec="seconds")},
}
with open(os.path.join(args.out, "export.yaml"), "w") as f:
    yaml.safe_dump(export, f, sort_keys=False)
print(f"wrote {args.out}: legs.pt arms.pt estimator.pt export.yaml")
