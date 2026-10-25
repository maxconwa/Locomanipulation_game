"""Export a checkpoint of the two-agent task for GOLEM's lmg_deploy package.

Writes into --out (a policies/<name> directory of the deploy package):
  legs.pt, arms.pt   TorchScript actors: skrl's observation scaler (RunningStandardScaler, eps 1e-8, clip +-5), the
                     LSTM (its cell state capped as in training, c <- s tanh(c / s)) and the MLP's mean, and the env's
                     +-clip_actions clamp, as forward(obs, h, c) -> (action, h, c) with hidden_size and num_layers
                     attributes; the deploy node carries h, c and zeroes them at engage.
  <name>.yaml        the deploy template (agent_scripts/hw_deploy_template.yaml): what no run changes.
  export.yaml        what the run trained with, which the node loads over the template: the navigation rest pose
                     (the arm target tables' default_poses), target_limits (joint -> [low, high], the run's clip of
                     every joint target into GOLEM's position e-stop range), gait_phase_cos, imu (the body its IMU
                     sat on and the gyro low-pass), and (mujoco_evals.common.read_deploy_settings) control_dt, kp, kd,
                     default_joint_pos, the per-joint action scales, max_ik_step, gait_period, zero_cmd_eps,
                     command_ranges and command_filter_tau; and the source files.
Needs only torch and pyyaml; no simulator.

    python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
        --out ~/GOLEM/core_ws/src/lmg_deploy/policies/<name>
"""

import argparse
import glob
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mujoco_evals.common import (  # noqa: E402  (standard library and yaml only)
    _yaml_load_any,
    read_deploy_settings,
    read_imu,
    read_target_limits,
)

TEMPLATE = REPO / "agent_scripts" / "hw_deploy_template.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0", help="Recorded in export.yaml.")
parser.add_argument("--clip_actions", type=float, default=10.0, help="The env's cfg.clip_actions.")
args = parser.parse_args()


class RecurrentActor(nn.Module):
    hidden_size: int
    num_layers: int
    cell_cap: float

    def __init__(self, mean, var, lstm: nn.LSTM, layers: nn.Sequential, skip_observations: bool, epsilon: float,
                 clip_obs: float, clip_actions: float, cell_cap: float = 0.0):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", torch.sqrt(var.float()))
        self.lstm, self.net = lstm, layers
        self.hidden_size, self.num_layers = lstm.hidden_size, lstm.num_layers
        self.skip_observations = skip_observations
        self.cell_cap = float(cell_cap)
        self.epsilon, self.clip_obs, self.clip_actions = epsilon, clip_obs, clip_actions

    def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.clamp((obs - self.mean) / (self.std + self.epsilon), -self.clip_obs, self.clip_obs)
        out, (h, c) = self.lstm(z.unsqueeze(1), (h, c))
        if self.cell_cap > 0.0:
            c = self.cell_cap * torch.tanh(c / self.cell_cap)
        features = out.squeeze(1)
        if self.skip_observations:
            features = torch.cat([features, z], dim=-1)
        return torch.clamp(self.net(features), -self.clip_actions, self.clip_actions), h, c


def mlp(state: dict, prefix: str) -> nn.Sequential:
    """Linear layers prefix.0, .2, ... with ELU between them."""
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


if os.path.basename(os.path.normpath(args.out)) == "export":
    raise SystemExit("--out: name the folder after the policy (policies/<name>), not 'export': <name>.yaml is the deploy yaml")
ckpt_path = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ckpt_path))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
os.makedirs(args.out, exist_ok=True)
for agent in ("legs", "arms"):
    pre, policy = ckpt[agent]["observation_preprocessor"], ckpt[agent]["policy"]
    if "lstm.weight_ih_l0" not in policy:
        raise ValueError(f"{agent}: not an LSTM actor (recurrent.LSTMGaussianPolicy)")
    n = pre["running_mean"].shape[0]
    layers = sum(1 for k in policy if k.startswith("lstm.weight_ih_l"))
    hidden = policy["lstm.weight_hh_l0"].shape[1]
    lstm = nn.LSTM(policy["lstm.weight_ih_l0"].shape[1], hidden, num_layers=layers, batch_first=True)
    lstm.load_state_dict({k[len("lstm."):]: v for k, v in policy.items() if k.startswith("lstm.")})
    net = mlp(policy, "net")
    actor = RecurrentActor(pre["running_mean"], pre["running_variance"], lstm, net, net[0].in_features == hidden + n,
                           1e-8, 5.0, args.clip_actions,
                           cell_cap=float(policy["cell_cap_value"][0]) if "cell_cap_value" in policy else 0.0).eval()
    torch.jit.script(actor).save(os.path.join(args.out, f"{agent}.pt"))
    print(f"{agent}: {n} obs -> LSTM {layers}x{hidden}{' + skip' if actor.skip_observations else ''}"
          f"{f' (c capped at {actor.cell_cap:g})' if actor.cell_cap > 0 else ''} -> {net[-1].out_features} actions")

env_yaml = Path(run_dir, "params", "env.yaml").read_text()
legs_terms = (_yaml_load_any(env_yaml).get("observations") or {}).get("legs") or {}
# the tables the run trained with: its env.yaml table_file (absolute, or in the run dir), else the newest in the run dir
m = re.search(r"^\s*table_file: (\S+)$", env_yaml, re.M)
named = os.path.join(run_dir, m.group(1)) if m else None
tables = [named] if named and os.path.isfile(named) else sorted(glob.glob(os.path.join(run_dir, "arm_target_tables*.pt")))
rest = None
if tables:
    data = torch.load(tables[-1], map_location="cpu", weights_only=False)
    rest = [[round(float(v), 6) for v in pose] for pose in data["default_poses"]]
limits = read_target_limits(Path(run_dir))

export = {
    "rest_pose": rest,
    "target_limits": {name: [round(low, 6), round(high, 6)] for name, (low, high) in limits.items()} if limits else None,
    # the actors observe each leg's gait phase cos after its sin (lmg_deploy controller: legs 97, arms 125)
    "gait_phase_cos": "gait_phase_cos" in legs_terms,
    "imu": read_imu(Path(run_dir)),
    **read_deploy_settings(Path(run_dir)),
    "source": {"checkpoint": ckpt_path, "target_tables": tables[-1] if tables else None, "task": args.task,
               "exported": datetime.now().isoformat(timespec="seconds")},
}
with open(os.path.join(args.out, "export.yaml"), "w") as f:
    yaml.safe_dump(export, f, sort_keys=False)
name = os.path.basename(os.path.normpath(args.out))
shutil.copyfile(TEMPLATE, os.path.join(args.out, f"{name}.yaml"))
print(f"wrote {args.out}: legs.pt arms.pt {name}.yaml export.yaml")
