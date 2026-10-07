"""Export a MAPPO checkpoint of the legs+arms task for GOLEM's locomotion_game_deploy package.

Writes into --out (a policies/<name> directory of the deploy package):
  legs.pt, arms.pt   TorchScript: skrl's observation scaler (RunningStandardScaler, eps 1e-8, clip +-5), the
                     policy MLP's mean, and the env's +-clip_actions clamp. Raw observation in, applied raw action out.
  estimator.pt       TorchScript: the pelvis-motion estimator (input norm + MLP) saved with the checkpoint.
  export.yaml        per-checkpoint values the node loads over its template: the navigation rest pose (the run's
                     arm target table default_poses) and the source files.
Needs only torch; no simulator.

    python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
        --out <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_golem
"""

import argparse
import glob
import os
import re
from datetime import datetime

import torch
import torch.nn as nn
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Flat-Golem-Direct-v0", help="Recorded in export.yaml.")
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
for agent in ("legs", "arms"):
    pre = ckpt[agent]["observation_preprocessor"]
    actor = Actor(pre["running_mean"], pre["running_variance"], mlp(ckpt[agent]["policy"], "net_container"),
                  1e-8, 5.0, args.clip_actions).eval()
    scripted = torch.jit.script(actor)
    scripted.save(os.path.join(args.out, f"{agent}.pt"))
    print(f"{agent}: {pre['running_mean'].shape[0]} obs -> {actor.net[-1].out_features} actions")

step = re.search(r"_(\d+)\.pt$", ckpt_path)
est_path = os.path.join(run_dir, "estimator", f"estimator_{step.group(1)}.pt") if step else None
if not est_path or not os.path.isfile(est_path):
    est_path = sorted(glob.glob(os.path.join(run_dir, "estimator", "estimator_*.pt")), key=os.path.getmtime)[-1]
est = torch.load(est_path, map_location="cpu", weights_only=False)["model"]
estimator = Estimator(est["norm.mean"], est["norm.var"], mlp(est, "net")).eval()
torch.jit.script(estimator).save(os.path.join(args.out, "estimator.pt"))
print(f"estimator: {est['norm.mean'].shape[0]} inputs ({est_path})")

tables = sorted(glob.glob(os.path.join(run_dir, "arm_target_tables*.pt")))
rest = None
if tables:
    data = torch.load(tables[-1], map_location="cpu", weights_only=False)
    rest = [[round(float(v), 6) for v in pose] for pose in data["default_poses"]]
export = {
    "rest_pose": rest,
    "source": {"checkpoint": ckpt_path, "estimator": est_path, "target_tables": tables[-1] if tables else None,
               "task": args.task, "exported": datetime.now().isoformat(timespec="seconds")},
}
with open(os.path.join(args.out, "export.yaml"), "w") as f:
    yaml.safe_dump(export, f, sort_keys=False)
print(f"wrote {args.out}: legs.pt arms.pt estimator.pt export.yaml")
