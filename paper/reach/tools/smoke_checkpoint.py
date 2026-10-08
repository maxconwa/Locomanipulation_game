"""A pipeline self-test fixture: an skrl-format MAPPO checkpoint whose actors output zero.

    python paper/reach/tools/smoke_checkpoint.py [--out paper/reach/results/_smoke_run]

With zero actions the legs hold their default joint targets and the arms run the damped-least-squares IK step alone,
so a standing-reachable goal should be reached and a goal 1 m lower should not. It exercises eval_isaac.py,
eval_mujoco.py and the analysis without a trained run; its results are never plotted (label "_smoke").
No estimator is written: evaluate it with --odometry true.
"""

import argparse
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "_smoke_run"))
p.add_argument("--legs_obs", type=int, default=95)
p.add_argument("--arms_obs", type=int, default=123)
a = p.parse_args()


def agent(n_obs: int, n_act: int) -> dict:
    sizes = [n_obs, 256, 128, 64, n_act]
    policy = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        policy[f"net_container.{2 * i}.weight"] = torch.zeros(fan_out, fan_in)
        policy[f"net_container.{2 * i}.bias"] = torch.zeros(fan_out)
    policy["log_std_parameter"] = torch.zeros(n_act)
    return {"policy": policy,
            "observation_preprocessor": {"running_mean": torch.zeros(n_obs), "running_variance": torch.ones(n_obs),
                                         "current_count": torch.tensor(1.0)}}


out = Path(a.out)
(out / "checkpoints").mkdir(parents=True, exist_ok=True)
torch.save({"legs": agent(a.legs_obs, 12), "arms": agent(a.arms_obs, 14)}, out / "checkpoints" / "agent_0.pt")
print(f"[smoke_checkpoint] wrote {out / 'checkpoints' / 'agent_0.pt'}")
