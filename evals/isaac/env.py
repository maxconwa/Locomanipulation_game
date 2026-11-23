"""What the Isaac scorecards share: one agent loaded from each of two checkpoints, the check that a checkpoint's agent
fits the env it is put in, and the curriculum state merged from two runs.

The other body's policy (a legs eval given --arms_checkpoint, an arms eval given --legs_checkpoint) runs in
LocoManip-Marl-Direct-v0: the legs and the arms side by side on a free pelvis. The env (its overrides, IMU model,
domain randomization) is the main policy's run's; the other checkpoint gives only its agent's weights and that
agent's curriculum entries. Before the weights load, each agent's observation terms in the env are compared with
the ones its run trained on (params/env.yaml): a different term list or width stops the eval with the difference.
Imports only the standard library, torch and yaml (call after the app is up).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

MARL_DIRECT = "LocoManip-Marl-Direct-v0"
# the curriculum entries (locomanip_marl_env.save_state) that belong to each agent's command
AGENT_STATE_KEYS = {"legs": ("base_velocity.lin_kernel_stage", "base_velocity.ang_kernel_stage"),
                    "arms": ("arm_targets.kernel_stage", "arm_goals_enabled")}


def read_env_yaml(run_dir: str | Path) -> dict:
    from evals.mujoco.common import _yaml_load_any
    path = Path(run_dir) / "params" / "env.yaml"
    return _yaml_load_any(path.read_text()) if path.is_file() else {}


def run_obs_terms(run_dir: str | Path, agent: str) -> list[str] | None:
    """The agent's observation terms, in order, as its run trained (None: the run did not record them)."""
    group = (read_env_yaml(run_dir).get("observations") or {}).get(agent)
    if not isinstance(group, dict):
        return None
    return [k for k, v in group.items() if isinstance(v, dict)]


def env_obs_terms(base, agent: str) -> list[str]:
    return list(base.observation_manager.active_terms[agent])


def check_agent_fits(base, agent: str, run_dir: str | Path, checkpoint: str | Path) -> None:
    """Stop (SystemExit) when the checkpoint's agent did not train on the env's observation terms for it, or its
    network's input width differs from the env's."""
    trained = run_obs_terms(run_dir, agent)
    here = env_obs_terms(base, agent)
    if trained is not None and trained != here:
        missing = [t for t in trained if t not in here]
        extra = [t for t in here if t not in trained]
        raise SystemExit(f"[evals] {agent} from {checkpoint}: its run trained on observation terms {trained}; this env"
                         f" has {here} (missing here: {missing}, new here: {extra})")
    width = base.observation_manager.group_obs_dim[agent][0]
    policy = torch.load(str(checkpoint), map_location="cpu", weights_only=False).get(agent, {}).get("policy", {})
    w_in = next((v.shape[1] for k, v in policy.items() if k.endswith("weight_ih_l0")), None)
    if w_in is None:
        w_in = next((v.shape[1] for k, v in policy.items() if k.endswith(".weight") and v.dim() == 2), None)
    if w_in is not None and w_in != width:
        raise SystemExit(f"[evals] {agent} from {checkpoint}: its network reads {w_in} observation values, this env"
                         f" gives {width} ({here})")


def _imu(d: dict) -> dict:
    imu = d.get("lowstate_imu") or {}
    return {"orientation": imu.get("orientation") or "raw", "quat_bias_deg": imu.get("quat_bias_deg")}


SHARED_SETTINGS = {
    "torso IMU model (lowstate_imu)": _imu,
    "gyro low-pass (gyro_lowpass_hz)": lambda d: d.get("gyro_lowpass_hz"),
    "physics step (sim.dt)": lambda d: (d.get("sim") or {}).get("dt"),
    "decimation": lambda d: d.get("decimation"),
    "gait period": lambda d: d.get("gait_period"),
    "action delay (action_delay_range)": lambda d: d.get("action_delay_range"),
}


def check_runs_compatible(main_run: str | Path, other_run: str | Path) -> None:
    """Stop (SystemExit) when the two runs trained on different settings both agents depend on (the env is the main
    run's, so the other agent would see what it never trained on): the IMU model, rates, gait period, delay."""
    if Path(main_run).resolve() == Path(other_run).resolve():
        return
    a, b = read_env_yaml(main_run), read_env_yaml(other_run)
    diff = [f"{name}: {get(a)!r} vs {get(b)!r}" for name, get in SHARED_SETTINGS.items() if get(a) != get(b)]
    if diff:
        raise SystemExit(f"[evals] {Path(main_run).name} and {Path(other_run).name} trained on different settings"
                         f" the env would impose on both agents:\n  " + "\n  ".join(diff)
                         + "\nPick an other-body policy trained on the same ones.")


def load_agent(runner, checkpoint: str | Path, uid: str) -> None:
    """Load one agent (its policy and observation scaler; the critic too when its state size fits) from a
    checkpoint into a MarlRunner's agent; the other agent is left as it is."""
    agent = runner.agent
    modules = torch.load(str(checkpoint), map_location=agent.device, weights_only=False)
    if uid not in modules:
        raise SystemExit(f"[evals] {checkpoint} has no {uid!r} agent (it has {list(modules)})")
    data = dict(modules[uid])
    saved = data.get("state_preprocessor") or {}
    if "running_mean" in saved and saved["running_mean"].shape[0] != agent.state_spaces[uid].shape[0]:
        for name in ("value", "state_preprocessor", "optimizer"):
            data.pop(name, None)
    for name, state in data.items():
        module = agent.checkpoint_modules[uid].get(name)
        if module is not None:
            module.load_state_dict(state)
            if hasattr(module, "eval"):
                module.eval()


def merged_state(main_state: str | None, other_state: str | None, other_agent: str, out_dir: str | Path) -> str | None:
    """A curriculum state file: the main run's, with the other agent's entries from the other run's (so, e.g., an
    arms policy runs at the arm kernel stage it reached, with its goals on)."""
    if other_state is None:
        return main_state
    main = torch.load(main_state, map_location="cpu", weights_only=False)["env_state"] if main_state else {}
    other = torch.load(other_state, map_location="cpu", weights_only=False)["env_state"]
    merged = dict(main)
    for key in AGENT_STATE_KEYS[other_agent]:
        if key in other:
            merged[key] = other[key]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(str(out_dir), "curriculum_state_merged.pt")
    torch.save({"env_state": merged}, path)
    print(f"[INFO] curriculum state: {main_state} with the {other_agent} entries of {other_state}: {merged}")
    return path
