"""Run the steerability benchmark on one arms system in Isaac Lab (design doc: https://claude.ai/artifact/9qn7ShQyeeuYccPNCw4wSL).

    python -m evals.isaac.arms.steerability.run_isaac --checkpoint <run>/checkpoints/agent_<N>.pt --system policy
        [--suite quick|full] [--split dev|test] [--modes nudge goal] [--freeze on|off] [--out <dir>]

Systems: policy (the checkpoint's arms policy, mean actions, on its controller); controller_integral (the same
controller, the arms' actions zero); controller (zero actions and no pose-error integral: the IK trajectory and the
gravity offset alone). The checkpoint's run overrides are replayed as evals/isaac/arms/get_scorecard.py does; the controller systems use it
only to set up the env. Bases: the split's seed draws goals from goal tables built for that seed (never a training
run's 42), half on the table rows and half off them (evals/isaac/arms/get_scorecard.py --offtable's). Writes results.npz, summary.md and
meta.json to --out (default logs/steerability/<split>/<system>[_freezeoff]).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

SUITES = {
    # bases, (translation mm, rotation deg) step pairs, repeats, shams per axis, settle s, approach s, repeat-test K,
    # outer-loop corrections
    "quick": dict(num_envs=32, scales=[(0.25, 0.25), (1.0, 1.0), (5.0, 5.0)], repeats=1, shams=1, settle_s=4.0,
                  approach_s=6.0, repeat_k=4, corrections=6),
    "full": dict(num_envs=128, scales=[(0.05, 0.05), (0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0),
                                       (5.0, 5.0), (10.0, 5.0)],
                 repeats=2, shams=2, settle_s=4.0, approach_s=6.0, repeat_k=8, corrections=10),
}
SPLIT_SEEDS = {"dev": 101, "test": 202}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--system", choices=["policy", "controller_integral", "controller"], required=True)
parser.add_argument("--task", default="LocoManip-ArmsOnly-v0")
parser.add_argument("--suite", choices=list(SUITES), default="quick")
parser.add_argument("--split", choices=list(SPLIT_SEEDS), default="dev")
parser.add_argument("--modes", nargs="+", choices=["nudge", "goal"], default=["nudge", "goal"])
parser.add_argument("--freeze", choices=["on", "off"], default="on")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    known_overrides,
    run_overrides,
    state_checkpoint_for,
)

from . import estimators as est  # noqa: E402
from . import summary  # noqa: E402
from .closed_loop import OuterLoop, run_closed_loop  # noqa: E402
from .isaac_arms import IsaacArms  # noqa: E402
from .runner import run  # noqa: E402
from .schedules import repeat_schedule, step_schedule  # noqa: E402

MM, DEG = 1e-3, np.pi / 180
suite = dict(SUITES[args_cli.suite])
if args_cli.num_envs:
    suite["num_envs"] = args_cli.num_envs
seed = SPLIT_SEEDS[args_cli.split]
checkpoint = os.path.abspath(args_cli.checkpoint)
run_dir = os.path.dirname(os.path.dirname(checkpoint))
replayed, dropped = known_overrides(run_overrides(run_dir), args_cli.task)
system_overrides = []
if args_cli.system == "controller":
    system_overrides.append("env.actions.arm_pos.integral_gain=0.0")
if args_cli.freeze == "off":
    system_overrides.append("env.actions.arm_pos.freeze_residual_on_goal=false")
sys.argv = [sys.argv[0]] + replayed + system_overrides + hydra_args
name = args_cli.system + ("_freezeoff" if args_cli.freeze == "off" else "")
out_dir = os.path.abspath(args_cli.out or os.path.join("logs", "steerability", args_cli.split, name))


def draw_bases(arm, action, robot) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Every env's base: a fresh goal from the command (half on the table rows, half off them), its seed's labels."""
    N = arm.num_envs
    ids = torch.arange(N, device=arm.device)
    arm.cfg.rest_goal_prob = 0.0
    offtable = torch.zeros(N, dtype=torch.bool, device=arm.device)
    offtable[N // 2:] = True
    for off in (False, True):
        arm.cfg.offtable_prob = 1.0 if off else 0.0
        arm._resample(ids[offtable == off])
    bases = arm.believed_b.cpu().numpy().astype(np.float64)
    q = arm.seed_q
    joints = torch.cat([torch.as_tensor(j) for j in arm._arm_joint_ids]).to(arm.device)
    limits = robot.data.soft_joint_pos_limits[:, joints].view(N, 2, -1, 2)
    margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q).amin(-1)
    kin = arm._kin
    p, R = kin._arm_links(q.to(kin.dtype))
    rw = action.cfg.rot_weight
    W = torch.tensor([1, 1, 1, rw, rw, rw], device=kin.device, dtype=kin.dtype)
    sigma_min = torch.linalg.svdvals(kin._jacobian(p, R) * W.view(6, 1))[..., -1]
    labels = {"margin": margin.cpu().numpy(), "sigma_min": sigma_min.cpu().numpy(),
              "offtable": offtable.cpu().numpy()[:, None].repeat(2, 1), "seed_q": q.cpu().numpy()}
    return bases, labels


def step_metrics(plan, result) -> tuple[dict, dict, dict]:
    """The step schedule's per-base metrics, its responses, and its + steps' dynamics (per item, NaN off phase 0)."""
    resp = est.step_responses(plan, result)
    per_base = summary.per_base_steps(plan, result, resp)
    M = len(plan.axis)
    settle = np.full(plan.axis.shape, np.nan)
    overshoot = np.full(plan.axis.shape, np.nan)
    for m in range(M):
        trace = np.take_along_axis(result.traces[1 + 4 * m], plan.axis[m][None, ..., None], -1)[..., 0]
        mag = plan.magnitude[m]
        band = np.where(plan.axis[m] < 3, np.maximum(0.5 * MM, 0.05 * mag), np.maximum(0.5 * DEG, 0.05 * mag))
        dyn = est.step_dynamics(trace.reshape(len(trace), -1), result.dt, band.reshape(-1))
        settle[m] = dyn["settle_time"].reshape(mag.shape)
        overshoot[m] = dyn["overshoot"].reshape(mag.shape)
    return per_base, resp, {"settle_time": settle, "overshoot": overshoot}


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = suite["num_envs"]
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = seed
    env_cfg.log_dir = out_dir  # the goal tables are built for this seed and saved here
    env_cfg.training = False
    env_cfg.state_checkpoint_path = state_checkpoint_for(checkpoint)
    env_cfg.episode_length_s = 1.0e5
    env_cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 1.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e7, 1.0e7), (1.0e7, 1.0e7)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e7, 1.0e7), 1.0
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    os.makedirs(out_dir, exist_ok=True)

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = MarlRunner(env, agent_cfg)
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    system = IsaacArms(env, runner, zero_actions=args_cli.system != "policy")
    system.reset()
    with torch.inference_mode():
        bases, labels = draw_bases(system.arm, system.action, system.robot)
    region = summary.regions(labels["margin"], labels["sigma_min"], labels["offtable"])
    print(f"[INFO] {len(bases)} bases; regions: {dict(zip(*np.unique(region, return_counts=True)))}", flush=True)
    rng = np.random.default_rng(seed)
    saved = {"bases": bases, "region": region.astype(str), **{f"label_{k}": v for k, v in labels.items()}}
    rows, report = {}, []
    scales = np.array([(t * MM, r * DEG) for t, r in suite["scales"]])

    for mode in args_cli.modes:
        plan = step_schedule(bases, scales, suite["repeats"], suite["shams"], suite["settle_s"], suite["approach_s"],
                             goal_mode=mode == "goal", seed=seed)
        print(f"[INFO] {mode}: step schedule, {len(plan.segments)} segments", flush=True)
        result = run(system, plan.segments, trace=True)
        per_base, resp, dyn = step_metrics(plan, result)
        saved.update({f"{mode}_settled": result.settled, f"{mode}_drift": result.drift, f"{mode}_reset": result.reset,
                      f"{mode}_axis": plan.axis, f"{mode}_scale": plan.scale, f"{mode}_magnitude": plan.magnitude,
                      f"{mode}_scales": plan.scales, f"{mode}_settle_time": dyn["settle_time"],
                      f"{mode}_overshoot": dyn["overshoot"],
                      **{f"{mode}_{k}": v for k, v in per_base.items()}})
        for s, (t, r) in enumerate(suite["scales"]):
            for key, label in (("trans_gain", f"translation gain at {t:g} mm"), ("rot_gain", f"rotation gain at {r:g} deg"),
                               ("trans_cond", f"translation condition number at {t:g} mm"),
                               ("trans_coupling", f"translation coupling at {t:g} mm"),
                               ("trans_gap_mm", f"hysteresis, translation (mm) at {t:g} mm"),
                               ("rot_gap_deg", f"hysteresis, rotation (deg) at {r:g} deg")):
                rows[f"{mode}: {label}"] = {g: summary.bootstrap_median(per_base[key][..., s][region == g], rng)
                                            for g in np.unique(region)}
            for kind, sel, unit, label in ((0, plan.axis < 3, MM, f"{t:g} mm"), (1, plan.axis >= 3, DEG, f"{r:g} deg")):
                keep = sel & (plan.scale == s) & resp["flat"]
                rows[f"{mode}: settling time (s), {label} step"] = {
                    g: summary.bootstrap_median(np.where(keep & (region == g)[None], dyn["settle_time"], np.nan).ravel(), rng)
                    for g in np.unique(region)}
        for g in np.unique(region):
            t_res, r_res = summary.pooled_resolution(plan, resp, region == g)
            rows.setdefault(f"{mode}: resolution, translation (mm)", {})[g] = (t_res,) * 3
            rows.setdefault(f"{mode}: resolution, rotation (deg)", {})[g] = (r_res,) * 3
            rows.setdefault(f"{mode}: bias validity radius at 0.5 mm (mm)", {})[g] = (
                summary.pooled_validity(plan, result, resp, region == g),) * 3
        rows[f"{mode}: settled trials share"] = {g: (float(resp["flat"][:, region == g].mean()),) * 3
                                                 for g in np.unique(region)}
        report.append(f"{mode}: {int(result.reset.sum())} segment-instances with a reset")

    for kind in ("varied", "matched"):
        K = suite["repeat_k"]
        shift = [k + 1 for k in range(K)] if kind == "varied" else [1] * K
        starts = np.stack([np.roll(bases, s, axis=0) for s in shift])
        rplan = repeat_schedule(bases, starts, suite["settle_s"], suite["approach_s"])
        rep = est.repeatability(rplan, run(system, rplan.segments))
        saved.update({f"repeat_{kind}_{k}": v for k, v in rep.items()})
        rows[f"repeatability, {kind} start: RMS radius (mm)"] = {
            g: summary.bootstrap_median(rep["radius_rms"][region == g] / MM, rng) for g in np.unique(region)}
        rows[f"repeatability, {kind} start: RMS rotation (deg)"] = {
            g: summary.bootstrap_median(rep["rot_rms"][region == g] / DEG, rng) for g in np.unique(region)}

    loop = OuterLoop(gain=1.0, look_then_move=True, settle_s=suite["settle_s"], noise_pos=0.25 * MM,
                     noise_rot=0.25 * DEG, corrections=suite["corrections"])
    cl = run_closed_loop(system, bases, loop, suite["approach_s"], seed)
    saved.update({"closed_loop_error": cl["error"], "closed_loop_command": cl["command"]})
    pos = np.linalg.norm(cl["error"][..., :3], axis=-1) / MM
    for c in range(len(pos)):
        rows[f"closed loop (K 1, 0.25 mm / deg noise): position error (mm) before correction {c}"] = {
            g: summary.bootstrap_median(pos[c][region == g], rng) for g in np.unique(region)}

    np.savez_compressed(os.path.join(out_dir, "results.npz"), **saved)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"checkpoint": checkpoint, "system": args_cli.system, "freeze": args_cli.freeze, "suite": args_cli.suite,
            "suite_cfg": suite, "split": args_cli.split, "seed": seed, "modes": args_cli.modes,
            "overrides": replayed + system_overrides + hydra_args, "dropped_overrides": dropped, "git": commit,
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    regions_ = dict(zip(*np.unique(region, return_counts=True)))
    md = (f"# Steerability: {name} ({args_cli.suite}, {args_cli.split})\n\n{os.path.basename(run_dir)}, "
          f"{os.path.basename(checkpoint)}; bases per region (instance x arm): {regions_}; "
          f"medians [95% bootstrap interval over bases].\n\n" + summary.table(rows) + "\n" + "\n".join(report) + "\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    env.close()  # as evals/isaac/arms/get_scorecard.py: without it the app hung at close until the timeout killed it


if __name__ == "__main__":
    main()
    simulation_app.close()
