"""Score a checkpoint's arms on standing arm goals: position error, vibration, wobble and variance.

    python scripts/skrl/eval_arms.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless [--task LocoManip-ArmsOnly-v0]
        [--num_envs 64] [--goals 8] [env.<field>=... overrides]

The run's own command-line overrides (<run>/params/hydra_overrides.txt) are applied first, as eval_scorecard.py does.
Setup: every env stands (zero velocity) and gets --goals arm goals in a row, GOAL_S each (training's), at depth level 0
from its own goal tables; deterministic actions (the mean, as deployed) unless --stochastic (or --ik_only: the arms'
actions zero, the IK and its trajectory alone, the reference the policy's residual is judged against); the run's domain
randomization, observation noise and action delay. Recorded every policy step (100 Hz): each wrist's position and
rotation error against its world target, its world position, and the arm joints' velocities.

Per goal and wrist, over the hold: the goal's last HOLD_S, after the reach.
  error      the position error at the goal's end (mean of its last END_S) and its mean over the hold; the rotation
             error at the end (the angle of the rotation from the wrist's orientation to the goal's, reported in
             degrees; the JSON keeps radians); the time until the wrist is first within REACH_POS m; reached (both
             wrists held inside the env's reach tolerances, 5 cm and 0.35 rad (20 deg), for 1 s; the arm command's
             own flag)
  vibration  the fast shake: the wrist's position less its moving average over SMOOTH_S (motion above ~1 / SMOOTH_S Hz),
             RMS over the hold, mm; and the arm joints' speed, RMS over the hold, rad/s
  wobble     the slow sway: the moving-averaged position with the hold's straight-line drift removed, RMS over the hold
             (mm) and its largest peak-to-peak along one axis (mm)
  variance   within a hold, the position error's standard deviation over time (mm); across holds, the standard
             deviation and 90th percentile of the end error over every wrist and goal (mm); and the arms policy's
             action std (its log-std parameter: the exploration noise the training samples with)

Goals during which the env reset (a fall; impossible pinned) are left out and counted.

Writes arms_eval_<N>.json and arms_eval_<N>.md (with --offtable, arms_eval_offtable_<N>.*; with --ik_only,
arms_eval[_offtable]_ik_only.*, no step) beside the checkpoint (or --out <dir>).
"""

import argparse
import os
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default=None, help="Default: the task the run was trained on (<run>/params/task.txt;"
                                                 " LocoManip-ArmsOnly-v0 for a run that did not record it).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--goals", type=int, default=8, help="Arm goals per env.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--walk", action="store_true",
                    help="Leave the velocity command as the task draws it (LocoManip-Marl-Direct-v0: the arms scored"
                         " while the legs walk); default: every env stands.")
parser.add_argument("--out", default=None, help="Directory for the files (default: beside the checkpoint).")
parser.add_argument("--offtable", action="store_true",
                    help="Every goal off the table rows (moved U[offtable_scale_range] toward the rest pose); files"
                         " arms_eval_offtable_<N>.*")
parser.add_argument("--stochastic", action="store_true",
                    help="Sample the actions, as in training (default: the mean action, as deployed).")
parser.add_argument("--ik_only", action="store_true",
                    help="Zero arms actions: the IK and its trajectory alone (the checkpoint only sets up the run);"
                         " files arms_eval[_offtable]_ik_only.*")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import evaluation_cfgs, replay_run  # noqa: E402

checkpoint, run_dir, args_cli.task, replayed = replay_run(args_cli.checkpoint, args_cli.task, hydra_args,
                                                          fallback="LocoManip-ArmsOnly-v0")

GOAL_S = 4.0     # s per arm goal, training's
HOLD_S = 2.0     # s: the goal's last part, scored as the hold
END_S = 0.25     # s: the end error's averaging window
SMOOTH_S = 0.1   # s: the moving average that splits vibration (faster) from wobble (slower)
REACH_POS = 0.05  # m: the time-to-reach threshold, the env's reach tolerance
# reached, scored again from the recorded errors at fixed tolerances (m, rad), whatever the env's own: both wrists
# held inside them for REACH_HOLD_S at any time in the goal
REACH_TOLS = {"5cm_20deg": (0.05, 0.35), "1mm_5deg": (0.001, math.radians(5.0))}
REACH_HOLD_S = 1.0


def pick(outputs, agent):
    """The agent's action: the mean, as deployed, unless --stochastic."""
    return outputs[0][agent] if args_cli.stochastic else outputs[-1][agent].get("mean_actions", outputs[0][agent])


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples along axis 0, the ends over what is there."""
    kernel = np.ones(n)
    total = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, x)
    count = np.convolve(np.ones(len(x)), kernel, mode="same").reshape((-1,) + (1,) * (x.ndim - 1))
    return total / count


def hold_metrics(err: np.ndarray, pos: np.ndarray, qd: np.ndarray, dt: float) -> dict:
    """One wrist's hold: err (T,) m, pos (T, 3) m, qd (T, 7) rad/s, at dt. Returns its scores (m, rad/s)."""
    smooth = moving_average(pos, max(1, int(round(SMOOTH_S / dt))))
    shake = pos - smooth
    t = np.arange(len(pos), dtype=float)
    slope, offset = np.polyfit(t, smooth, 1)  # per axis
    drift = np.outer(t, slope) + offset
    sway = smooth - drift
    return {"hold_error": float(err.mean()), "hold_error_std": float(err.std()),
            "vibration_rms": float(np.sqrt((shake ** 2).sum(1).mean())), "joint_speed_rms": float(np.sqrt((qd ** 2).mean())),
            "wobble_rms": float(np.sqrt((sway ** 2).sum(1).mean())), "wobble_p2p": float((sway.max(0) - sway.min(0)).max())}


def markdown(res: dict) -> str:
    """The summary table."""
    m = res["mean"]
    mm = lambda v: f"{1000 * v:.1f}"  # noqa: E731
    kind = "zero arms (IK alone)" if res.get("ik_only") else "sampled" if res["stochastic"] else "mean"
    tol = res.get("reach_tol", [0.05, 0.35])
    lines = [f"# Arms: {res['run']} @ {res['step']}", "",
             f"{res['num_envs']} envs x {res['goals']} standing arm goals of {GOAL_S:g} s ({res['holds']} wrist holds);"
             f" hold = the last {HOLD_S:g} s; {kind} actions;"
             f" {res['goals_with_reset']} goals with a reset (a fall) left out.", "",
             "| | Value |", "| --- | --- |",
             f"| Reached (both wrists within {mm(tol[0])} mm / {math.degrees(tol[1]):.0f} deg for 1 s, the env's flag)"
             f" | {100 * res['reached']:.1f}% |",
             *[f"| Reached at {name.replace('_', ' / ').replace('cm', ' cm').replace('mm', ' mm').replace('deg', ' deg')}"
               f" (from the errors) | {100 * v:.1f}% |" for name, v in res.get("reached_at", {}).items()],
             f"| Position error at the goal's end, mean (mm) | {mm(m['end_error'])} |",
             f"| Position error over the hold, mean (mm) | {mm(m['hold_error'])} |",
             f"| Rotation error at the goal's end, mean (deg) | {math.degrees(m['end_rot_error']):.1f} |",
             f"| Time to within {100 * REACH_POS:g} cm, median of those that got there (s) | {res['time_to_reach_median']:.2f}"
             f" ({100 * res['reach_share']:.0f}% got there) |",
             f"| Vibration: wrist shake above ~{1 / SMOOTH_S:g} Hz, RMS (mm) | {mm(m['vibration_rms'])} |",
             f"| Vibration: arm joint speed in the hold, RMS (rad/s) | {m['joint_speed_rms']:.3f} |",
             f"| Wobble: slow sway with the drift removed, RMS (mm) | {mm(m['wobble_rms'])} |",
             f"| Wobble: largest peak-to-peak along one axis (mm) | {mm(m['wobble_p2p'])} |",
             f"| Variance: error std over time within a hold (mm) | {mm(m['hold_error_std'])} |",
             f"| Variance: end error across holds, std / p90 (mm) | {mm(res['end_error_std'])} / {mm(res['end_error_p90'])} |",
             f"| Variance: arms policy action std (exploration) | {res['policy_std_mean']:.3f} |",
             "", "Per wrist (left / right): end error "
             f"{mm(res['per_arm']['end_error'][0])} / {mm(res['per_arm']['end_error'][1])} mm, vibration "
             f"{mm(res['per_arm']['vibration_rms'][0])} / {mm(res['per_arm']['vibration_rms'][1])} mm, wobble "
             f"{mm(res['per_arm']['wobble_rms'][0])} / {mm(res['per_arm']['wobble_rms'][1])} mm."]
    return "\n".join(lines) + "\n"


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    evaluation_cfgs(env_cfg, agent_cfg, checkpoint, run_dir)
    # nominal starts, comparable with runs before the pre-pose resets (the engage is checked in hw_stress_mujoco.py)
    if getattr(env_cfg.events, "prepose", None) is not None:
        env_cfg.events.prepose.params["prob"] = 0.0
    env_cfg.episode_length_s = 1.0e4
    env_cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 1.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    if not args_cli.walk:
        vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 1.0

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    runner = MarlRunner(env, agent_cfg)
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    robot = base.scene["robot"]
    arm = base.command_manager.get_term("arm_targets")
    arm._judge_kernels = lambda ended: None  # the kernel stage the run reached, not moved by this evaluation
    arm.goals_enabled, arm.goals_enabled_at = True, None
    # the table's rows as drawn, or with --offtable every goal off them
    arm.cfg.offtable_prob = 1.0 if args_cli.offtable else 0.0
    arm.cfg.rest_goal_prob = 0.0  # no rest-pose goals (ArmTargetsCommandCfg.rest_goal_prob): reaching only
    arm_joints = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)[0]
    half = len(arm_joints) // 2  # left arm's joints first, as ARM_JOINT_NAMES lists them
    N, dt = base.num_envs, base.step_dt
    steps = int(round(GOAL_S / dt))
    hold, end = int(round(HOLD_S / dt)), int(round(END_S / dt))
    ids = torch.arange(N, device=base.device)
    err = np.zeros((N, args_cli.goals, steps, 2), np.float32)
    rot = np.zeros((N, args_cli.goals, 2), np.float32)
    rot_t = np.zeros((N, args_cli.goals, steps, 2), np.float32)
    pos = np.zeros((N, args_cli.goals, steps, 2, 3), np.float32)
    qd = np.zeros((N, args_cli.goals, steps, len(arm_joints)), np.float32)
    reached = np.zeros((N, args_cli.goals), bool)
    fell = np.zeros((N, args_cli.goals), bool)  # a reset (a fall) during the goal: the hold is left out
    with torch.inference_mode():
        obs, _ = base.reset()
        states = base.state()
        for g in range(args_cli.goals):
            arm._resample(ids)  # a fresh standing goal for every env
            for k in range(steps):
                outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
                actions = {a: pick(outputs, a) for a in base.possible_agents}
                if args_cli.ik_only:
                    actions["arms"] = torch.zeros_like(actions["arms"])
                obs, _, terminated, truncated, _ = base.step(actions)
                runner.agent.reset_rnn(terminated["arms"] | truncated["arms"])  # a reset env starts from a zero state
                fell[:, g] |= (terminated["arms"] | truncated["arms"]).cpu().numpy()
                states = base.state()
                pos_error, rot_error = arm.errors()
                err[:, g, k] = pos_error.cpu().numpy()
                rot_t[:, g, k] = rot_error.cpu().numpy()
                pos[:, g, k] = robot.data.body_pos_w[:, arm.body_ids].cpu().numpy()
                qd[:, g, k] = robot.data.joint_vel[:, arm_joints].cpu().numpy()
            rot[:, g] = rot_error.cpu().numpy()
            reached[:, g] = arm.goal_reached.cpu().numpy()
            print(f"[INFO] goal {g + 1}/{args_cli.goals}: reached {reached[:, g].mean():.0%}, end error"
                  f" {err[:, g, -end:].mean():.3f} m, {int(fell[:, g].sum())} envs reset", flush=True)

    per = {k: np.zeros((N, args_cli.goals, 2)) for k in
           ("end_error", "hold_error", "hold_error_std", "vibration_rms", "joint_speed_rms", "wobble_rms", "wobble_p2p")}
    reach_t = np.full((N, args_cli.goals, 2), np.nan)
    for e in range(N):
        for g in range(args_cli.goals):
            for a in range(2):
                joints = slice(0, half) if a == 0 else slice(half, None)
                h = hold_metrics(err[e, g, -hold:, a], pos[e, g, -hold:, a], qd[e, g, -hold:, joints], dt)
                for key, value in h.items():
                    per[key][e, g, a] = value
                per["end_error"][e, g, a] = err[e, g, -end:, a].mean()
                within = np.nonzero(err[e, g, :, a] < REACH_POS)[0]
                if len(within):
                    reach_t[e, g, a] = (within[0] + 1) * dt
    held = int(round(REACH_HOLD_S / dt))
    reached_at = {}
    for name, (pos_tol, rot_tol) in REACH_TOLS.items():
        inside = ((err < pos_tol) & (rot_t < rot_tol)).all(axis=-1)  # (N, goals, steps)
        run = np.apply_along_axis(lambda v: np.convolve(v, np.ones(held), mode="valid").max(), 2, inside.astype(float))
        reached_at[name] = run >= held
    keep = ~fell
    if not keep.any():
        raise SystemExit("every goal had a reset: nothing to score")
    per = {k: v[keep] for k, v in per.items()}  # (holds kept, 2)
    reach_t, reached, rot = reach_t[keep], reached[keep], rot[keep]
    log_std = runner.agent.policies["arms"].log_std_parameter.detach()
    step = re.search(r"_(\d+)\.pt$", checkpoint)
    res = {"run": os.path.basename(run_dir), "step": int(step.group(1)) if step else None, "checkpoint": checkpoint,
           "task": args_cli.task, "overrides": replayed + hydra_args, "stochastic": args_cli.stochastic,
           "ik_only": args_cli.ik_only,
           "num_envs": N, "goals": args_cli.goals, "holds": int(keep.sum() * 2), "goals_with_reset": int(fell.sum()),
           "policy_dt": dt,
           "offtable": args_cli.offtable,
           "reached": float(reached.mean()),
           "reach_tol": [float(arm.cfg.reach_pos_tol), float(arm.cfg.reach_rot_tol)],
           "reached_at": {name: float(v[keep].mean()) for name, v in reached_at.items()},
           "mean": {**{k: float(v.mean()) for k, v in per.items()}, "end_rot_error": float(rot.mean())},
           "per_arm": {k: [float(v[..., 0].mean()), float(v[..., 1].mean())] for k, v in per.items()},
           "end_error_std": float(per["end_error"].std()), "end_error_p90": float(np.percentile(per["end_error"], 90)),
           "reach_share": float(np.isfinite(reach_t).mean()),
           "time_to_reach_median": float(np.nanmedian(reach_t)) if np.isfinite(reach_t).any() else float("nan"),
           "policy_std_mean": float(log_std.exp().mean()), "policy_std": log_std.exp().cpu().tolist()}
    out_dir = args_cli.out or os.path.dirname(checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    tag = "arms_eval_offtable" if args_cli.offtable else "arms_eval"
    if args_cli.ik_only:
        tag += "_ik_only"
    elif res["step"] is not None:
        tag = f"{tag}_{res['step']}"
    tag += "_sampled" if args_cli.stochastic else ""
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, f"{tag}.md"), "w") as f:
        f.write(markdown(res))
    print(markdown(res))
    print(f"[INFO] wrote {os.path.join(out_dir, tag)}.json / .md")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
