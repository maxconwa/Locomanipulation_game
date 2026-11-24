"""Do the frozen stage-1 decoders stand under main's randomisation? LocoManip-Marl-Pulse-v0 with a zero policy output.

  python -m pretraining.scripts.pulse_baseline [--num-envs 4096] [--steps 1000] [--no-prior] [--nominal] [--out <json>]

Every agent's latent residual u is 0, so z = tanh(prior(proprio)) (with --no-prior, z = 0): the prior's guess of what
the stage-1 policy would do, decoded, under the task's full randomisation, commands and resets (training mode, so the
arm goals start at step 1 as in training). Nothing learns. Printed, over the run: the mean length of the episodes that
ended (s) and how they ended, falls per minute of simulated robot time, the share of env-steps on which a GOLEM e-stop
trip started per agent and the share of env-steps in trip per cause, and the wrists' PD torque (GOLEM's estimate,
nominal gains) as a share of their e-stop torque and their speed. Also the env's own Episode_Reward/* means at resets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

TASK = "LocoManip-Marl-Pulse-v0"


def run(args) -> dict:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from locomanipulation_game.tasks.locomanip_marl.golem_safety import GOLEM_JOINT_NAMES

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.no_prior:
        cfg.pulse_prior = None
    if args.nominal:  # main's pre-pose resets, gain draws, action delay and encoder offsets off
        cfg.events.prepose = None
        cfg.events.actuator_scales = None
        cfg.action_delay_range = None
        cfg.encoder_offset_rad = None
    cfg.sync_derived()
    env = gym.make(TASK, cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    robot = env.scene["robot"]
    monitor = env.golem_monitor
    wrists = [i for i, name in enumerate(GOLEM_JOINT_NAMES) if "wrist" in name]
    wrist_ids = [monitor.joint_ids[i] for i in wrists]
    env.reset()
    zero = {a: torch.zeros(n, env.cfg.action_spaces[a], device=dev) for a in env.cfg.possible_agents}
    ended_len, ended_fell, ended_timeout = [], 0, 0
    trip_starts = {a: 0.0 for a in env.cfg.possible_agents}
    in_trip = {a: 0.0 for a in env.cfg.possible_agents}
    safety: dict[str, float] = {}
    wrist_tau, wrist_tau_max, wrist_dq = 0.0, 0.0, 0.0
    rewards: dict[str, list[float]] = {}
    with torch.no_grad():
        for step in range(args.steps):
            length = env.episode_length_buf.clone()
            _, _, terminated, truncated, _ = env.step(zero)
            term, trunc = terminated["legs"], truncated["legs"]
            done = term | trunc
            if done.any():
                ended_len += (length[done].float() + 1.0).mul(env.step_dt).tolist()
                ended_fell += int((env.termination_manager.get_term("fell") & done).sum())  # computed before the resets
                ended_timeout += int(trunc.sum())
            for a in trip_starts:
                trip_starts[a] += float(env.golem_log[f"Safety/trip_starts_{a}"])
                in_trip[a] += float(env.golem_tripped_by[a].float().mean())
            for k, v in env.golem_log.items():
                safety[k] = safety.get(k, 0.0) + float(v)
            over = monitor._motor_torque().abs() > monitor.tau_max
            joint_over = over.float().mean(0) if step == 0 else joint_over + over.float().mean(0)
            tau = monitor._motor_torque()[:, wrists].abs() / monitor.tau_max[wrists]
            wrist_tau += float(tau.mean())
            wrist_tau_max = max(wrist_tau_max, float(tau.max()))
            wrist_dq += float(robot.data.joint_vel[:, wrist_ids].abs().mean())
            for k, v in env.extras.get("log", {}).items():
                if k.startswith("Episode_Reward/") or k.startswith("Episode_Termination/"):
                    rewards.setdefault(k, []).append(float(v))
            if step % 100 == 0:
                print(f"[pulse_baseline] step {step}: ended {len(ended_len)} ({ended_fell} fell)", flush=True)
    robot_minutes = n * args.steps * env.step_dt / 60.0
    out = {
        "num_envs": n, "steps": args.steps, "prior": not args.no_prior, "nominal": args.nominal, "seconds_per_env": args.steps * env.step_dt,
        "episodes_ended": len(ended_len), "fell": ended_fell, "timed_out": ended_timeout,
        "mean_ended_episode_s": sum(ended_len) / len(ended_len) if ended_len else None,
        "falls_per_robot_minute": ended_fell / robot_minutes,
        "trip_start_share_per_step": {a: v / args.steps for a, v in trip_starts.items()},
        "in_trip_share_per_step": {a: v / args.steps for a, v in in_trip.items()},
        "safety_log_mean": {k: v / args.steps for k, v in sorted(safety.items())},
        "wrist_torque_over_estop_mean": wrist_tau / args.steps, "wrist_torque_over_estop_max": wrist_tau_max,
        "wrist_abs_dq_mean": wrist_dq / args.steps,
        "torque_over_estop_share_per_joint": {
            name: round(float(v) / args.steps, 4) for name, v in zip(GOLEM_JOINT_NAMES, joint_over) if v > 0},
        "episode_logs_mean": {k: sum(v) / len(v) for k, v in sorted(rewards.items())},
    }
    print("[pulse_baseline] " + json.dumps(out, indent=1), flush=True)
    return out


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-prior", action="store_true")
    p.add_argument("--nominal", action="store_true", help="without the pre-pose resets, gain draws, delay, offsets")
    p.add_argument("--out", default=None, help="write the result as json here")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    AppLauncher(args)
    try:
        out = run(args)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # as fit_prior: the app's close can hang


if __name__ == "__main__":
    main()
