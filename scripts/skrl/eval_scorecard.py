"""Score a checkpoint's velocity tracking against Unitree's sport mode, with the H1-2 Sport Mode Walking Report's
metric code (scorecard_metrics.py), and its wrist tracking on standing arm goals.

    python scripts/skrl/eval_scorecard.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless [--num_envs 168]
        [env.<field>=... overrides]

The run's own command-line overrides (<run>/params/hydra_overrides.txt, which train.py writes) are applied first, so
the env is the one it trained in (those whose key the current cfg no longer has are dropped, and listed); overrides
given here come after them.

Setup: arm goals off (navigation only, the wrists at their rest pose), no pushes, terrain row 0 (bumps of +-height_at_0,
flat by default), the run's domain randomization, observation noise and action delay; deterministic actions.
Every env stands 2 s, then runs its own shuffled sequence of --segments_per_env scripted segments, each followed by 2 s of zero command:
  step   a command held 4 s: each axis alone (vx +-0.2/0.4/0.6, vy +-0.15/0.25, wz +-0.2/0.35/0.5) and 8 combinations,
         at the baseline's ranges (vx +-0.7, vy +-0.3, wz +-0.5), each axis scaled to the run's own (its command
         term's ranges: a run trained on sport mode's 0.55 / 0.54 / 0.97 gets vx 0.16-0.47, vy 0.27-0.45, wz 0.39-0.97)
  flick  the sport-mode operator's stick: 150 ms pulses at the axis's full value (the run's range edge), one every
         U(0.3, 0.5) s, for 5 s; per axis and sign
  small  a command of norm U(0.05, 0.15) over (vx, vy, wz) in a random direction, held 4 s
Each segment type gets the same number of slots over all envs. The command goes through the run's command term (its
low-pass, if it trained with one), as a deploy would send it. Measured: the true pelvis velocity in the yaw frame and
the world yaw rate, the knee angles, the torso's roll and pitch, falls and GOLEM e-stop trips. An env's data ends at
its first fall. Everything is resampled to 100 Hz (commands held, the rest interpolated) for the metric code.

Then the wrist phase: every env stands (zero velocity) and gets --wrist_goals arm goals in a row, WRIST_GOAL_S each,
at depth level 0 (heights 1.0-1.7 m, which the standing arms reach), from the env's own goal tables. Per goal: reached
(both wrists held within the env's reach tolerances, 5 cm and 0.35 rad, for 1 s), the closest approach and the final
error (mean of both wrists); goals during which the env fell count apart.

Scored (scorecard_metrics.score): per axis the fitted gain, delay and tau; on time spent stepping, the velocity
averaged over one stride (the run's gait_period) against the lag-allowed command: turn sense, cos, speed and turn
ratios, linear and angular error, also per segment kind; stride period median and IQR; torso roll / pitch sd; falls;
and the share of commanded time not stepping (overall and for the small commands). No creep correction.

Writes scorecard_<N>.json and scorecard_<N>.md beside the checkpoint (or --out <dir>).
"""

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=168)
parser.add_argument("--segments_per_env", type=int, default=6)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=None, help="Directory for the scorecard files (default: beside the checkpoint).")
parser.add_argument("--analysis_dir", default=None, help="The sport-mode bag analysis (scorecard_metrics).")
parser.add_argument("--wrist_goals", type=int, default=6, help="Arm goals per env in the wrist phase (0: none).")
parser.add_argument("--save_raw", action="store_true", help="Also save the 100 Hz series (npz).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import LOWER_JOINT_NAMES, TORSO_LINK_NAME  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    known_overrides,
    run_overrides,
    state_checkpoint_for,
)

checkpoint = os.path.abspath(args_cli.checkpoint)
run_dir = os.path.dirname(os.path.dirname(checkpoint))
replayed, dropped = known_overrides(run_overrides(run_dir), args_cli.task)
if dropped:
    print(f"[INFO] run overrides the current cfg no longer has (their values are the defaults now): {dropped}")
sys.argv = [sys.argv[0]] + replayed + hydra_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard_metrics as sm  # noqa: E402

SETTLE_S, HOLD_S, FLICK_S, GAP_S = 2.0, 4.0, 5.0, 2.0
WRIST_GOAL_S = 4.0  # s per arm goal in the wrist phase, as in training
PULSE_S, PULSE_EVERY = 0.15, (0.3, 0.5)
BASE_FULL = (0.7, 0.3, 0.5)  # the baseline's range edges, which STEPS are written for
KINDS = ("step", "flick", "small")
STEPS = ([(s * v, 0.0, 0.0) for v in (0.2, 0.4, 0.6) for s in (1, -1)]
         + [(0.0, s * v, 0.0) for v in (0.15, 0.25) for s in (1, -1)]
         + [(0.0, 0.0, s * v) for v in (0.2, 0.35, 0.5) for s in (1, -1)]
         + [(0.4, 0.0, 0.3), (0.4, 0.0, -0.3), (0.3, 0.2, 0.0), (0.3, -0.2, 0.0), (-0.3, 0.0, 0.3),
            (-0.3, 0.0, -0.3), (0.3, 0.15, 0.3), (0.3, -0.15, -0.3)])
FLICKS = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
N_SMALL = 12


def program(num_envs: int, per_env: int, dt: float, rng: np.random.Generator, full: tuple[float, float, float]):
    """(commands (N, T, 3), kind index (N, T): -1 settle/padding, else KINDS' index, the segment's gap included).
    full: the run's range edge per axis (a full stick); the steps are scaled by full / BASE_FULL."""
    scale = np.asarray(full) / np.asarray(BASE_FULL)
    types = ([("step", tuple(np.asarray(c) * scale)) for c in STEPS] + [("flick", f) for f in FLICKS]
             + [("small", None)] * N_SMALL)
    slots = [types[i % len(types)] for i in range(num_envs * per_env)]
    rng.shuffle(slots)
    seqs = []
    for e in range(num_envs):
        cmd, kind = [np.zeros((int(round(SETTLE_S / dt)), 3))], [np.full(int(round(SETTLE_S / dt)), -1)]
        for name, spec in slots[e * per_env:(e + 1) * per_env]:
            if name == "step" or name == "small":
                if name == "small":
                    d = rng.normal(size=3)
                    spec = d / np.linalg.norm(d) * rng.uniform(0.05, 0.15)
                n = int(round(HOLD_S / dt))
                seg = np.tile(np.asarray(spec, float), (n, 1))
            else:
                axis, sign = spec
                n = int(round(FLICK_S / dt))
                seg, t = np.zeros((n, 3)), 0.0
                while t < FLICK_S:
                    a, b = int(round(t / dt)), int(round((t + PULSE_S) / dt))
                    seg[a:min(b, n), axis] = sign * full[axis]
                    t += rng.uniform(*PULSE_EVERY)
            gap = int(round(GAP_S / dt))
            cmd += [seg, np.zeros((gap, 3))]
            kind += [np.full(n + gap, KINDS.index(name))]
        seqs.append((np.concatenate(cmd), np.concatenate(kind)))
    T = max(len(c) for c, _ in seqs)
    cmds = np.zeros((num_envs, T, 3))
    kinds = np.full((num_envs, T), -1)
    for e, (c, k) in enumerate(seqs):
        cmds[e, :len(c)], kinds[e, :len(k)] = c, k
    return cmds, kinds


def to_100hz(x: np.ndarray, dt: float, hold: bool, n: int, offset: float) -> np.ndarray:
    """(T, ...) samples at times offset + k dt -> n samples at 0.01 s steps: held (the latest sample) or linear."""
    t_src = offset + np.arange(len(x)) * dt
    t = np.arange(n) * sm.DT
    if hold:
        k = np.clip(np.searchsorted(t_src, t + 1e-9, side="right") - 1, 0, len(x) - 1)
        return x[k]
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t, t_src, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def fmt(v, digits=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def markdown(res: dict) -> str:
    lag = res["tracking_vs_lagged"]
    lines = [f"# Sport-mode scorecard: {res['run']} @ {res['step']}", "",
             f"{res['num_envs']} envs, {res['seconds_valid']:.0f} s of valid data, {res['seconds_stepping']:.0f} s"
             f" stepping; policy {1 / res['policy_dt']:.0f} Hz, stride window {res['stride_samples'] * sm.DT:.2f} s."
             f" Scored against the lag-allowed command (no creep correction). Commands at the run's ranges: full"
             f" stick vx {res['full'][0]:g}, vy {res['full'][1]:g}, wz {res['full'][2]:g}.", "",
             "| Metric | This checkpoint | Sport mode | Target | Met |", "| --- | --- | --- | --- | --- |"]
    for key, label, sport, target, sense in sm.TARGETS:
        v = lag.get(key, float("nan"))
        lines.append(f"| {label} | {fmt(v)} | {sport} | {'≥' if sense == 'ge' else '≤'} {target} |"
                     f" {'yes' if sm.verdict(v, target, sense) else 'no'} |")
    lines += ["", "| Axis | Gain | Delay (s) | Tau (s) | Bias | R² |", "| --- | --- | --- | --- | --- | --- |"]
    for axis, fit in res["response"].items():
        lines.append(f"| {axis} | {fit['gain']:.3f} | {fit['delay']:.2f} | {fit['tau']:.2f} | {fit['bias']:+.3f} |"
                     f" {fit['r2']:.3f} |")
    g, b = res["gait"], res["tilt"]
    lines += ["", f"- Stride period: median {fmt(g['stride_T_median'])} s, IQR {fmt(g['stride_T_iqr'])} s"
                  f" ({g['n_strides']} strides); sport mode 0.700 s (0.01 s)",
              f"- Torso roll / pitch sd while stepping: {fmt(b['roll_sd_deg'], 2)} / {fmt(b['pitch_sd_deg'], 2)} deg;"
              f" sport mode 0.83 / 0.75",
              f"- Falls: {res['falls']} of {res['num_envs']} envs ({res['falls_per_min']:.3f} per env-minute; by the"
              f" segment they fell in: {', '.join(f'{k} {v}' for k, v in res['falls_by_kind'].items())});"
              f" e-stop trips in {100 * res['estop_share']:.2f}% of steps",
              f"- Commanded but not stepping: {100 * res['commanded_not_stepping_share']:.1f}% of"
              f" {res['seconds_commanded']:.0f} s commanded; small commands {100 * res['small_not_stepping_share']:.1f}%",
              "", "| Segment kind | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for kind, t in res.get("subsets", {}).items():
        lines.append(f"| {kind} | {fmt(t.get('turn_sign_agree', np.nan))} | {fmt(t.get('speed_ratio_median', np.nan))} |"
                     f" {fmt(t.get('turn_ratio_median', np.nan))} | {fmt(t.get('cos_v_mean', np.nan))} |"
                     f" {fmt(t.get('ang_err_mean', np.nan))} | {fmt(t.get('lin_err_mean', np.nan))} |")
    w = res.get("wrist")
    if w:
        lines += ["", f"Wrist tracking ({w['goals']} standing arm goals of {WRIST_GOAL_S:g} s at depth level 0, goals"
                      f" during a fall left out: {w['fell']}): reached {100 * w['reached']:.1f}%, closest approach"
                      f" {100 * w['best_error']:.1f} cm, final error {100 * w['final_error']:.1f} cm /"
                      f" {w['final_rot_error']:.2f} rad"]
    lines += ["", f"Run overrides: `{' '.join(res['overrides']) or '(none)'}`", ""]
    return "\n".join(lines)


def wrist_phase(base, runner, arm, obs, goals: int) -> dict:
    """The wrist phase (module docstring): every env standing, `goals` arm goals in a row at depth level 0. Returns the
    share of goals reached, the mean closest approach and final position error (m) and final rotation error (rad),
    over the goals without a fall, and the number of goals with one."""
    n, dt = base.num_envs, base.step_dt
    ids = torch.arange(n, device=base.device)
    arm.cfg.arm_goal_prob, arm.goals_enabled, arm.goals_enabled_at = 1.0, True, None
    arm.level[:] = 0
    steps = int(round(WRIST_GOAL_S / dt))
    reached, best, final, final_rot, fell = (np.zeros((n, goals)) for _ in range(5))
    states = base.state()
    for g in range(goals):
        arm._resample(ids)  # a fresh goal for every env (its timer stays out of reach: resampling_time_range)
        for j in range(steps):
            outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}
            obs, _, terminated, _, _ = base.step(actions)
            states = base.state()
            fell[:, g] = np.maximum(fell[:, g], terminated[base.possible_agents[0]].cpu().numpy())
        pos_error, rot_error = arm.errors()
        reached[:, g] = arm.goal_reached.cpu().numpy()
        best[:, g] = arm.goal_best_error.cpu().numpy()
        final[:, g], final_rot[:, g] = pos_error.mean(1).cpu().numpy(), rot_error.mean(1).cpu().numpy()
        print(f"[INFO] wrist goal {g + 1}/{goals}: reached {reached[:, g].mean():.0%}, closest approach"
              f" {best[:, g].mean():.3f} m")
    ok = fell == 0
    return {"goals": int(ok.sum()), "fell": int((~ok).sum()), "reached": float(reached[ok].mean()),
            "best_error": float(best[ok].mean()), "final_error": float(final[ok].mean()),
            "final_rot_error": float(final_rot[ok].mean())}


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    analysis = sm.load_analysis(args_cli.analysis_dir or sm.DEFAULT_ANALYSIS_DIR)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = run_dir
    env_cfg.training = False
    env_cfg.state_checkpoint_path = state_checkpoint_for(checkpoint)
    env_cfg.episode_length_s = 1.0e4
    env_cfg.episode_length_mean_s = 0.0  # a run trained on drawn episode lengths: one long episode here
    env_cfg.events.push_robot = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 0.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 0.0
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    runner = MarlRunner(env, agent_cfg)
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    robot, vel = base.scene["robot"], base.command_manager.get_term("base_velocity")
    arm = base.command_manager.get_term("arm_targets")
    arm._record_outcomes = lambda ended: None
    arm.update_levels = lambda env_ids, fell: None
    terrain = base.scene.terrain
    terrain.terrain_levels[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    knees = robot.find_joints(LOWER_JOINT_NAMES, preserve_order=True)[0]
    torso = robot.find_bodies(TORSO_LINK_NAME)[0][0]
    N, dt = base.num_envs, base.step_dt
    rng = np.random.default_rng(args_cli.seed)
    ranges = vel_cfg.ranges
    full = tuple(float(max(abs(r[0]), abs(r[1]))) for r in (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z))
    cmds, kinds = program(N, args_cli.segments_per_env, dt, rng, full)
    T = cmds.shape[1]
    cmd_t = torch.tensor(cmds, dtype=torch.float, device=base.device)
    rec = {k: np.zeros((N, T) + s, np.float32) for k, s in
           (("vel", (3,)), ("q", (12,)), ("rp", (2,)), ("seen", (3,)))}
    fell_at = np.full(N, T)
    estop = np.zeros((N, T), bool)
    print(f"[INFO] scorecard: {N} envs x {T} steps ({T * dt:.0f} s) at {1 / dt:.0f} Hz")
    with torch.inference_mode():
        obs, _ = base.reset()
        states = base.state()
        for k in range(T):
            vel.vel_command_b[:] = cmd_t[:, k]
            vel.time_left[:] = 1.0e6
            vel.is_standing_env[:] = False
            outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}
            obs, _, terminated, _, _ = base.step(actions)
            states = base.state()
            done = terminated[base.possible_agents[0]].cpu().numpy()
            fell_at = np.where(done & (fell_at == T), k, fell_at)
            estop[:, k] = base.golem_tripped.cpu().numpy()
            yaw = torch.atan2(2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3]
                                   + robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
                              1 - 2 * (robot.data.root_quat_w[:, 2] ** 2 + robot.data.root_quat_w[:, 3] ** 2))
            vw = robot.data.root_lin_vel_w
            c, s = torch.cos(yaw), torch.sin(yaw)
            rec["vel"][:, k] = torch.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1],
                                            robot.data.root_ang_vel_w[:, 2]], 1).cpu().numpy()
            rec["q"][:, k] = robot.data.joint_pos[:, knees].cpu().numpy()
            roll, pitch, _ = euler_xyz_from_quat(robot.data.body_quat_w[:, torso])
            rec["rp"][:, k] = torch.stack([roll, pitch], 1).cpu().numpy()
            rec["seen"][:, k] = vel.command.cpu().numpy()
            if k % 500 == 0:
                print(f"[INFO] step {k}/{T}: {int((fell_at < T).sum())} fallen")
        wrist = wrist_phase(base, runner, arm, obs, args_cli.wrist_goals) if args_cli.wrist_goals > 0 else None

    # -- 100 Hz series, each cut at its env's first fall
    rp = np.unwrap(rec["rp"].astype(float), axis=1)
    series = []
    for e in range(N):
        end = fell_at[e]                        # steps 0..end-1 are valid (the fall step's state is the reset's)
        if end < 2 * int(round(1.0 / dt)):
            continue
        n = int(end * dt / sm.DT)
        series.append(dict(
            cmd=to_100hz(cmds[e, :end], dt, True, n, 0.0),
            kind=to_100hz(kinds[e, :end], dt, True, n, 0.0),
            vel=to_100hz(rec["vel"][e, :end].astype(float), dt, False, n, dt),
            q=to_100hz(rec["q"][e, :end].astype(float), dt, False, n, dt),
            rp=to_100hz(rp[e, :end], dt, False, n, dt),
            estop=to_100hz(estop[e, :end], dt, True, n, dt)))
    masks = [analysis.gait.stepping({"q": s["q"]}) for s in series]
    cmds100, vels100 = [s["cmd"] for s in series], [s["vel"] for s in series]
    stride = int(round(base.cfg.gait_period / sm.DT))
    subsets = {kind: [s["kind"] == i for s in series] for i, kind in enumerate(KINDS)}
    res = sm.score(analysis, cmds100, vels100, masks, stride, subsets=subsets)
    commanded = [sm.commanded(c) for c in cmds100]
    small = subsets["small"]
    seconds = lambda ms: float(sum(m.sum() for m in ms) * sm.DT)  # noqa: E731
    step = re.search(r"_(\d+)\.pt$", checkpoint)
    res.update(
        run=os.path.basename(run_dir), step=int(step.group(1)) if step else None, checkpoint=checkpoint,
        overrides=replayed + hydra_args, num_envs=N, policy_dt=dt, full=list(full), gait_period=base.cfg.gait_period,
        seconds_valid=seconds([np.ones(len(c), bool) for c in cmds100]), seconds_stepping=seconds(masks),
        seconds_commanded=seconds(commanded),
        commanded_not_stepping_share=seconds([c & ~m for c, m in zip(commanded, masks)]) / max(seconds(commanded), 1e-9),
        small_not_stepping_share=seconds([c & s & ~m for c, s, m in zip(commanded, small, masks)])
        / max(seconds([c & s for c, s in zip(commanded, small)]), 1e-9),
        falls=int((fell_at < T).sum()),
        falls_by_kind={name: int(sum(1 for e in range(N) if fell_at[e] < T and kinds[e, fell_at[e]] == i))
                       for i, name in enumerate(("settle",) + KINDS, start=-1)},
        falls_per_min=float((fell_at < T).sum() / (np.minimum(fell_at, T).sum() * dt / 60)),
        estop_share=float(np.mean([s["estop"].mean() for s in series])) if series else float("nan"),
        gait=sm.stride_stats([s["q"][:, [3, 9]] for s in series], masks),
        tilt=sm.tilt_stats([s["rp"] for s in series], masks),
        wrist=wrist,
    )
    res["targets_met"] = sum(sm.verdict(res["tracking_vs_lagged"].get(k, np.nan), t, d) for k, _, _, t, d in sm.TARGETS)
    out_dir = args_cli.out or os.path.dirname(checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"scorecard_{res['step']}" if res["step"] is not None else "scorecard"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(out_dir, f"{tag}.md"), "w") as f:
        f.write(markdown(res))
    if args_cli.save_raw:
        np.savez_compressed(os.path.join(out_dir, f"{tag}_raw.npz"), cmds=cmds, kinds=kinds, fell_at=fell_at,
                            dt=dt, **{k: v for k, v in rec.items()})
    print(markdown(res))
    print(f"[INFO] wrote {os.path.join(out_dir, tag)}.json / .md")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
