"""The legs scorecard in Isaac: velocity tracking against Unitree's sport mode (the H1-2 Sport Mode Walking Report's
metric code), the gait and the feet, posture and safety; with an arms policy, also how the arms do meanwhile.

    python -m evals.isaac.legs.get_scorecard --checkpoint <run>/checkpoints/agent_<N>.pt [--arms_checkpoint <ckpt>|same]
    python -m evals.isaac.legs.get_scorecard --run <run name or dir> [--steps all|last|<N>,<N>]   (one process each,
                                                                                            cached steps skipped)
    python -m evals.isaac.legs.get_scorecard --latent_run <run dir or checkpoint>   (a latent policy, both agents)
        [--num_envs 84] [--seed 7] [--force] [--out <base>] [env.<field>=... overrides]

The run's own command-line overrides (<run>/params/hydra_overrides.txt) are applied first, so the env is the one it
trained in (those whose key the current cfg no longer has are dropped, and listed); overrides given here come after.

The arms: without --arms_checkpoint every arm joint is held at 0 rad (the arm action's hold_zero: no IK, no
residual). With one (`same`: this checkpoint's own arms), the env is LocoManip-Marl-Direct-v0 and the arms policy
runs its own goals on its own timer while the legs are scored; the scorecard adds a "Coupled" section (the wrists'
errors during the program). See evals/isaac/env.py for what must match between the two runs. The env is the run's own
when it trained both agents (Marl-Direct, Marl-Torso), so a Torso run is scored with `--arms_checkpoint same`.

--latent_run: a policy in the stage-1 decoders' latent (evals/isaac/latent.py: an skrl run of LocoManip-Marl-Pulse-v0,
or a train_commands run), its legs and its arms together, as `--arms_checkpoint same`: the same program, scoring and
Coupled section.

Setup: no pushes, nominal starts (no pre-pose), terrain row 0, the run's domain randomization with nominal motors
(conditions.NOMINAL_MOTORS; --run_motors keeps the run's), observation noise and action delay; mean actions
(--stochastic samples). The segment program and the scoring: evals/common/legs_scorecard.py. With --wrist_goals the
legs then stand while the arms get that many goals in a row (needs an arms policy).

Writes logs/evals/<run>/isaac/legs/scorecard_<N>[_sampled][_with_<arms run>_<M>].{json,md} (or --out <base>).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import conditions as C  # noqa: E402
from evals.common import dispatch, outputs, runs  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--run", default=None, help="Score this run's checkpoints (--steps), one Isaac process each.")
parser.add_argument("--steps", default="all", help="With --run: all, last, or a comma list of env steps.")
parser.add_argument("--arms_checkpoint", default=None, help="Run this arms policy meanwhile (`same`: this checkpoint's).")
parser.add_argument("--latent_run", default=None, help="A latent policy's run (or skrl checkpoint): both its agents.")
parser.add_argument("--settings_run", default=None, help="With a train_commands --latent_run (required): the Pulse"
                    " skrl run whose overrides, IMU model and curriculum state it runs with.")
parser.add_argument("--task", default=None, help="Default: the run's task (LocoManip-Marl-Direct-v0 with"
                    " --arms_checkpoint, unless the run trained both agents).")
parser.add_argument("--num_envs", type=int, default=C.DEFAULTS[("isaac", "legs")]["num_envs"])
parser.add_argument("--segments_per_env", type=int, default=C.DEFAULTS[("isaac", "legs")]["segments_per_env"])
parser.add_argument("--seed", type=int, default=C.DEFAULTS[("isaac", "legs")]["seed"])
parser.add_argument("--wrist_goals", type=int, default=0,
                    help="Standing arm goals after the program (arms policy only).")
parser.add_argument("--run_motors", action="store_true", help="Keep the run's motor randomization (default: nominal).")
parser.add_argument("--stochastic", action="store_true", help="Sample the actions (default: the mean, as deployed).")
parser.add_argument("--save_raw", action="store_true", help="Also save the policy-rate series (npz).")
parser.add_argument("--force", action="store_true", help="Score again even if the json exists.")
parser.add_argument("--out", default=None, help="Output path without extension (default: logs/evals/...).")


def out_base(checkpoint: str, arms_checkpoint: str | None, stochastic: bool) -> str:
    run_dir = runs.run_dir_of(checkpoint)
    suffix = "_sampled" if stochastic else ""
    if arms_checkpoint:
        other = run_dir if arms_checkpoint == "same" else runs.run_dir_of(arms_checkpoint)
        suffix += f"_with_{runs.short(other)}_{runs.step_of(checkpoint if arms_checkpoint == 'same' else arms_checkpoint)}"
    return str(outputs.base(runs.name(run_dir), "isaac", "legs", "scorecard", runs.step_of(checkpoint), suffix))


if __name__ == "__main__" and "--run" in sys.argv:
    a, _ = parser.parse_known_args()
    sys.exit(dispatch.run_steps("evals.isaac.legs.get_scorecard", a.run, a.steps,
                                lambda c: out_base(c, a.arms_checkpoint, a.stochastic), a.force,
                                dispatch.strip(sys.argv[1:], {"--run", "--steps"})))

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None and args_cli.latent_run is None:
    parser.error("--checkpoint, --run or --latent_run is required")
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, LOWER_JOINT_NAMES, TORSO_LINK_NAME  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    evaluation_cfgs, replay_run, state_checkpoint_for)

from evals.common import legs_scorecard as L  # noqa: E402
from evals.common import scorecard  # noqa: E402
from evals.isaac import env as E  # noqa: E402
from evals.isaac.latent import LatentPolicy  # noqa: E402

latent = LatentPolicy.from_run(args_cli.latent_run, args_cli.settings_run) if args_cli.latent_run else None
coupled = latent is not None or args_cli.arms_checkpoint is not None
arms_ckpt = None
extra = ([] if args_cli.run_motors else list(C.NOMINAL_MOTORS)) + hydra_args
if latent is not None:
    checkpoint, run_dir, args_cli.task, replayed = latent.replay(extra)
    arms_ckpt = checkpoint
    OUT = args_cli.out or str(outputs.base(runs.name(run_dir), "isaac", "legs", "scorecard", latent.run.step,
                                           "_sampled" if args_cli.stochastic else ""))
else:
    if coupled:
        arms_ckpt = os.path.abspath(args_cli.checkpoint if args_cli.arms_checkpoint == "same"
                                    else args_cli.arms_checkpoint)
    task = args_cli.task or (E.two_agent_task(runs.run_dir_of(args_cli.checkpoint)) if coupled else None)
    checkpoint, run_dir, args_cli.task, replayed = replay_run(args_cli.checkpoint, task, extra)
    OUT = args_cli.out or out_base(checkpoint, args_cli.arms_checkpoint, args_cli.stochastic)
WRIST_GOAL_S = 4.0


def pick(outputs_, agent: str) -> torch.Tensor:
    return outputs_[0][agent] if args_cli.stochastic else outputs_[-1][agent].get("mean_actions", outputs_[0][agent])


def act(runner, base, obs, states) -> dict:
    if latent is not None:
        return latent.act(obs)
    outputs_ = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    actions = {a: pick(outputs_, a) for a in base.possible_agents}
    if not coupled and "arms" in actions:
        actions["arms"] = torch.zeros_like(actions["arms"])  # hold_zero ignores it; zero for the record
    return actions


def reset_policy(runner, done) -> None:
    (latent.reset if latent is not None else runner.agent.reset_rnn)(done)


def wrist_phase(base, runner, arm, obs, goals: int) -> dict:
    """Every env standing, `goals` arm goals in a row: reached share, closest approach and final errors (m, rad)
    over the goals without a fall, and the number of goals with one."""
    n, dt = base.num_envs, base.step_dt
    ids = torch.arange(n, device=base.device)
    arm.cfg.arm_goal_prob, arm.goals_enabled, arm.goals_enabled_at = 1.0, True, None
    steps = int(round(WRIST_GOAL_S / dt))
    reached, best, final, final_rot, fell = (np.zeros((n, goals)) for _ in range(5))
    states = base.state()
    for g in range(goals):
        arm._resample(ids)
        for _ in range(steps):
            obs, _, terminated, truncated, _ = base.step(act(runner, base, obs, states))
            states = base.state()
            agent = base.possible_agents[0]
            reset_policy(runner, terminated[agent] | truncated[agent])
            fell[:, g] = np.maximum(fell[:, g], terminated[agent].cpu().numpy())
        pos_error, rot_error = arm.errors()
        reached[:, g] = arm.goal_reached.cpu().numpy()
        best[:, g] = arm.goal_best_error.cpu().numpy()
        final[:, g], final_rot[:, g] = pos_error.mean(1).cpu().numpy(), rot_error.mean(1).cpu().numpy()
    ok = fell == 0
    return {"goals": int(ok.sum()), "fell": int((~ok).sum()), "reached": float(reached[ok].mean()),
            "best_error": float(best[ok].mean()), "final_error": float(final[ok].mean()),
            "final_rot_error": float(final_rot[ok].mean())}


def coupled_rows(pos_err: np.ndarray, rot_err: np.ndarray, arm_mode: np.ndarray, fell_at: np.ndarray) -> list[dict]:
    """The wrists while the legs ran the program: errors over the steps an arm goal was active, before a fall."""
    valid = np.arange(pos_err.shape[1])[None] < fell_at[:, None]
    m = arm_mode & valid
    p, r = pos_err[m].mean(-1) if m.any() else np.array([]), rot_err[m].mean(-1) if m.any() else np.array([])
    pct = (lambda x, q, s: float(np.percentile(x, q) * s) if len(x) else float("nan"))
    sec = "Coupled: the arms policy's own goals while the legs walk"
    return [dict(section=sec, key="coupled.goal_share", label="Steps with an arm goal active, share",
                 value=float(m.sum() / max(valid.sum(), 1))),
            dict(section=sec, key="coupled.pos_err_median_mm", label="Wrist position error, median (mm)",
                 value=pct(p, 50, 1000), digits=1),
            dict(section=sec, key="coupled.pos_err_p90_mm", label="Wrist position error, p90 (mm)",
                 value=pct(p, 90, 1000), digits=1),
            dict(section=sec, key="coupled.rot_err_median_deg", label="Wrist rotation error, median (deg)",
                 value=pct(r, 50, 180 / np.pi), digits=1)]


if coupled and latent is None:
    E.check_runs_compatible(run_dir, runs.run_dir_of(arms_ckpt))


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    evaluation_cfgs(env_cfg, agent_cfg, *(latent.cfg_run if latent is not None else (checkpoint, run_dir)))
    if coupled and latent is None:
        env_cfg.state_checkpoint_path = E.merged_state(
            env_cfg.state_checkpoint_path, state_checkpoint_for(arms_ckpt), "arms", os.path.dirname(OUT))
    elif not coupled:
        env_cfg.actions.arm_pos.hold_zero = True
    if getattr(env_cfg.events, "prepose", None) is not None:
        env_cfg.events.prepose.params["prob"] = 0.0  # nominal starts; the engage: evals/deploy
    env_cfg.episode_length_s = 1.0e4
    env_cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    if not (coupled and arm_cfg.independent_of_legs):
        arm_cfg.arm_goal_prob = 0.0
        arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 0.0

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    if latent is not None:
        runner = None
        latent.bind(env, agent_cfg, args_cli.stochastic)
    else:
        runner = MarlRunner(env, agent_cfg)
        E.check_agent_fits(base, "legs", run_dir, checkpoint)
        E.load_agent(runner, checkpoint, "legs")
        if coupled:
            E.check_agent_fits(base, "arms", runs.run_dir_of(arms_ckpt), arms_ckpt)
            E.load_agent(runner, arms_ckpt, "arms")
        runner.agent.enable_training_mode(False, apply_to_models=True)
    robot, vel = base.scene["robot"], base.command_manager.get_term("base_velocity")
    arm = base.command_manager.get_term("arm_targets")
    arm._judge_kernels = lambda ended: None  # the kernel stage the run reached, not moved by this evaluation
    contacts = base.scene["contact_forces"]
    legs = robot.find_joints(LOWER_JOINT_NAMES, preserve_order=True)[0]
    feet = robot.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
    feet_c = contacts.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
    torso = robot.find_bodies(TORSO_LINK_NAME)[0][0]
    N, dt = base.num_envs, base.step_dt
    rng = np.random.default_rng(args_cli.seed)
    ranges = vel_cfg.ranges
    full = tuple(float(max(abs(r[0]), abs(r[1]))) for r in (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z))
    cmds, kinds = L.program(N, args_cli.segments_per_env, dt, rng, full)
    T = cmds.shape[1]
    cmd_t = torch.tensor(cmds, dtype=torch.float, device=base.device)
    rec = {k: np.zeros((N, T) + s, np.float32) for k, s in
           (("vel", (3,)), ("q", (12,)), ("rp", (2,)), ("seen", (3,)), ("foot_z", (2,)))}
    rec["contact"] = np.zeros((N, T, 2), bool)
    arm_rec = {"pos": np.zeros((N, T, 2), np.float32), "rot": np.zeros((N, T, 2), np.float32),
               "mode": np.zeros((N, T), bool)} if coupled else None
    fell_at = np.full(N, T)
    estop = np.zeros((N, T), bool)
    print(f"[INFO] legs scorecard: {N} envs x {T} steps ({T * dt:.0f} s) at {1 / dt:.0f} Hz;"
          f" arms {'policy ' + arms_ckpt if coupled else 'held at 0 rad'}")
    with torch.inference_mode():
        obs, _ = base.reset()
        states = base.state()
        for k in range(T):
            vel.vel_command_b[:] = cmd_t[:, k]
            vel.time_left[:] = 1.0e6
            vel.is_standing_env[:] = False
            obs, _, terminated, truncated, _ = base.step(act(runner, base, obs, states))
            states = base.state()
            agent = base.possible_agents[0]
            reset_policy(runner, terminated[agent] | truncated[agent])
            done = terminated[agent].cpu().numpy()
            fell_at = np.where(done & (fell_at == T), k, fell_at)
            estop[:, k] = base.golem_tripped.cpu().numpy()
            q = robot.data.root_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            vw = robot.data.root_lin_vel_w
            c, s = torch.cos(yaw), torch.sin(yaw)
            rec["vel"][:, k] = torch.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1],
                                            robot.data.root_ang_vel_w[:, 2]], 1).cpu().numpy()
            rec["q"][:, k] = robot.data.joint_pos[:, legs].cpu().numpy()
            roll, pitch, _ = euler_xyz_from_quat(robot.data.body_quat_w[:, torso])
            rec["rp"][:, k] = torch.stack([roll, pitch], 1).cpu().numpy()
            rec["seen"][:, k] = vel.command.cpu().numpy()
            rec["foot_z"][:, k] = robot.data.body_pos_w[:, feet, 2].cpu().numpy()
            rec["contact"][:, k] = (contacts.data.net_forces_w_history[:, :, feet_c, 2].max(dim=1).values > 5.0).cpu().numpy()
            if coupled:
                pe, re_ = arm.errors()
                arm_rec["pos"][:, k], arm_rec["rot"][:, k] = pe.cpu().numpy(), re_.cpu().numpy()
                arm_rec["mode"][:, k] = arm.arm_mode.cpu().numpy()
            if k % 500 == 0:
                print(f"[INFO] step {k}/{T}: {int((fell_at < T).sum())} fallen")
        wrist = wrist_phase(base, runner, arm, obs, args_cli.wrist_goals) if (coupled and args_cli.wrist_goals) else None

    res = L.compute(cmds, kinds, {k: rec[k] for k in ("vel", "q", "rp", "foot_z", "contact")}, fell_at, dt,
                    base.cfg.gait_period, estop)
    if coupled:
        res["coupled_rows"] = coupled_rows(arm_rec["pos"], arm_rec["rot"], arm_rec["mode"], fell_at)
    res.update(
        backend="isaac", body="legs", run=runs.name(run_dir),
        step=latent.run.step if latent is not None else runs.step_of(checkpoint), checkpoint=checkpoint,
        task=args_cli.task, trainer=latent.trainer if latent is not None else runs.trainer(run_dir),
        overrides=replayed + extra, stochastic=args_cli.stochastic,
        full=list(full), wrist=wrist,
        other_policy=(f"arms policy {arms_ckpt}" if coupled else None),
        # absent for a joint-space run: its json as before
        **({"latent": latent.kind, "settings_run": latent.settings_run} if latent is not None else {}),
        **C.stamp("isaac", "legs", num_envs=N, segments_per_env=args_cli.segments_per_env, seed=args_cli.seed,
                  wrist_goals=args_cli.wrist_goals, motors="run" if args_cli.run_motors else "nominal",
                  actions="sampled" if args_cli.stochastic else "mean",
                  arms=("policy" if coupled else "zero"), extra_overrides=hydra_args))
    rows = L.rows(res)
    scorecard.write(OUT, res, rows, L.header(res), L.footer(res))
    if args_cli.save_raw:
        np.savez_compressed(OUT + "_raw.npz", cmds=cmds, kinds=kinds, fell_at=fell_at, dt=dt, **rec)
    print(open(OUT + ".md").read())
    print(f"[INFO] wrote {OUT}.json / .md")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
