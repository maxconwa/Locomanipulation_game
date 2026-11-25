"""The arms scorecard in Isaac: arm goals held in a row; position and rotation error, reach, vibration, wobble and
variance (evals/common/arms_scorecard.py), against the IK alone on the same goals.

    python -m evals.isaac.arms.get_scorecard --checkpoint <run>/checkpoints/agent_<N>.pt [--legs_checkpoint <ckpt>|same]
    python -m evals.isaac.arms.get_scorecard --run <run> [--steps all|last|<N>,...]   (one process each; the run's IK-only
                                                                                   reference first; cached skipped)
    python -m evals.isaac.arms.get_scorecard --latent_run <run dir or checkpoint>   (a latent policy, both agents)
        [--num_envs 64] [--goals 8] [--offtable] [--stochastic] [--ik_only] [--steerability] [--force]

The run's own overrides are applied first (as the legs scorecard). The legs: without --legs_checkpoint the env is
LocoManip-ArmsOnly-v0, the pelvis pinned and the legs at their default pose. With one (`same`: this checkpoint's own
legs) it is LocoManip-Marl-Direct-v0, the pelvis free and the legs policy balancing; the goals are then scored twice,
standing (zero velocity) and while the legs walk (a velocity command drawn from the run's ranges, new each goal).
evals/isaac/env.py checks that each agent fits the env. A run that trained both agents (Marl-Direct, Marl-Torso) is
scored in its own task with --legs_checkpoint (a Torso run's arms, which drive the torso, need `same`).

--latent_run: a policy in the stage-1 decoders' latent (evals/isaac/latent.py: an skrl run of LocoManip-Marl-Pulse-v0,
or a train_commands run), its arms and its legs together, as `--legs_checkpoint same` (no IK, so no --ik_only).

--ik_only: zero arms actions (the IK and its trajectory alone; the checkpoint only sets the run up): the reference
the policy's residual is judged against, written as scorecard_ik_only and used as the Reference column of the run's
other scorecards. --steerability: afterwards (about 1.5 h more), the steerability benchmark's quick suite on the dev split
(evals/isaac/arms/steerability, its own Isaac process), its summary appended.

Writes logs/evals/<run>/isaac/arms/scorecard_<N>[_offtable][_sampled][_with_<legs run>_<M>].{json,md}.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import conditions as C  # noqa: E402
from evals.common import dispatch, outputs, runs  # noqa: E402

D = C.DEFAULTS[("isaac", "arms")]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--run", default=None)
parser.add_argument("--steps", default="all")
parser.add_argument("--legs_checkpoint", default=None, help="Run this legs policy meanwhile, the pelvis free (`same`).")
parser.add_argument("--latent_run", default=None, help="A latent policy's run (or skrl checkpoint): both its agents.")
parser.add_argument("--settings_run", default=None, help="With a train_commands --latent_run (required): the Pulse"
                    " skrl run whose overrides, IMU model and curriculum state it runs with.")
parser.add_argument("--task", default=None, help="Default: LocoManip-ArmsOnly-v0 (with --legs_checkpoint: Marl-Direct,"
                    " or the run's own task when it trained both agents).")
parser.add_argument("--num_envs", type=int, default=D["num_envs"])
parser.add_argument("--goals", type=int, default=D["goals"], help="Arm goals per env (and per phase).")
parser.add_argument("--seed", type=int, default=D["seed"])
parser.add_argument("--offtable", action="store_true", help="Every goal off the table rows.")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--ik_only", action="store_true", help="Zero arms actions: the IK alone (the reference).")
parser.add_argument("--steerability", action="store_true", help="Also run the steerability benchmark (quick, dev).")
parser.add_argument("--force", action="store_true")
parser.add_argument("--out", default=None)


def out_base(checkpoint: str, a) -> str:
    run_dir = runs.run_dir_of(checkpoint)
    suffix = "_offtable" if a.offtable else ""
    if a.ik_only:
        return str(outputs.base(runs.name(run_dir), "isaac", "arms", "scorecard" + suffix + "_ik_only", None))
    suffix += "_sampled" if a.stochastic else ""
    if a.legs_checkpoint:
        other = run_dir if a.legs_checkpoint == "same" else runs.run_dir_of(a.legs_checkpoint)
        suffix += f"_with_{runs.short(other)}_{runs.step_of(checkpoint if a.legs_checkpoint == 'same' else a.legs_checkpoint)}"
    return str(outputs.base(runs.name(run_dir), "isaac", "arms", "scorecard", runs.step_of(checkpoint), suffix))


if __name__ == "__main__" and "--run" in sys.argv:
    a, _ = parser.parse_known_args()
    passthrough = dispatch.strip(sys.argv[1:], {"--run", "--steps"})
    code = 0
    if not a.ik_only:  # the run's IK-only reference first (same goals, same table/off-table choice)
        ref = argparse.Namespace(**{**vars(a), "ik_only": True, "legs_checkpoint": None})
        ref_args = dispatch.strip(passthrough, {"--legs_checkpoint", "--steerability", "--stochastic"}) + ["--ik_only"]
        code = dispatch.run_steps("evals.isaac.arms.get_scorecard", a.run, "last", lambda c: out_base(c, ref),
                                  a.force, ref_args)
    code |= dispatch.run_steps("evals.isaac.arms.get_scorecard", a.run, a.steps, lambda c: out_base(c, a), a.force,
                               passthrough)
    sys.exit(code)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None and args_cli.latent_run is None:
    parser.error("--checkpoint, --run or --latent_run is required")
if args_cli.latent_run and (args_cli.ik_only or args_cli.steerability):
    parser.error("--latent_run: no IK to score alone (--ik_only), no steerability benchmark for latent policies")
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    evaluation_cfgs, replay_run, state_checkpoint_for)

from evals.common import arms_scorecard as A  # noqa: E402
from evals.common import scorecard  # noqa: E402
from evals.isaac import env as E  # noqa: E402
from evals.isaac.latent import LatentPolicy  # noqa: E402

latent = LatentPolicy.from_run(args_cli.latent_run, args_cli.settings_run) if args_cli.latent_run else None
coupled = latent is not None or args_cli.legs_checkpoint is not None
legs_ckpt = None
if latent is not None:
    checkpoint, run_dir, args_cli.task, replayed = latent.replay(hydra_args)
    legs_ckpt = checkpoint
    OUT = args_cli.out or str(outputs.base(runs.name(run_dir), "isaac", "arms", "scorecard", latent.run.step,
                                           ("_offtable" if args_cli.offtable else "")
                                           + ("_sampled" if args_cli.stochastic else "")))
else:
    if coupled:
        legs_ckpt = os.path.abspath(args_cli.checkpoint if args_cli.legs_checkpoint == "same"
                                    else args_cli.legs_checkpoint)
    task = args_cli.task or (E.two_agent_task(runs.run_dir_of(args_cli.checkpoint)) if coupled
                             else "LocoManip-ArmsOnly-v0")
    checkpoint, run_dir, args_cli.task, replayed = replay_run(args_cli.checkpoint, task, hydra_args)
    OUT = args_cli.out or out_base(checkpoint, args_cli)


def pick(outputs_, agent):
    return outputs_[0][agent] if args_cli.stochastic else outputs_[-1][agent].get("mean_actions", outputs_[0][agent])


def run_goals(base, runner, arm, robot, arm_joints, obs, walk_cmds=None):
    """args_cli.goals goals for every env; walk_cmds (N, goals, 3): the velocity command per goal (None: stand).
    Returns the recordings and the last obs."""
    N, dt, G = base.num_envs, base.step_dt, args_cli.goals
    steps = int(round(A.GOAL_S / dt))
    ids = torch.arange(N, device=base.device)
    err, rot_t = (np.zeros((N, G, steps, 2), np.float32) for _ in range(2))
    pos = np.zeros((N, G, steps, 2, 3), np.float32)
    qd = np.zeros((N, G, steps, len(arm_joints)), np.float32)
    reached, fell = np.zeros((N, G), bool), np.zeros((N, G), bool)
    vel = base.command_manager.get_term("base_velocity")
    states = base.state()
    for g in range(G):
        arm._resample(ids)
        for k in range(steps):
            if walk_cmds is not None:
                vel.vel_command_b[:] = torch.tensor(walk_cmds[:, g], dtype=torch.float, device=base.device)
                vel.time_left[:] = 1.0e6
                vel.is_standing_env[:] = False
            if latent is not None:
                actions = latent.act(obs)
            else:
                outputs_ = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
                actions = {a: pick(outputs_, a) for a in base.possible_agents}
            if args_cli.ik_only:
                actions["arms"] = torch.zeros_like(actions["arms"])
            obs, _, terminated, truncated, _ = base.step(actions)
            (latent.reset if latent is not None else runner.agent.reset_rnn)(terminated["arms"] | truncated["arms"])
            fell[:, g] |= (terminated["arms"] | truncated["arms"]).cpu().numpy()
            states = base.state()
            pe, re_ = arm.errors()
            err[:, g, k], rot_t[:, g, k] = pe.cpu().numpy(), re_.cpu().numpy()
            pos[:, g, k] = robot.data.body_pos_w[:, arm.body_ids].cpu().numpy()
            qd[:, g, k] = robot.data.joint_vel[:, arm_joints].cpu().numpy()
        reached[:, g] = arm.goal_reached.cpu().numpy()
        print(f"[INFO] goal {g + 1}/{G}{' (walking)' if walk_cmds is not None else ''}: reached {reached[:, g].mean():.0%},"
              f" end error {err[:, g, -int(round(A.END_S / dt)):].mean():.3f} m, {int(fell[:, g].sum())} envs reset",
              flush=True)
    return A.compute(err, rot_t, pos, qd, fell, dt, reached), obs


if coupled and latent is None:
    E.check_runs_compatible(run_dir, runs.run_dir_of(legs_ckpt))


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    evaluation_cfgs(env_cfg, agent_cfg, *(latent.cfg_run if latent is not None else (checkpoint, run_dir)))
    if coupled and latent is None:
        env_cfg.state_checkpoint_path = E.merged_state(
            env_cfg.state_checkpoint_path, state_checkpoint_for(legs_ckpt), "legs", os.path.dirname(OUT))
    if getattr(env_cfg.events, "prepose", None) is not None:
        env_cfg.events.prepose.params["prob"] = 0.0
    env_cfg.episode_length_s = 1.0e4
    env_cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 1.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 1.0

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    if latent is not None:
        runner = None
        latent.bind(env, agent_cfg, args_cli.stochastic)
    else:
        runner = MarlRunner(env, agent_cfg)
        E.check_agent_fits(base, "arms", run_dir, checkpoint)
        E.load_agent(runner, checkpoint, "arms")
        if coupled:
            E.check_agent_fits(base, "legs", runs.run_dir_of(legs_ckpt), legs_ckpt)
            E.load_agent(runner, legs_ckpt, "legs")
        runner.agent.enable_training_mode(False, apply_to_models=True)
    robot = base.scene["robot"]
    arm = base.command_manager.get_term("arm_targets")
    arm._judge_kernels = lambda ended: None
    arm.goals_enabled, arm.goals_enabled_at = True, None
    arm.cfg.offtable_prob = 1.0 if args_cli.offtable else 0.0
    arm.cfg.rest_goal_prob = 0.0
    arm_joints = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)[0]
    with torch.inference_mode():
        obs, _ = base.reset()
        res, obs = run_goals(base, runner, arm, robot, arm_joints, obs)
        if coupled:
            r = vel_cfg.ranges
            rng = np.random.default_rng(args_cli.seed)
            walk = np.stack([rng.uniform(*r.lin_vel_x, (base.num_envs, args_cli.goals)),
                             rng.uniform(*r.lin_vel_y, (base.num_envs, args_cli.goals)),
                             rng.uniform(*r.ang_vel_z, (base.num_envs, args_cli.goals))], -1) * 0.6
            res["walk"], obs = run_goals(base, runner, arm, robot, arm_joints, obs, walk)
    if latent is not None:
        # no IK, so no IK-only reference: the Reference column stays empty
        std, ref_base, ref = latent.policy_std("arms"), None, None
    else:
        std = runner.agent.policies["arms"].log_std_parameter.detach().exp()
        ref_base = out_base(checkpoint, argparse.Namespace(**{**vars(args_cli), "ik_only": True}))
        ref = json.load(open(ref_base + ".json")) if (not args_cli.ik_only and outputs.cached(ref_base)) else None
    res.update(
        backend="isaac", body="arms", run=runs.name(run_dir),
        step=latent.run.step if latent is not None else runs.step_of(checkpoint), checkpoint=checkpoint,
        task=args_cli.task, trainer=latent.trainer if latent is not None else runs.trainer(run_dir),
        overrides=replayed + hydra_args,
        stochastic=args_cli.stochastic, ik_only=args_cli.ik_only, offtable=args_cli.offtable,
        other_policy=(f"legs policy {legs_ckpt}" if coupled else None),
        reach_tol=[float(arm.cfg.reach_pos_tol), float(arm.cfg.reach_rot_tol)],
        policy_std_mean=None if (args_cli.ik_only or std is None) else float(std.mean()),
        policy_std=None if std is None else std.cpu().tolist(), reference=ref_base + ".json" if ref else None,
        # absent for a joint-space run: its json as before
        **({"latent": latent.kind, "settings_run": latent.settings_run} if latent is not None else {}),
        **C.stamp("isaac", "arms", num_envs=base.num_envs, goals=args_cli.goals, seed=args_cli.seed, motors="run",
                  actions="zero" if args_cli.ik_only else "sampled" if args_cli.stochastic else "mean",
                  legs="policy" if coupled else "pinned", offtable=args_cli.offtable, extra_overrides=hydra_args))
    env.close()
    if args_cli.steerability and not args_cli.ik_only:
        steer_out = OUT + "_steerability"
        code = subprocess.call([sys.executable, "-m", "evals.isaac.arms.steerability.run_isaac", "--checkpoint",
                                checkpoint, "--system", "policy", "--suite", "quick", "--split", "dev",
                                "--out", steer_out])
        summary = os.path.join(steer_out, "summary.md")
        res["steerability"] = open(summary).read() if code == 0 and os.path.isfile(summary) else \
            f"(the benchmark exited {code}; see {steer_out})"
        res["steerability_dir"] = steer_out
    scorecard.write(OUT, res, A.rows(res, ref), A.header(res, res["reference"]), A.footer(res))
    print(open(OUT + ".md").read())
    print(f"[INFO] wrote {OUT}.json / .md")


if __name__ == "__main__":
    main()
    simulation_app.close()
