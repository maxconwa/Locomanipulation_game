"""A video of a policy in Isaac running THE command program (evals/common/program.py): one env, a third-person camera
following the robot.

    python -m evals.isaac.get_video --legs <run>/checkpoints/agent_<N>.pt [--arms <ckpt>|same]
    python -m evals.isaac.get_video --arms <ckpt>                                  (the pelvis pinned)
        [--every 2] [--out <mp4>] [env.<field>=... overrides]

Legs active: the env's own hoist (the pre-pose reset event, mdp/prepose.py) on the program's fixed schedule: hung with
the feet HANG_LIFT off the ground, lowered onto its feet, the rope released; then the moves, each followed by a stop
(the velocity command's arrows: commanded and actual). Without --arms every arm joint is held at 0 rad. Arms active: a
goal every GOAL_EVERY_S (the arm command's frame markers: the goal, and the wrist's current frame). Arms alone:
LocoManip-ArmsOnly-v0, the pelvis pinned, goals only. Both: LocoManip-Marl-Direct-v0 (evals/isaac/env.py checks the
two runs fit). The env is the main run's (the legs' when given), its overrides replayed, nominal motors, mean actions.

Writes logs/evals/<run>/isaac/<legs|arms>/video_<N>[_with_<arms run>_<M>].mp4 and a _small.mp4 (960 px) if ffmpeg is
there. Needs --enable_cameras (set here).
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # the repo root
from evals.common import conditions as C  # noqa: E402
from evals.common import outputs, program, runs  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--legs")
parser.add_argument("--arms", help="An arms checkpoint, or `same` (the legs checkpoint's own arms).")
parser.add_argument("--every", type=int, default=2)
parser.add_argument("--out")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not (args_cli.legs or args_cli.arms):
    parser.error("--legs and/or --arms")
args_cli.headless, args_cli.enable_cameras = True, True
simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    evaluation_cfgs, replay_run, state_checkpoint_for)

from evals.isaac import env as E  # noqa: E402

LEGS = args_cli.legs is not None
ARMS = args_cli.arms is not None
legs_ckpt = os.path.abspath(args_cli.legs) if LEGS else None
arms_ckpt = (legs_ckpt if args_cli.arms == "same" else os.path.abspath(args_cli.arms)) if ARMS else None
main_ckpt = legs_ckpt or arms_ckpt
task = E.MARL_DIRECT if (LEGS and ARMS) else (None if LEGS else "LocoManip-ArmsOnly-v0")
extra = list(C.NOMINAL_MOTORS) + hydra_args
checkpoint, run_dir, task, replayed = replay_run(main_ckpt, task, extra)
if LEGS and ARMS:
    E.check_runs_compatible(run_dir, runs.run_dir_of(arms_ckpt))
if args_cli.out:
    OUT = args_cli.out
else:
    tag = f"video_{runs.step_of(main_ckpt)}"
    if LEGS and ARMS:
        tag += f"_with_{runs.short(runs.run_dir_of(arms_ckpt))}_{runs.step_of(arms_ckpt)}"
    OUT = str(outputs.eval_dir(runs.name(run_dir), "isaac", "legs" if LEGS else "arms") / f"{tag}.mp4")


def hoist_params(params: dict) -> dict:
    """The pre-pose event set to the program's hoist, every reset: hung, lowered, settled, released."""
    fixed = lambda v: (v, v)  # noqa: E731
    params = dict(params)
    params.update(prob=1.0, hung_prob=1.0, partial_prob=0.0, lift_range=fixed(program.HANG_LIFT),
                  release_now_prob=0.0, lower_prob=1.0, lower_start_range=fixed(program.HANG_S),
                  lower_duration_range=fixed(program.LOWER_S), lower_slack_range=fixed(program.HANG_SLACK),
                  settle_range=fixed(program.SETTLE_S), release_duration_range=fixed(program.RELEASE_S),
                  sag_prob=0.0, fast_vel_prob=0.0, pitch_range=(0.0, 0.0), roll_range=(0.0, 0.0),
                  joint_ranges={k: (0.0, 0.0) for k in params["joint_ranges"]})
    return params


@hydra_task_config(task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 0
    evaluation_cfgs(env_cfg, agent_cfg, checkpoint, run_dir)
    if LEGS and ARMS:
        env_cfg.state_checkpoint_path = E.merged_state(env_cfg.state_checkpoint_path, state_checkpoint_for(arms_ckpt),
                                                       "arms", os.path.dirname(OUT))
    if LEGS and not ARMS:
        env_cfg.actions.arm_pos.hold_zero = True
    prepose = getattr(env_cfg.events, "prepose", None)
    if prepose is not None:
        if LEGS:
            prepose.params = hoist_params(prepose.params)
        else:
            prepose.params["prob"] = 0.0
    env_cfg.episode_length_s = 1.0e4
    env_cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = env_cfg.commands.arm_targets, env_cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 1.0 if ARMS else 0.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis, vel_cfg.debug_vis = ARMS, LEGS
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 0.0
    env_cfg.viewer.origin_type, env_cfg.viewer.asset_name, env_cfg.viewer.env_index = "asset_root", "robot", 0
    # the robot starts facing +x (no drawn yaw), the camera front-left of it at hip height, the whole body in view
    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is not None and "pose_range" in reset_base.params:
        reset_base.params["pose_range"] = {**reset_base.params["pose_range"], "yaw": (0.0, 0.0)}
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (2.2, 2.6, 0.5), (0.0, 0.0, -0.2)
    env_cfg.viewer.resolution = (1280, 720)

    env = SkrlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode="rgb_array"))
    base = env.unwrapped
    runner = MarlRunner(env, agent_cfg)
    if LEGS:
        E.check_agent_fits(base, "legs", run_dir, legs_ckpt)
        E.load_agent(runner, legs_ckpt, "legs")
    if ARMS:
        E.check_agent_fits(base, "arms", runs.run_dir_of(arms_ckpt), arms_ckpt)
        E.load_agent(runner, arms_ckpt, "arms")
    runner.agent.enable_training_mode(False, apply_to_models=True)
    robot, vel = base.scene["robot"], base.command_manager.get_term("base_velocity")
    arm = base.command_manager.get_term("arm_targets")
    arm._judge_kernels = lambda ended: None
    if ARMS:
        arm.goals_enabled, arm.goals_enabled_at = True, None
        arm.cfg.rest_goal_prob = 0.0
    r = vel_cfg.ranges
    full = tuple(float(max(abs(x[0]), abs(x[1]))) for x in (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z))
    segs = program.segments(LEGS, full)
    dt = base.step_dt
    goal_at = set(int(round(t / dt)) for t in program.goal_times(segs, LEGS)) if ARMS else set()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = imageio.get_writer(OUT, fps=round(1.0 / (dt * args_cli.every)), codec="libx264", quality=8,
                                macro_block_size=1)
    mode = ("legs " + ("+ arms policies" if ARMS else "policy, arms held at 0 rad") if LEGS
            else "arms policy, pelvis pinned")
    falls, goal_n = 0, 0
    ids = torch.zeros(1, dtype=torch.long, device=base.device)
    with torch.inference_mode():
        obs, _ = base.reset()
        states = base.state()
        for _ in range(20):
            base.sim.render()
        for k in range(int(round(program.duration(segs) / dt))):
            t = k * dt
            seg = program.at(segs, t)
            vel.vel_command_b[:] = torch.tensor([seg.cmd], dtype=torch.float, device=base.device)
            vel.time_left[:] = 1.0e6
            vel.is_standing_env[:] = False
            if k in goal_at:
                arm._resample(ids)
                goal_n += 1
            outputs_ = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
            actions = {a: outputs_[-1][a].get("mean_actions", outputs_[0][a]) for a in base.possible_agents}
            if LEGS and not ARMS:
                actions["arms"] = torch.zeros_like(actions["arms"])
            obs, _, terminated, truncated, _ = base.step(actions)
            states = base.state()
            agent = base.possible_agents[0]
            runner.agent.reset_rnn(terminated[agent] | truncated[agent])
            falls += int(bool(terminated[agent][0]))
            if k % args_cli.every:
                continue
            frame = base.render()
            if frame is None or frame.size == 0:
                continue
            frame = np.ascontiguousarray(frame[..., :3])
            q = robot.data.root_quat_w[0]
            yaw = torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            vw = robot.data.root_lin_vel_w[0]
            c, s = torch.cos(yaw), torch.sin(yaw)
            meas = (float(c * vw[0] + s * vw[1]), float(-s * vw[0] + c * vw[1]), float(robot.data.root_ang_vel_w[0, 2]))
            lines = [f"Isaac Lab ({task}), {mode}   t = {t:5.1f} s   {seg.name}"]
            if LEGS:
                lines += [f"cmd  vx {seg.cmd[0]:+.2f}  vy {seg.cmd[1]:+.2f}  wz {seg.cmd[2]:+.2f}",
                          f"meas vx {meas[0]:+.2f}  vy {meas[1]:+.2f}  wz {meas[2]:+.2f}   falls {falls}"]
                hoist = getattr(base, "_prepose", None)
                share = float(hoist.support_share(base.episode_length_buf * dt)[0]) if hoist is not None else 0.0
                if share > 0:
                    lines.append(f"hoist carries {share:.0%} of the weight")
            if ARMS and bool(arm.arm_mode[0]):
                pe, re_ = arm.errors()
                lines.append(f"arm goal {goal_n}: wrist error L {1000 * float(pe[0, 0]):5.1f} mm"
                             f" {np.degrees(float(re_[0, 0])):4.1f} deg,  R {1000 * float(pe[0, 1]):5.1f} mm"
                             f" {np.degrees(float(re_[0, 1])):4.1f} deg")
            legend = (["arrows: commanded and actual velocity (Isaac's command markers)"] if LEGS else []) + \
                     (["frames: the arm goal (large) and the wrist (its current pose)"] if ARMS else [])
            h = frame.shape[0]
            for i, line in enumerate(lines):
                y = 28 + 26 * i
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            for i, line in enumerate(legend):
                y = h - 16 - 26 * (len(legend) - 1 - i)
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            writer.append_data(frame)
    writer.close()
    print(f"[evals] wrote {OUT}: {program.duration(segs):.0f} s, falls {falls}")
    env.close()
    if shutil.which("ffmpeg"):
        small = OUT[:-4] + "_small.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT, "-vf", "scale=960:-2", "-crf", "28", small],
                       check=False)
        print(f"[evals] wrote {small}")


if __name__ == "__main__":
    main()
    simulation_app.close()
