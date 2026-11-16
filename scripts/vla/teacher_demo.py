"""The scripted teacher driving LocoManip-VlaUpper-v0: its success by configuration and a video of the three cameras.

    python scripts/vla/teacher_demo.py --headless --enable_cameras --num_envs 32 --steps 1200 --out logs/vla/teacher_demo
"""

import argparse
import json
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--out", default="logs/vla/teacher_demo")
parser.add_argument("--video_envs", type=int, default=2, help="envs whose cameras go in the video")
parser.add_argument("--video_steps", type=int, default=600)
parser.add_argument("--debug", action="store_true", help="print each env's phase and waypoint error every 20 steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.vla_upper_env_cfg import CAMERA_NAMES, VlaUpperEnvCfg  # noqa: E402
from locomanipulation_game.vla.teacher import PHASE_NAMES, StackTeacher  # noqa: E402


def main():
    cfg = VlaUpperEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.log_dir = os.path.abspath(args_cli.out)
    cfg.training = False
    os.makedirs(cfg.log_dir, exist_ok=True)
    env = gym.make("LocoManip-VlaUpper-v0", cfg=cfg).unwrapped
    env.reset()
    teacher = StackTeacher(env)
    everyone = torch.arange(env.num_envs, device=env.device)
    teacher.reset(everyone)
    legs = torch.zeros(env.num_envs, env.cfg.action_spaces["legs"], device=env.device)
    episodes, start, frame = [], time.time(), 0
    infeasible = int((~teacher.feasible).sum())
    for step in range(args_cli.steps):
        action = teacher.act()
        phase_before = teacher.phase.clone()
        env.step({"legs": legs, "arms": action})
        if step < args_cli.video_steps:
            images = env.camera_images()
            rows = [np.concatenate([np.asarray(Image.fromarray(images[name][e].cpu().numpy()).resize((448 if name == 'head_cam' else 336, 336)))
                                    for name in CAMERA_NAMES], axis=1)
                    for e in range(min(args_cli.video_envs, env.num_envs))]
            Image.fromarray(np.concatenate(rows, axis=0)).save(f"{cfg.log_dir}/f_{frame:04d}.png")
            frame += 1
        ended = env.ended.nonzero(as_tuple=False).flatten()
        for e in ended.tolist():
            episodes.append({"bin": env.ended_bins[e].tolist(), "success": bool(env.ended_success[e]),
                             "phase": PHASE_NAMES[int(phase_before[e])], "feasible": bool(teacher.feasible[e])})
        teacher.reset(ended)
        infeasible += int((~teacher.feasible[ended]).sum())
        if args_cli.debug:
            pos, _ = env.blocks_in_pelvis()
            tcp_p, tcp_R = env.arm_term.measured_tcp()
            for e in range(min(env.num_envs, 3)):
                ph = int(teacher.phase[e])
                if 1 <= ph <= 7:
                    arm = int(env.arm[e])
                    rel = tcp_R[e, arm].T @ (pos[e, env.pick_slot[e]] - tcp_p[e, arm])
                    print(f"[debug] {step} env {e} {PHASE_NAMES[ph]:8s} block in jaws {[round(v, 3) for v in teacher.block_in_jaws[e].tolist()]}"
                          f" angle {teacher.angle[e].item():.2f}"
                          f" size {env.block_size[e, env.pick_slot[e]].item():.3f} closure"
                          f" {env.gripper_term.closure()[e, arm].item():.2f} block z {pos[e, env.pick_slot[e], 2].item():.3f}"
                          f" tcp z {tcp_p[e, arm, 2].item():.3f} lead"
                          f" {(env.arm_term.commanded_tcp()[0][e, arm] - tcp_p[e, arm]).norm().item():.3f}"
                          f" from place {[round(v, 3) for v in teacher.block_from_place[e].tolist()]} slide"
                          f" {[round(v, 2) for v in teacher.slide[e].tolist()]}", flush=True)
        if step % 100 == 0:
            done = len(episodes)
            rate = sum(ep["success"] for ep in episodes) / max(done, 1)
            print(f"[teacher] step {step}: {done} episodes, success {rate:.2f}, {(step + 1) / (time.time() - start):.1f}"
                  f" steps/s, phases {torch.bincount(teacher.phase, minlength=10).tolist()}, longest episode"
                  f" {int(env.episode_length_buf.max())}/{env.max_episode_length} steps", flush=True)
    done = len(episodes)
    summary = {"episodes": done, "success": sum(ep["success"] for ep in episodes) / max(done, 1),
               "infeasible_plans": infeasible, "restarts": teacher.restarts,
               "failed_in_phase": {name: sum(1 for ep in episodes if not ep["success"] and ep["phase"] == name)
                                   for name in PHASE_NAMES},
               "sampler_rate": env.sampler.rate.tolist(), "sampler_count": env.sampler.count.tolist()}
    with open(f"{cfg.log_dir}/summary.json", "w") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, indent=1)
    print("[teacher] " + json.dumps({k: v for k, v in summary.items() if not k.startswith("sampler")}), flush=True)
    if frame:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", f"{cfg.log_dir}/f_%04d.png",
                        "-pix_fmt", "yuv420p", f"{cfg.log_dir}/cameras.mp4"], check=True)
        for name in os.listdir(cfg.log_dir):
            if name.startswith("f_") and int(name[2:6]) % 60:
                os.remove(f"{cfg.log_dir}/{name}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
os._exit(0)  # simulation_app.close() hangs after camera renders
