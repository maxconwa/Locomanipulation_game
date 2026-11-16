"""Evaluate a policy of LocoManip-VlaUpper-v0: its success by configuration, how the failures went, a camera video.

    python scripts/vla/eval.py --headless --enable_cameras --checkpoint logs/vla/<run>/checkpoints/policy_<N>.pt
    python scripts/vla/eval.py --headless --enable_cameras --teacher --out logs/vla/teacher_eval

Configurations are drawn uniformly (the success-guided sampler's floor at 1), the policy acts deterministically, each
env runs the same number of episodes, `--episodes` in all (rounded up). With --teacher the scripted teacher drives instead (the reference).
Writes <out>/eval.json (default out: <run>/eval_<N>) and cameras.mp4: per row one of the first `--video_envs` envs,
an outside view of the robot and the table, then its three cameras.

A failed episode is one of: no_grasp (no block ever left the table), wrong_block (only blocks other than the
instructed one did), misplaced (the instructed block was lifted, and the episode ended with it elsewhere).
against_teacher compares the policy's action with the teacher's at the same states, by the teacher's phase.
"""

import argparse
import json
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--teacher", action="store_true")
parser.add_argument("--out", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=300)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--block_seed", type=int, default=0, help="the envs' block sizes and colours (training: 0)")
parser.add_argument("--video_envs", type=int, default=4)
parser.add_argument("--video_steps", type=int, default=800)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.teacher and not args_cli.checkpoint:
    parser.error("give --checkpoint or --teacher")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.mdp.vla import BLOCK_SIZE_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.vla_upper_env_cfg import CAMERA_NAMES, VlaUpperEnvCfg  # noqa: E402
from locomanipulation_game.vla.teacher import PHASE_NAMES, StackTeacher  # noqa: E402

OUTSIDE_EYE, OUTSIDE_TARGET = (1.75, -1.15, 1.75), (0.25, 0.0, 1.0)  # m, env frame: the video's outside view


def outside_camera():
    """A camera per env looking at the robot and the table from the front right, above (for the video only)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils.math import quat_from_matrix

    eye, target = torch.tensor(OUTSIDE_EYE), torch.tensor(OUTSIDE_TARGET)
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    left = torch.nn.functional.normalize(torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0]), forward), dim=0)
    rot = quat_from_matrix(torch.stack([forward, left, torch.linalg.cross(forward, left)], dim=1))
    return TiledCameraCfg(prim_path="{ENV_REGEX_NS}/outside_cam", width=448, height=336, data_types=["rgb"],
                          offset=TiledCameraCfg.OffsetCfg(pos=OUTSIDE_EYE, rot=tuple(rot.tolist()), convention="world"),
                          spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, clipping_range=(0.05, 30.0)))

TABLES = "logs/steerability/dev/policy/arm_target_tables_armsonly_v2.pt"
LIFTED = 0.02  # m above its resting height: a block counts as lifted


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if args_cli.out:
        out = os.path.abspath(args_cli.out)
    else:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args_cli.checkpoint)))
        out = os.path.join(run_dir, "eval_" + os.path.basename(args_cli.checkpoint).split("_")[-1].split(".")[0])
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(TABLES):
        import shutil

        shutil.copy(TABLES, out)
    cfg = VlaUpperEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.block_seed = args_cli.block_seed
    cfg.sampler_floor = 1.0  # uniform over the configurations
    cfg.log_dir = out
    cfg.training = False
    if args_cli.video_envs:
        cfg.scene.outside_cam = outside_camera()
    env = gym.make("LocoManip-VlaUpper-v0", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    teacher = StackTeacher(env)
    runner = None
    if not args_cli.teacher:
        from locomanipulation_game.vla.encoders import FrozenEncoders
        from locomanipulation_game.vla.model import VlaPolicy
        from locomanipulation_game.vla.runner import PolicyRunner

        state = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)
        policy = VlaPolicy(**state["config"]).to(dev).eval()
        policy.load_state_dict(state["policy"])
        train_args = state.get("args", {})
        runner = PolicyRunner(env, FrozenEncoders(dev), policy, replan=train_args.get("replan", 4),
                              vlm_every=train_args.get("vlm_every", 8))
        print(f"[eval] {args_cli.checkpoint} (step {state['step']})", flush=True)

    obs, _ = env.reset()
    everyone = torch.arange(n, device=dev)
    teacher.reset(everyone)
    if runner:
        runner.reset(everyone)
    legs = torch.zeros(n, env.cfg.action_spaces["legs"], device=dev)
    rows = torch.arange(n, device=dev)
    # per episode: how high the instructed block and the highest other block on the table got above their start
    start_z = torch.stack([b.data.root_pos_w[:, 2] for b in env.blocks], dim=1)
    pick_rise = torch.zeros(n, device=dev)
    other_rise = torch.zeros(n, device=dev)
    episodes, frame, step = [], 0, 0
    agree = torch.zeros(len(PHASE_NAMES), 5, device=dev)
    # every env runs the same number of episodes: counting the first to end overall would favour the short ones,
    # which are the successes
    per_env = -(-args_cli.episodes // n)
    counted = torch.zeros(n, dtype=torch.long, device=dev)
    while int(counted.min()) < per_env:
        images = env.camera_images()
        label = teacher.act()  # also when the policy drives: the teacher's phase says how far the episode got
        if runner:
            runner.refresh_task(images["head_cam"])
            wrist = torch.stack([images["left_wrist_cam"], images["right_wrist_cam"]], dim=1)
            action = runner.act(wrist, runner.encoders.head_small(images["head_cam"]), obs["arms"],
                                torch.ones(n, dtype=torch.bool, device=dev))
            # how the policy's action compares with the teacher's there, by the teacher's phase: the planned arm's
            # position delta's size, the teacher's, and the cosine between them
            a_s = action.view(n, -1)[:, :12].view(n, 2, 6)[rows, env.arm, :3]
            a_t = label[:, :12].view(n, 2, 6)[rows, env.arm, :3]
            cos = torch.nn.functional.cosine_similarity(a_s, a_t, dim=1)
            grip_ok = ((action[:, 12:][rows, env.arm] > 0) == (label[:, 12:][rows, env.arm] > 0)).float()
            for k in range(len(PHASE_NAMES)):
                m = teacher.phase == k
                if m.any():
                    agree[k] += torch.tensor([m.sum(), a_s[m].norm(dim=1).sum(), a_t[m].norm(dim=1).sum(),
                                              cos[m].sum(), grip_ok[m].sum()], device=dev)
        else:
            action = label
        if step < args_cli.video_steps and args_cli.video_envs:
            views = {"outside_cam": env.scene["outside_cam"].data.output["rgb"][..., :3], **images}
            tiles = [np.concatenate([np.asarray(Image.fromarray(views[name][e].cpu().numpy()).resize((336 if "wrist" in name else 448, 336)))
                                     for name in ("outside_cam", *CAMERA_NAMES)], axis=1) for e in range(min(args_cli.video_envs, n))]
            Image.fromarray(np.concatenate(tiles, axis=0)).save(f"{out}/f_{frame:05d}.png")
            frame += 1
        z = torch.stack([b.data.root_pos_w[:, 2] for b in env.blocks], dim=1)
        rise = (z - start_z) * env.active
        pick_rise = torch.maximum(pick_rise, rise[rows, env.pick_slot])
        rise[rows, env.pick_slot] = 0.0
        other_rise = torch.maximum(other_rise, rise.amax(dim=1))
        pick_class = env.block_class[rows, env.pick_slot].clone()
        target_class = env.block_class[rows, env.target_slot].clone()
        instructions = list(env.instructions)
        lengths = env.episode_length_buf.clone()
        arms = env.arm.clone()
        both = teacher.both_reach.clone()
        obs, _, _, _, _ = env.step({"legs": legs, "arms": action.clamp(-1.0, 1.0)})
        if runner:
            runner.step()
        step += 1
        ended = env.ended.nonzero(as_tuple=False).flatten()
        for e in ended.tolist():
            counted[e] += 1
            if counted[e] > per_env:
                continue
            success = bool(env.ended_success[e])
            if success:
                outcome = "success"
            elif pick_rise[e] > LIFTED:
                outcome = "misplaced"
            elif other_rise[e] > LIFTED:
                outcome = "wrong_block"
            else:
                outcome = "no_grasp"
            b = env.ended_bins[e].tolist()
            episodes.append({"success": success, "outcome": outcome, "blocks": b[0] + cfg.num_blocks_range[0],
                             "pick": BLOCK_SIZE_NAMES[int(pick_class[e])], "target": BLOCK_SIZE_NAMES[int(target_class[e])],
                             "arm": ("left", "right")[int(arms[e])], "both_reach": bool(both[e]), "seconds": float(lengths[e] + 1) * env.step_dt,
                             "instruction": instructions[e]})
        if len(ended):
            teacher.reset(ended)
            if runner:
                runner.reset(ended)
            start_z[ended] = torch.stack([b.data.root_pos_w[ended, 2] for b in env.blocks], dim=1)
            pick_rise[ended] = 0.0
            other_rise[ended] = 0.0
        if step % 200 == 0:
            done = len(episodes)
            print(f"[eval] step {step}: {done} episodes, success"
                  f" {sum(ep['success'] for ep in episodes) / max(done, 1):.3f}", flush=True)

    def rate(key=None, value=None):
        chosen = [ep for ep in episodes if key is None or ep[key] == value]
        return {"episodes": len(chosen), "success": sum(ep["success"] for ep in chosen) / max(len(chosen), 1)}

    summary = {
        "policy": "teacher" if args_cli.teacher else args_cli.checkpoint,
        "overall": rate(),
        "outcomes": {name: sum(ep["outcome"] == name for ep in episodes) / len(episodes)
                     for name in ("success", "misplaced", "wrong_block", "no_grasp")},
        "by_blocks": {str(k): rate("blocks", k) for k in sorted({ep["blocks"] for ep in episodes})},
        "by_pick": {k: rate("pick", k) for k in BLOCK_SIZE_NAMES},
        "by_target": {k: rate("target", k) for k in BLOCK_SIZE_NAMES},
        "by_arm": {k: rate("arm", k) for k in ("left", "right")},
        "both_arms_reached": rate("both_reach", True),
        "against_teacher": {name: {"steps": int(row[0]), "policy_delta": round(float(row[1] / row[0]), 3),
                                   "teacher_delta": round(float(row[2] / row[0]), 3),
                                   "cosine": round(float(row[3] / row[0]), 3), "gripper_agrees": round(float(row[4] / row[0]), 3)}
                            for name, row in zip(PHASE_NAMES, agree) if row[0] > 0},
        "success_seconds_mean": float(np.mean([ep["seconds"] for ep in episodes if ep["success"]] or [float("nan")])),
    }
    with open(f"{out}/eval.json", "w") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, indent=1)
    print("[eval] " + json.dumps(summary), flush=True)
    if frame:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", f"{out}/f_%05d.png",
                        "-pix_fmt", "yuv420p", f"{out}/cameras.mp4"], check=True)
        for name in os.listdir(out):
            if name.startswith("f_"):
                os.remove(f"{out}/{name}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
os._exit(0)  # simulation_app.close() hangs after camera renders
