"""Record the scripted teacher's episodes of LocoManip-VlaUpper-v0 as a dataset of raw camera images.

    python scripts/vla/collect_dataset.py --headless --enable_cameras --out logs/vla/datasets/teacher --episodes 5000 \
        --seed 100 --block_seed 100

Each saved episode is one trajectory: the instruction, then per 20 Hz step the observation (the three cameras' images,
the arms' proprioception) and the teacher's action taken on it. Only the episodes the teacher stacked are kept. The
floor is chroma green (vla_upper_env_cfg.GREEN_SCREEN) for background augmentation. Run again with the same --out to
add episodes (other seeds: the envs' block sizes are fixed per process, by --block_seed). <out>/README.md describes
the files.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True)
parser.add_argument("--episodes", type=int, default=5000, help="successful episodes to add")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--block_seed", type=int, default=100, help="the envs' block sizes (training: 0)")
parser.add_argument("--crf", type=int, default=12, help="x264 quality (lower is better; 18 is visually lossless)")
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--antialiasing", default="Off", help="the renderer's mode: the default DLSS blends frames, which left\n"
                    "streaks of the last episode in a new one's first frames and of the next camera's tile along the image edge")
parser.add_argument("--no_green", action="store_true", help="keep the grey floor")
parser.add_argument("--visuals", default=None, metavar="TEXTURE_DIR",
                    help="draw each episode's table surface, block finish and lamp (mdp/vla.py VisualRandomizer), with the\n"
                    "textures of this folder (<name>.jpg and index.json; 'plain' for none)")
parser.add_argument("--reflections", action="store_true", help="with --visuals: mirror reflections and fully metallic blocks")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.mdp.vla import (  # noqa: E402
    BLOCK_COLORS, BLOCK_SIZE_NAMES, VisualRandomizationCfg, make_matte,
)
from locomanipulation_game.tasks.locomanip_marl.vla_upper_env_cfg import CAMERA_NAMES, GREEN_SCREEN, VlaUpperEnvCfg  # noqa: E402
from locomanipulation_game.vla.teacher import PHASE_NAMES, StackTeacher  # noqa: E402

SHARD = 1000  # episodes per directory
TABLES = "logs/steerability/dev/policy/arm_target_tables_armsonly_v2.pt"

README = """# Teacher trajectories of LocoManip-VlaUpper-v0

Block stacking with the Unitree H1-2's arms and Magpie grippers (pelvis pinned), driven by the scripted, privileged
teacher (vla/teacher.py). Only episodes the teacher stacked are here. The floor is chroma green, an unlit sheet of RGB {green}
(no shadows on it), for background replacement; one of the block colours is a darker green {block_green}.

## Files

- `meta.jsonl`: one line per episode: `id`, `instruction` (the language goal), `length` (steps T), `arm`, `both_reach`
  (either arm's plan was feasible; the teacher then picked one at random), `blocks` (per block on the table: slot,
  size name, edge m, colour), `pick_slot`, `target_slot`, `table_top` (m), `seed`, `block_seed`, `env`.
- `episodes/<id // {shard}>/<id>_<camera>.mp4`: the camera's T frames at {fps:.0f} fps, H.264 yuv444p crf {crf}
  (full-resolution chroma; decode with ffmpeg, PyAV, decord or OpenCV). Cameras: {cameras}.
- `episodes/<id // {shard}>/<id>.npz`:
  - `proprio` (T, {proprio}) float32: the arms' observation (joint positions, velocities, gripper closure, with the
    encoders' noise; then the controller's commanded grasp points), as the policy gets it
  - `action` (T, 14) float32 in [-1, 1]: per arm (left, right) the grasp point's position delta (x {pos_scale} m) and
    rotation delta as axis-angle (x {rot_scale} rad), in the pelvis frame; then each gripper's command (+1 close, -1 open)
  - `phase` (T,) uint8: the teacher's phase, an index into {phases}
  - `block_pos` (T, {slots}, 3), `block_quat` (T, {slots}, 4 wxyz) float32: every block slot in the pelvis frame
    (privileged; blocks not on the table are parked behind the robot)

A trajectory is: instruction, then (observation_t, action_t) for t = 0..T-1, where observation_t is frame t of each
video and `proprio[t]`, and `action[t]` is what the teacher did next. The episode ended (stacked and held
{hold} s) after `action[T-1]`.

## Reading an episode

```python
import json, numpy as np, imageio.v3 as iio
ep = json.loads(open("meta.jsonl").readline())
d = f"episodes/{{ep['id'] // {shard}:03d}}/{{ep['id']:06d}}"
data = np.load(d + ".npz")
head = iio.imread(d + "_head_cam.mp4")  # (T, H, W, 3) uint8
assert len(head) == ep["length"] == len(data["action"])
```
"""


def encode(path: str, frames: np.ndarray, fps: float, crf: int):
    """frames (T, H, W, 3) uint8 -> an H.264 file, every frame kept, keyframes every 16."""
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv444p", "-g", "16",
           "-threads", "1", path]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[-500:])


def save_episode(out: str, meta: dict, images: dict, arrays: dict, fps: float, crf: int) -> dict:
    stem = f"{out}/episodes/{meta['id'] // SHARD:03d}/{meta['id']:06d}"
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for name, frames in images.items():
        encode(f"{stem}_{name}.mp4", np.stack(frames), fps, crf)
    np.savez_compressed(stem + ".npz", **{k: np.stack(v) for k, v in arrays.items()})
    return meta


def main():
    out = os.path.abspath(args_cli.out)
    os.makedirs(out + "/episodes", exist_ok=True)
    if os.path.isfile(TABLES):  # the arms task's target tables (unused here): building them takes a contact test
        shutil.copy(TABLES, out)
    meta_path = out + "/meta.jsonl"
    first = sum(1 for _ in open(meta_path)) if os.path.isfile(meta_path) else 0
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    cfg = VlaUpperEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.block_seed = args_cli.block_seed
    cfg.sampler_floor = 1.0  # uniform over the configurations
    cfg.log_dir = out
    cfg.training = False
    cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    cfg.scene.env_spacing = 6.0  # at the task's 4 m the head camera sees the next env's feet and parked blocks
    if not args_cli.no_green:
        # a sheet over the ground plane (which draws a grid), lit by nothing: one colour, no shadows
        cfg.scene.green_floor = AssetBaseCfg(
            prim_path="/World/GreenFloor", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
            spawn=sim_utils.CuboidCfg(size=(1000.0, 1000.0, 0.002), visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0), emissive_color=GREEN_SCREEN, roughness=1.0)))
    if args_cli.visuals:
        cfg.visuals = VisualRandomizationCfg(
            seed=args_cli.seed, texture_dir="" if args_cli.visuals == "plain" else os.path.abspath(args_cli.visuals))
        if args_cli.reflections:
            cfg.visuals.block_metallic_range = (0.3, 1.0)
            cfg.sim.render.enable_reflections = True
    env = gym.make("LocoManip-VlaUpper-v0", cfg=cfg).unwrapped
    if not args_cli.no_green:
        make_matte("/World/GreenFloor")  # no sheen of the lights on the floor: every pixel of it one colour
    n, dev = env.num_envs, env.device
    teacher = StackTeacher(env)
    fps = 1.0 / env.step_dt
    color_names = list(BLOCK_COLORS)

    obs, _ = env.reset()
    every = torch.arange(n, device=dev)
    teacher.reset(every)
    legs = torch.zeros(n, env.cfg.action_spaces["legs"], device=dev)
    keys = ("proprio", "action", "phase", "block_pos", "block_quat")
    images = [{name: [] for name in CAMERA_NAMES} for _ in range(n)]
    arrays = [{k: [] for k in keys} for _ in range(n)]

    def episode_meta(e: int) -> dict:
        slots = env.active[e].nonzero().flatten().tolist()
        return {"instruction": env.instructions[e], "arm": ("left", "right")[int(env.arm[e])],
                "both_reach": bool(teacher.both_reach[e]),
                "blocks": [{"slot": s, "size": BLOCK_SIZE_NAMES[int(env.block_class[e, s])], "edge_m": round(float(env.block_size[e, s]), 4),
                            "color": color_names[int(env.block_color[e, s])]} for s in slots],
                "pick_slot": int(env.pick_slot[e]), "target_slot": int(env.target_slot[e]),
                "table_top": round(float(env.table_top[e]), 4), "seed": args_cli.seed, "block_seed": args_cli.block_seed, "env": e,
                **({"visuals": env.visuals.draws[e]} if env.visuals is not None else {})}

    metas = [episode_meta(e) for e in range(n)]
    if first == 0:
        with open(out + "/README.md", "w") as f:
            f.write(README.format(green=GREEN_SCREEN, block_green=BLOCK_COLORS["green"], shard=SHARD, fps=fps, crf=args_cli.crf,
                                  cameras=", ".join(f"{name} {tuple(v.shape[1:3])[::-1]}" for name, v in env.camera_images().items()),
                                  proprio=obs["arms"].shape[1], pos_scale=env.arm_term.cfg.pos_scale, rot_scale=env.arm_term.cfg.rot_scale,
                                  phases=PHASE_NAMES, slots=len(env.blocks), hold=cfg.success_hold_s))
    pool = ThreadPoolExecutor(args_cli.workers)
    pending, submitted, saved, ended_all, steps, frames = [], 0, 0, 0, 0, 0
    start = time.time()
    meta_file = open(meta_path, "a")

    def collect(block: bool):
        nonlocal saved, frames
        for job in list(pending):
            if block or job.done():
                meta = job.result()
                meta_file.write(json.dumps(meta) + "\n")
                meta_file.flush()
                saved += 1
                frames += meta["length"]
                pending.remove(job)

    while submitted < args_cli.episodes:
        action = teacher.act()
        cams = {name: v.cpu().numpy() for name, v in env.camera_images().items()}
        pos, quat = env.blocks_in_pelvis()
        step = {"proprio": obs["arms"].cpu().numpy().astype(np.float32), "action": action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32),
                "phase": teacher.phase.cpu().numpy().astype(np.uint8), "block_pos": pos.cpu().numpy().astype(np.float32),
                "block_quat": quat.cpu().numpy().astype(np.float32)}
        for e in range(n):
            for name in CAMERA_NAMES:
                images[e][name].append(cams[name][e].copy())
            for k in keys:
                arrays[e][k].append(step[k][e])
        obs, _, _, _, _ = env.step({"legs": legs, "arms": action.clamp(-1.0, 1.0)})
        steps += 1
        ended = env.ended.nonzero(as_tuple=False).flatten()
        success = env.ended_success.clone()
        if len(ended):
            teacher.reset(ended)
        for e in ended.tolist():
            ended_all += 1
            if success[e] and submitted < args_cli.episodes:
                meta = {"id": first + submitted, "length": len(arrays[e]["action"]), **metas[e]}
                pending.append(pool.submit(save_episode, out, meta, images[e], arrays[e], fps, args_cli.crf))
                submitted += 1
            images[e] = {name: [] for name in CAMERA_NAMES}
            arrays[e] = {k: [] for k in keys}
            metas[e] = episode_meta(e)
        collect(block=False)
        while len(pending) > 4 * args_cli.workers:  # the encoders are behind: wait rather than hold the frames
            collect(block=True)
        if steps % 200 == 0:
            took = time.time() - start
            print(f"[collect] step {steps}: {saved} saved ({first + saved} in all), teacher success {submitted / max(ended_all, 1):.3f},"
                  f" {steps / took:.2f} steps/s, {submitted / took * 3600:.0f} episodes/h, {len(pending)} encoding", flush=True)
    collect(block=True)
    meta_file.close()
    print(f"[collect] added {saved} episodes ({frames} steps) in {time.time() - start:.0f} s; {first + saved} in {out}", flush=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
import sys  # noqa: E402

sys.stdout.flush()
os._exit(0)
