"""The Magpie grippers opening and closing on coupled hinge targets, the pelvis pinned: a video and the jaw gap.

    python scripts/vla/gripper_demo.py --headless --enable_cameras --out logs/vla/gripper_demo

Each finger is a parallelogram four-bar (crank, finger, rocker) whose closing joint the USD lacks, so the three
hinges are driven together: crank theta, finger -theta, rocker theta (mirrored on the other finger), which keeps the
pad parallel to the gripper's axis as the real linkage does.
"""

import argparse
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="logs/vla/gripper_demo")
parser.add_argument("--theta_closed", type=float, default=1.45, help="crank angle of the closed jaws, rad")
parser.add_argument("--cycles", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG  # noqa: E402

DT = 0.005
FPS = 40


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    cfg = H1_2_MAGPIE_CFG.replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = True
    # arms forward, elbows bent, so both grippers are in front of the torso
    cfg.init_state.joint_pos = {**cfg.init_state.joint_pos, ".*_shoulder_pitch_joint": -0.5, ".*_elbow_joint": 0.3}
    robot = Articulation(cfg)
    camera = Camera(CameraCfg(prim_path="/World/cam", width=960, height=720, data_types=["rgb"],
                              spawn=sim_utils.PinholeCameraCfg(focal_length=24.0)))
    sim.reset()
    # sim.reset() leaves the joints at the USD's zero pose
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    hinge = {}
    for g in ("lg", "rg"):
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for k, s in ((1, 1.0), (2, -1.0), (3, 1.0)):
                hinge[robot.find_joints(f"{g}_{side}_hinge_{k}")[0][0]] = sign * s
    ids = list(hinge)
    signs = torch.tensor([hinge[i] for i in ids], device=sim.device)
    fingers = {g: robot.find_bodies([f"{g}_left_finger", f"{g}_right_finger"], preserve_order=True)[0] for g in ("lg", "rg")}
    mount = robot.find_bodies("rg_mount")[0][0]

    target = robot.data.default_joint_pos.clone()
    os.makedirs(args_cli.out, exist_ok=True)
    period, settle = 3.0, 2.0
    steps = int((settle + args_cli.cycles * period) / DT)
    rows, frame = [], 0
    for i in range(steps):
        t = max(i * DT - settle, 0.0)  # the arms settle under gravity first, jaws open
        closure = 0.5 - 0.5 * math.cos(2 * math.pi * t / period)  # 0 open, 1 closed
        target[:, ids] = closure * args_cli.theta_closed * signs
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        if i == int(settle / DT) - 20:
            # look along the hinge axis (mount x) at the jaws, from a little ahead of them (mount +z)
            rot = matrix_from_quat(robot.data.body_quat_w[0, mount])
            mid = robot.data.body_pos_w[0, fingers["rg"]].mean(0)
            eye = mid + 0.30 * rot[:, 0] * torch.sign(rot[2, 0] + 1e-6) + 0.08 * rot[:, 2]
            camera.set_world_poses_from_view(eye[None], mid[None])
        if i % int(1 / (FPS * DT)) == 0:
            sim.render()
            camera.update(DT)
            if i >= int(settle / DT):
                Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{args_cli.out}/f_{frame:04d}.png")
                frame += 1
            gap = {g: (robot.data.body_pos_w[0, b[0]] - robot.data.body_pos_w[0, b[1]]).norm().item() for g, b in fingers.items()}
            err = (robot.data.joint_pos[0, ids] - target[0, ids]).abs().max().item()
            rows.append((t, closure, gap["lg"], gap["rg"], err))
    rows = np.array(rows)
    np.savetxt(f"{args_cli.out}/gap.csv", rows, header="t closure lg_finger_dist rg_finger_dist max_hinge_err", fmt="%.5f")
    print(f"[gripper] finger-body distance open {rows[0, 2]:.4f} m, min {rows[:, 2].min():.4f} m; "
          f"max hinge tracking error {rows[:, 4].max():.4f} rad", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{args_cli.out}/f_%04d.png",
                    "-pix_fmt", "yuv420p", f"{args_cli.out}/gripper.mp4"], check=True)
    for f in os.listdir(args_cli.out):
        if f.startswith("f_") and int(f[2:6]) % 30:
            os.remove(f"{args_cli.out}/{f}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
os._exit(0)  # simulation_app.close() hangs after a camera render
