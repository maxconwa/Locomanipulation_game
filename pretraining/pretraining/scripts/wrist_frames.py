"""Do the clips' wrist frames agree with main's? A kinematic replay of clips on main's plant.

  python -m pretraining.scripts.wrist_frames --motion-dir <dir> [--clips 40] [--stride 5] --out <dir>

Each chosen clip is replayed kinematically on ``robot_cfg(MAIN)`` (gravity off, no ground; the root pose and the joints
are written at every replayed frame and the articulation's kinematics updated, nothing is simulated forward), and
``<out>/wrist_frames.json`` gets, as aggregates only (no clip names: the clips' licence):

- ``fk_pos_err_m`` / ``fk_rot_err_deg`` (p50, p95, max over frames and both wrists): Isaac's ``*_wrist_yaw_link`` pose
  in the pelvis frame against ``pretraining.kinematics``' FK (the URDF stage 1 tracks on) on the same frames;
- ``goal_coverage``: the share of main's arm-target table poses (``ArmTargetsCommandCfg.table_file``, per arm and
  over both) with a clip wrist pose of the same arm within ``POS_TOL_M`` and ``ROT_TOL_DEG``;
- ``goal_vs_tool_frame``: the rotation (deg, max over frames) between main's goal frame (``*_wrist_yaw_link`` in the
  pelvis frame, as ArmTargetsCommand takes it) and stage 1's tool frame (``marl.rewards.ee_in_pelvis``: the same link
  plus ``GRASP_OFFSET``, a translation, so 0 is expected) and that offset.

Exit 0 once written (``os._exit``: the app's close has hung).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import torch

from pretraining.math import quat_error_magnitude, subtract_frame_transforms

POS_TOL_M = 0.1
ROT_TOL_DEG = 30.0
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")  # main's LEFT_EE_BODY, RIGHT_EE_BODY
_PRIM = "/World/WristFramesRobot"


def percentiles(x: torch.Tensor) -> dict[str, float]:
    """p50, p95 and max of ``x`` (any shape)."""
    x = x.flatten().double()
    return {"p50": float(x.quantile(0.5)), "p95": float(x.quantile(0.95)), "max": float(x.max())}


def in_pelvis(pelvis_pos, pelvis_quat, body_pos, body_quat):
    """Bodies ``(..., B, 3|4)`` in the pelvis frame ``(..., 3|4)``: positions and wxyz quaternions."""
    shape = body_pos.shape[:-1]
    pos, quat = subtract_frame_transforms(
        pelvis_pos.unsqueeze(-2).expand(*shape, 3).reshape(-1, 3),
        pelvis_quat.unsqueeze(-2).expand(*shape, 4).reshape(-1, 4), body_pos.reshape(-1, 3), body_quat.reshape(-1, 4))
    return pos.view(*shape, 3), quat.view(*shape, 4)


def pose_errors(pos_a, quat_a, pos_b, quat_b) -> tuple[torch.Tensor, torch.Tensor]:
    """Per pose: position error (m) and rotation error (deg)."""
    pos_err = (pos_a - pos_b).norm(dim=-1)
    rot_err = quat_error_magnitude(quat_a.reshape(-1, 4), quat_b.reshape(-1, 4)).view(pos_err.shape)
    return pos_err, torch.rad2deg(rot_err)


def coverage(goal_pos, goal_quat, pos, quat, pos_tol=POS_TOL_M, rot_tol_deg=ROT_TOL_DEG, chunk=2048) -> torch.Tensor:
    """(G,) bool: each goal pose ``(G, 3|4)`` has a pose ``(P, 3|4)`` within ``pos_tol`` m and ``rot_tol_deg``."""
    cos_half = math.cos(math.radians(rot_tol_deg) / 2.0)
    pos, quat = pos.to(goal_pos), quat.to(goal_quat)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    out = []
    for g in range(0, goal_pos.shape[0], chunk):
        gp, gq = goal_pos[g:g + chunk], goal_quat[g:g + chunk]
        gq = gq / gq.norm(dim=-1, keepdim=True)
        near = torch.cdist(gp, pos) <= pos_tol
        # the angle between two unit quaternions is 2 acos |<q1, q2>| (either sign: the double cover)
        aligned = (gq @ quat.T).abs() >= cos_half
        out.append((near & aligned).any(dim=1))
    return torch.cat(out)


def tool_frame_rotation_deg(pelvis_pos, pelvis_quat, wrist_pos, wrist_quat) -> tuple[torch.Tensor, list[float]]:
    """(frames, 2) angle (deg) between main's goal frame (the wrist links in the pelvis frame) and stage 1's tool
    frame (``ee_in_pelvis`` on the same bodies), and stage 1's ``GRASP_OFFSET``."""
    from pretraining.constants import TRACKED_BODIES
    from pretraining.marl.config import GRASP_OFFSET, PELVIS, WRISTS
    from pretraining.marl.rewards import ee_in_pelvis
    from pretraining.math import matrix_from_quat

    n = pelvis_pos.shape[0]
    # ee_in_pelvis reads the tracked bodies by index: the pelvis and the wrists set, the others unused
    body_pos = torch.zeros(n, len(TRACKED_BODIES), 3, dtype=pelvis_pos.dtype, device=pelvis_pos.device)
    body_quat = torch.zeros(n, len(TRACKED_BODIES), 4, dtype=pelvis_pos.dtype, device=pelvis_pos.device)
    body_quat[..., 0] = 1.0
    body_pos[:, PELVIS], body_quat[:, PELVIS] = pelvis_pos, pelvis_quat
    for k, i in enumerate(WRISTS):
        body_pos[:, i], body_quat[:, i] = wrist_pos[:, k], wrist_quat[:, k]
    _, tool_rot = ee_in_pelvis(body_pos, matrix_from_quat(body_quat))
    _, goal_quat = in_pelvis(pelvis_pos, pelvis_quat, wrist_pos, wrist_quat)
    rel = matrix_from_quat(goal_quat).transpose(-1, -2) @ tool_rot
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)), list(GRASP_OFFSET)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--clips", type=int, default=40, help="clips drawn with --seed (all, if fewer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=5, help="replay every stride-th frame of a clip")
    p.add_argument("--table", type=Path, default=None,
                   help="arm-target tables (.pt; default main's ArmTargetsCommandCfg.table_file, ARM_TARGET_TABLE)")
    p.add_argument("--out", type=Path, required=True)
    return p


def replay(args) -> dict:
    """Needs the Isaac app (``main`` launches it)."""
    import isaaclab.sim as sim_utils
    import numpy as np
    from isaaclab.assets import Articulation

    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import ARM_TARGET_TABLE
    from pretraining.constants import JOINT_NAMES
    from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
    from pretraining.motions import check_motion_dir
    from pretraining.plants import MAIN
    from pretraining.robot import robot_cfg

    files = check_motion_dir(args.motion_dir)
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in sorted(rng.choice(len(files), min(args.clips, len(files)), replace=False))]
    fk = ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), ("pelvis", *WRIST_BODIES))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=MAIN.sim_dt, device="cpu", gravity=(0.0, 0.0, 0.0)))
    robot = Articulation(robot_cfg(MAIN).replace(prim_path=_PRIM))
    sim.reset()
    ids = robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0]
    body_ids = robot.find_bodies(["pelvis", *WRIST_BODIES], preserve_order=True)[0]
    isaac_pos, isaac_quat, fk_pos, fk_quat, pel_pos, pel_quat, wr_pos, wr_quat = ([] for _ in range(8))
    frames = 0
    for f in files:
        clip = np.load(f)
        names = [str(n) for n in clip["joint_names"]]
        cols = [names.index(j) for j in JOINT_NAMES]
        for t in range(0, clip["joint_pos"].shape[0], args.stride):
            q = torch.tensor(clip["joint_pos"][t][cols], dtype=torch.float32)
            root = torch.cat([torch.tensor(clip["root_pos_w"][t], dtype=torch.float32),
                              torch.tensor(clip["root_quat_w"][t], dtype=torch.float32)]).unsqueeze(0)
            root[:, 3:] /= root[:, 3:].norm(dim=-1, keepdim=True)
            joint_pos = robot.data.default_joint_pos.clone()
            joint_pos[0, ids] = q
            robot.write_root_pose_to_sim(root)
            robot.write_root_velocity_to_sim(torch.zeros(1, 6))
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
            sim.forward()  # the link transforms of the written state, nothing stepped
            robot.update(MAIN.sim_dt)
            bp, bq = robot.data.body_pos_w[0, body_ids], robot.data.body_quat_w[0, body_ids]
            pos, quat = in_pelvis(bp[0], bq[0], bp[1:], bq[1:])
            isaac_pos.append(pos)
            isaac_quat.append(quat)
            pel_pos.append(bp[0])
            pel_quat.append(bq[0])
            wr_pos.append(bp[1:])
            wr_quat.append(bq[1:])
            fpos, fquat, _, _ = fk(root[0, :3], root[0, 3:], q)
            pos, quat = in_pelvis(fpos[0], fquat[0], fpos[1:], fquat[1:])
            fk_pos.append(pos)
            fk_quat.append(quat)
            frames += 1
    sim_utils.delete_prim(_PRIM)
    sim.clear_instance()

    stack = torch.stack
    isaac_pos, isaac_quat, fk_pos, fk_quat = stack(isaac_pos), stack(isaac_quat), stack(fk_pos), stack(fk_quat)
    pos_err, rot_err = pose_errors(isaac_pos, isaac_quat, fk_pos, fk_quat)
    rot, offset = tool_frame_rotation_deg(stack(pel_pos), stack(pel_quat), stack(wr_pos), stack(wr_quat))

    table_path = args.table or Path(ARM_TARGET_TABLE)
    data = torch.load(table_path, map_location="cpu", weights_only=False)
    if list(data["body_names"]) != list(WRIST_BODIES):
        raise ValueError(f"{table_path}: tables for {data['body_names']}, expected {list(WRIST_BODIES)}")
    per_arm, hits, total = {}, 0, 0
    for arm, (name, table) in enumerate(zip(("left", "right"), data["tables"])):
        # the tables' rows: position and wxyz quaternion of the wrist link in the pelvis frame (ArmTargetsCommand)
        covered = coverage(table[:, :3].float(), table[:, 3:7].float(), isaac_pos[:, arm], isaac_quat[:, arm])
        per_arm[name] = float(covered.float().mean())
        hits, total = hits + int(covered.sum()), total + covered.numel()
    return {
        "clips": len(files), "frames": frames, "stride": args.stride, "plant": MAIN.name,
        "fk_pos_err_m": percentiles(pos_err), "fk_rot_err_deg": percentiles(rot_err),
        "goal_coverage": {**per_arm, "both": hits / total, "pos_tol_m": POS_TOL_M, "rot_tol_deg": ROT_TOL_DEG,
                          "table": table_path.name, "table_poses": total},
        "goal_vs_tool_frame": {"rotation_deg_max": float(rot.max()), "grasp_offset_m": offset},
    }


def main() -> None:
    args = build_parser().parse_args()
    from isaaclab.app import AppLauncher

    sys.argv = sys.argv[:1]  # Kit reads sys.argv
    AppLauncher(headless=True)
    try:
        out = replay(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "wrist_frames.json").write_text(json.dumps(out, indent=1))
        print("[wrist_frames] " + json.dumps(out, indent=1), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # app.close() has hung in this environment


if __name__ == "__main__":
    main()
