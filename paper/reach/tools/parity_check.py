"""Does the MuJoCo port (reachlib/mujoco_sim.py) build the same observations and arm targets as the Isaac env?

    python paper/reach/tools/parity_check.py --side isaac  [--out results/_parity/isaac.json]   # env_isaaclab51
    python paper/reach/tools/parity_check.py --side mujoco [--out results/_parity/mujoco.json]  # any MuJoCo python

The Isaac side puts the robot in a fixed, non-trivial state (tilted pelvis, both arms bent away from default, legs
crouched a little), sets a known arm command, turns observation noise off and records both actors' observations
term by term, the odometry frame, and the joint targets the action terms produce for a zero action (the legs' default
targets; the arms' one IK step). The MuJoCo side loads that state into the MJCF and computes the same quantities;
the comparison prints the largest difference per term. The Isaac side takes one env step after writing the state, so its
IMU, Jacobians and action terms describe the same state; the MuJoCo side loads that post-step state with its
velocities. The IMU acceleration is left out: it depends on the step's dynamics, which the two simulators differ in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REACH = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--side", choices=["isaac", "mujoco"], required=True)
p.add_argument("--out", default=None)
p.add_argument("--task", default="LocoManip-Marl-Direct-v0")
a, rest = p.parse_known_args()
OUT = Path(a.out) if a.out else REACH / "results" / "_parity" / f"{a.side}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

ARM_Q = {"left_shoulder_pitch_joint": -0.6, "left_shoulder_roll_joint": 0.3, "left_shoulder_yaw_joint": 0.2,
         "left_elbow_joint": 0.9, "left_wrist_roll_joint": 0.3, "left_wrist_pitch_joint": 0.2,
         "left_wrist_yaw_joint": -0.3, "right_shoulder_pitch_joint": -0.3, "right_shoulder_roll_joint": -0.4,
         "right_shoulder_yaw_joint": -0.1, "right_elbow_joint": 1.2, "right_wrist_roll_joint": -0.2,
         "right_wrist_pitch_joint": -0.1, "right_wrist_yaw_joint": 0.4}
LEG_Q = {"left_hip_pitch_joint": -0.5, "right_hip_pitch_joint": -0.45, "left_knee_joint": 0.9, "right_knee_joint": 0.85,
         "left_ankle_pitch_joint": -0.4, "right_ankle_pitch_joint": -0.38}
ROOT = [0.0, 0.0, 0.92, 0.9950042, 0.0399467, 0.0898501, 0.0]          # pelvis tilted ~5 deg roll, ~10 deg pitch
GOAL_B = [[0.35, 0.25, -0.05, 0.9238795, 0.0, 0.3826834, 0.0], [0.30, -0.22, 0.10, 0.9659258, 0.0, 0.0, 0.258819]]

if a.side == "isaac":
    sys.path.insert(0, str(REACH))
    from reachlib.common import DEFAULT_ASSETS
    os.environ.setdefault("CL_ASSETS_DIR", str(DEFAULT_ASSETS))   # the task reads it when imported
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]] + rest
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    aa = ap.parse_args()
    aa.headless = True
    app = AppLauncher(aa).app
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import locomanipulation_game.tasks  # noqa: F401
    sys.path.insert(0, str(REACH))
    from reachlib import common as C
    cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 2
    cfg.log_dir = str(REACH / "results" / "_parity" / "envdir")
    os.makedirs(cfg.log_dir, exist_ok=True)
    import shutil
    if not os.path.isfile(os.path.join(cfg.log_dir, "arm_target_tables_v3.pt")):
        shutil.copy(C.GOALS_DIR / "arm_target_tables_v3.pt", cfg.log_dir)
    cfg.estimator.train = False
    cfg.events.push_robot = None
    for g in ("legs", "arms", "odometry"):
        getattr(cfg.observations, g).enable_corruption = False
    env = gym.make(a.task, cfg=cfg)
    base = env.unwrapped
    base.reset()
    robot = base.scene["robot"]
    names = robot.joint_names
    q = robot.data.default_joint_pos.clone()
    for n, v in {**ARM_Q, **LEG_Q}.items():
        q[:, names.index(n)] = v
    origin = base.scene.env_origins
    root = torch.tensor(ROOT, device=base.device).repeat(2, 1)
    root[:, :3] += origin
    robot.write_root_pose_to_sim(root)
    robot.write_root_velocity_to_sim(torch.zeros(2, 6, device=base.device))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(dt=0.0)
    arm = base.command_manager.get_term("arm_targets")
    vel = base.command_manager.get_term("base_velocity")
    arm.arm_mode[:] = True
    arm.use_estimate[:] = False
    arm.time_left[:] = 100.0
    arm.believed_b[:] = torch.tensor(GOAL_B, device=base.device)
    arm.anchor_w[:] = torch.tensor(GOAL_B, device=base.device)          # unused by the observations
    vel.vel_command_b[:] = 0.0
    vel.time_left[:] = 100.0
    vel.is_standing_env[:] = True
    # one real step, so the IMU, the Jacobians and the action terms all describe the same state
    zero = {"legs": torch.zeros(2, 12, device=base.device), "arms": torch.zeros(2, 14, device=base.device)}
    obs_step, _, _, _, _ = base.step(zero)
    import numpy as np
    obs = {g: obs_step[g][0].tolist() for g in ("legs", "arms")}
    dims = {g: dict(zip(base.observation_manager.active_terms[g],
                        [int(np.prod(d)) for d in base.observation_manager.group_obs_term_dim[g]]))
            for g in ("legs", "arms")}
    applied = {t: base.action_manager.get_term(t).processed_actions[0].tolist() for t in ("joint_pos", "arm_pos")}
    believed = arm.believed_b[0].tolist()
    d = robot.data
    state = {"root": torch.cat([d.root_link_pos_w[0] - origin[0], d.root_link_quat_w[0]]).tolist(),
             "root_vel": torch.cat([d.root_link_lin_vel_w[0], d.root_link_ang_vel_w[0]]).tolist(),
             "joints": dict(zip(names, d.joint_pos[0].tolist())), "joint_vel": dict(zip(names, d.joint_vel[0].tolist()))}
    # the next targets from this state, for a zero action
    base.action_manager.process_action(torch.zeros(2, 26, device=base.device))
    targets = {t: base.action_manager.get_term(t).processed_actions[0].tolist() for t in ("joint_pos", "arm_pos")}
    out = {"obs": obs, "dims": dims, "targets": targets, "applied_before": applied, "believed": believed, **state}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[parity] wrote {OUT}", flush=True)
    sys.stdout.flush()
    os._exit(0)

# ---------------------------------------------------------------- mujoco side
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402
from reachlib.mujoco_sim import ALL_NAMES, MarlMujoco  # noqa: E402
from reachlib.policy import load_actors  # noqa: E402

ref = json.loads((OUT.parent / "isaac.json").read_text())
meta = json.loads((C.GOALS_DIR / "tables_meta.json").read_text())
smoke = REACH / "results" / "_smoke_run" / "checkpoints" / "agent_0.pt"
sim = MarlMujoco(load_actors(str(smoke)), None, rest_pose_b=np.array(meta["rest_pose_b"]))
sim.reset(np.random.default_rng(0))
d = sim.d
d.qpos[:] = 0.0
d.qpos[0:7] = ref["root"]
for n in ALL_NAMES:
    d.qpos[sim.qadr[ALL_NAMES.index(n)]] = ref["joints"][n]
    d.qvel[sim.vadr[ALL_NAMES.index(n)]] = ref["joint_vel"][n]
lin_w, ang_w = np.array(ref["root_vel"][:3]), np.array(ref["root_vel"][3:])
d.qvel[0:3] = lin_w
q = np.array(ref["root"][3:])
from reachlib.mujoco_sim import qconj, qrot
d.qvel[3:6] = qrot(qconj(q), ang_w)                              # MuJoCo's free joint: angular velocity in body frame
mujoco.mj_forward(sim.m, d)
prev = ref["applied_before"]
sim.applied[sim.leg_idx] = prev["joint_pos"]
sim.applied[sim.arm_idx] = prev["arm_pos"]
sim.residual[:] = 0.0
sim.believed = np.array(ref["believed"], dtype=float)
sim.arm_mode = True
sim.NOISE_POLICY = {k: 0.0 for k in sim.NOISE_POLICY}
obs = sim.observations()
sim.process_actions(np.zeros(12), np.zeros(14))
targets = {"joint_pos": sim.applied[sim.leg_idx].tolist(), "arm_pos": sim.applied[sim.arm_idx].tolist()}
report = {}
for g in ("legs", "arms"):
    off = 0
    for term, dim in ref["dims"][g].items():
        x_i = np.array(ref["obs"][g][off:off + dim])
        x_m = obs[g][off:off + dim]
        report[f"{g}.{term}"] = float(np.max(np.abs(x_i - x_m))) if term != "base_lin_acc" else float("nan")
        off += dim
for t in ("joint_pos", "arm_pos"):
    report[f"target.{t}"] = float(np.max(np.abs(np.array(ref["targets"][t]) - np.array(targets[t]))))
for k, v in report.items():
    print(f"[parity] {k:28s} max |isaac - mujoco| = {v:.2e}")
(OUT.parent / "report.json").write_text(json.dumps(report, indent=1))
