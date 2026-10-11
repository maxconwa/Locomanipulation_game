"""Drive a checkpoint by hand in MuJoCo: walk it with the arrow keys, then switch to reaching and move its wrist
targets with the keys or the mouse.

    python mujoco_evals/human_control.py [--checkpoint <run>/checkpoints/agent_<N>.pt] [--physics robocasa|isaac]
        [--odometry learned|true]

Needs a display and a python with mujoco, torch and pyyaml (not the Isaac Lab one). The policy loop is mujoco_sim.py's
port of the env, run in real time at 50 Hz, plus the legs' gait clock, which the reach evaluation never needs (it
only stands): sin(2 pi phase) of each leg, phase = (time since reset mod 0.8 s) / 0.8 s, the right leg half a cycle
behind, both 0 while the command's norm is under 0.1 (legs_rewards.leg_phase).

Two modes, as in training: walking (the velocity command is live, the wrists hold their rest pose) or reaching (zero
velocity, a world-fixed target per wrist; the policies see it in the pelvis frame, moved by the learned pelvis
estimator as on the robot, or by the true motion with --odometry true).

    Enter             switch between walking and reaching
  walking
    Up / Down         forward speed +- 0.1 m/s (training range +-0.7)
    Left / Right      turn left / right, yaw rate +- 0.1 rad/s (+-0.5)
    Page Up / Down    sideways speed, left / right, +- 0.05 m/s (+-0.3)
    End               stop
  reaching (targets start at the rest pose, in the robot's heading frame at the switch)
    Up / Down         targets forward / back 2 cm
    Left / Right      targets left / right 2 cm
    Page Up / Down    targets up / down 2 cm (low targets make it squat)
    Insert            which targets the keys move: both, left, right
    End               targets back to the rest pose
    mouse             double-click a target, then Ctrl + right-drag moves it, Ctrl + left-drag turns it
  always
    Home, Backspace   reset the robot (it also resets by itself 2 s after a fall)
    Ctrl + right-drag on the robot pushes it

The overlay shows the command, the pelvis velocity averaged over one gait cycle, and while reaching each wrist's
error to its target and how far the estimator has moved the command off it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# the repository root in place of this directory, so mujoco_evals' modules are only ever imported as mujoco_evals.*
sys.path[0] = str(Path(__file__).resolve().parents[1])
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from mujoco_evals import common as C  # noqa: E402
from mujoco_evals.eval_mujoco import _estimator_for  # noqa: E402
from mujoco_evals.mujoco_sim import MarlMujoco, apply, load_estimator, mjcf_path, qconj, qmul, relative  # noqa: E402
from mujoco_evals.policy import load_actors  # noqa: E402

DEFAULT_CHECKPOINT = C.REPO / "logs/skrl/locomanip_marl/2026-10-10_12-11-16_kernels_gtodom_200k/checkpoints/agent_196800.pt"

# the task's velocity command ranges (locomanip_marl_env_cfg.MarlCommandsCfg.base_velocity) and the key steps
VX_MAX, VY_MAX, WZ_MAX = 0.7, 0.3, 0.5
VX_STEP, VY_STEP, WZ_STEP = 0.1, 0.05, 0.1
TARGET_STEP = 0.02                 # m per key press
# legs_rewards.py
GAIT_PERIOD, ZERO_CMD_EPS = 0.8, 0.1
GAIT_OBS = slice(81, 83)           # the gait phase in both actors' observations (after the joint velocities)
RESET_AFTER_FALL_S = 2.0

KEY = {"ENTER": 257, "KP_ENTER": 335, "BACKSPACE": 259, "INSERT": 260, "RIGHT": 262, "LEFT": 263, "DOWN": 264,
       "UP": 265, "PAGE_UP": 266, "PAGE_DOWN": 267, "HOME": 268, "END": 269}           # GLFW key codes
TARGET_COLORS = ((0.95, 0.45, 0.1, 0.9), (0.1, 0.55, 0.95, 0.9))                      # left orange, right blue
IDLE_COLOR = (0.6, 0.6, 0.6, 0.35)


def build_model(assets: Path) -> mujoco.MjModel:
    """The CL_Assets MJCF plus one mocap target per wrist (a sphere and its axes, no collision, no mass on the
    robot): mocap bodies add no joints, so the robot's qpos, qvel and body ids are those of the MJCF alone."""
    spec = mujoco.MjSpec.from_file(str(mjcf_path(assets)))
    for side in ("left", "right"):
        body = spec.worldbody.add_body(name=f"{side}_wrist_target", mocap=True, pos=[0.0, 0.0, -1.0])
        def geom(**kw):
            body.add_geom(contype=0, conaffinity=0, group=2, **kw)
        geom(name=f"{side}_target_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=IDLE_COLOR)
        for axis, rgba in enumerate(((0.9, 0.1, 0.1, 0.9), (0.1, 0.8, 0.1, 0.9), (0.1, 0.1, 0.9, 0.9))):
            half = [0.004, 0.004, 0.004]
            half[axis] = 0.04
            pos = [0.0, 0.0, 0.0]
            pos[axis] = 0.04
            geom(name=f"{side}_target_axis{axis}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, rgba=rgba)
    return spec.compile()


class HumanControlSim(MarlMujoco):
    """MarlMujoco with the legs' gait clock in the observations, so it can walk."""

    def reset(self, rng: np.random.Generator):
        self.steps = 0
        super().reset(rng)

    def step(self, obs: dict, legs_blind: bool = False, arms_ik: bool = False):
        self.steps += 1                    # the env's episode_length_buf counts the step before its observations
        return super().step(obs, legs_blind, arms_ik)

    def observations(self) -> dict[str, np.ndarray]:
        obs = super().observations()
        if np.linalg.norm(self.vel_cmd) >= ZERO_CMD_EPS:
            phase = (self.steps * self.POLICY_DT % GAIT_PERIOD) / GAIT_PERIOD
            gait = np.sin(2.0 * math.pi * np.array([phase, (phase + 0.5) % 1.0]))
            for o in obs.values():
                o[GAIT_OBS] = gait
        return obs


class Controller:
    def __init__(self, sim: HumanControlSim, odometry: str):
        self.sim, self.odometry = sim, odometry
        m = sim.m
        self.mocap = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_target")]
                      for s in ("left", "right")]
        self.geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_target") and "ball" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
            for s in ("left", "right")]
        self.keys = collections.deque()
        self.rng = np.random.default_rng(0)
        self.vel_window = collections.deque(maxlen=int(round(GAIT_PERIOD / C.POLICY_DT)))
        self.reset()

    # -- state
    def reset(self):
        self.sim.reset(self.rng)
        self.sim.vel_cmd = np.zeros(3)
        self.reaching = False
        self.which = 0                     # 0 both, 1 left, 2 right
        self.fallen_since = None
        self.vel_window.clear()
        self.obs = self.sim.observations()
        self._place_idle_markers()

    def _heading(self) -> np.ndarray:
        w, x, y, z = self.sim.root()[1]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    def _place_idle_markers(self):
        """While walking the markers sit at the rest pose, following the pelvis."""
        p, q = self.sim.root()
        for arm, mid in enumerate(self.mocap):
            pose = apply(p, q, self.sim.rest_b[arm])
            self.sim.d.mocap_pos[mid], self.sim.d.mocap_quat[mid] = pose[:3], pose[3:]

    def _color(self, active: bool):
        for arm, gs in enumerate(self.geoms):
            for g in gs:
                self.sim.m.geom_rgba[g] = TARGET_COLORS[arm] if active else IDLE_COLOR

    def start_reaching(self):
        sim = self.sim
        sim.vel_cmd = np.zeros(3)
        sim.set_goal(sim.rest_b.copy(), use_estimate=self.odometry == "learned")
        self.goal_heading = self._heading()
        for arm, mid in enumerate(self.mocap):
            sim.d.mocap_pos[mid], sim.d.mocap_quat[mid] = sim.anchor[arm, :3], sim.anchor[arm, 3:]
        self.rest_anchor = sim.anchor.copy()
        self.reaching = True
        self._color(True)

    def stop_reaching(self):
        sim = self.sim
        sim.arm_mode, sim.anchor = False, None
        sim.believed = sim.rest_b.copy()
        self.reaching = False
        self._color(False)

    def _sync_targets(self):
        """Move the command to the markers, which the keys or the mouse may have moved. The policies see the target
        through the pelvis pose the odometry believes in: T_est = anchor o believed^-1 (the same for both wrists),
        so a moved target keeps the odometry's drift and only the target changes."""
        sim = self.sim
        new = np.stack([np.r_[sim.d.mocap_pos[mid], sim.d.mocap_quat[mid]] for mid in self.mocap])
        new[:, 3:] /= np.linalg.norm(new[:, 3:], axis=1, keepdims=True)
        if np.allclose(new, sim.anchor, atol=1e-9):
            return False
        b = sim.believed[0]
        q_est = qmul(sim.anchor[0, 3:], qconj(b[3:]))
        p_est = sim.anchor[0, :3] - apply(np.zeros(3), q_est, np.r_[b[:3], 1.0, 0.0, 0.0, 0.0])[:3]
        sim.anchor = new
        sim.believed = np.stack([relative(p_est, q_est, a) for a in new])
        return True

    # -- keys (called on the viewer's thread: only queue them)
    def on_key(self, key: int):
        self.keys.append(key)

    def _handle(self, key: int):
        sim = self.sim
        if key in (KEY["HOME"], KEY["BACKSPACE"]):
            self.reset()
            return
        if key in (KEY["ENTER"], KEY["KP_ENTER"]):
            self.stop_reaching() if self.reaching else self.start_reaching()
            return
        if not self.reaching:
            v = sim.vel_cmd
            step = {KEY["UP"]: (0, VX_STEP), KEY["DOWN"]: (0, -VX_STEP), KEY["PAGE_UP"]: (1, VY_STEP),
                    KEY["PAGE_DOWN"]: (1, -VY_STEP), KEY["LEFT"]: (2, WZ_STEP), KEY["RIGHT"]: (2, -WZ_STEP)}
            if key in step:
                axis, dv = step[key]
                v[axis] = round(float(np.clip(v[axis] + dv, -(VX_MAX, VY_MAX, WZ_MAX)[axis], (VX_MAX, VY_MAX, WZ_MAX)[axis])), 3)
            elif key == KEY["END"]:
                v[:] = 0.0
            return
        if key == KEY["INSERT"]:
            self.which = (self.which + 1) % 3
            return
        arms = (0, 1) if self.which == 0 else (self.which - 1,)
        if key == KEY["END"]:
            for arm in arms:
                sim.d.mocap_pos[self.mocap[arm]] = self.rest_anchor[arm, :3]
                sim.d.mocap_quat[self.mocap[arm]] = self.rest_anchor[arm, 3:]
            return
        move = {KEY["UP"]: (1, 0, 0), KEY["DOWN"]: (-1, 0, 0), KEY["LEFT"]: (0, 1, 0), KEY["RIGHT"]: (0, -1, 0),
                KEY["PAGE_UP"]: (0, 0, 1), KEY["PAGE_DOWN"]: (0, 0, -1)}
        if key in move:
            d_w = apply(np.zeros(3), self.goal_heading, np.r_[TARGET_STEP * np.array(move[key]), 1.0, 0, 0, 0])[:3]
            for arm in arms:
                sim.d.mocap_pos[self.mocap[arm]] += d_w

    # -- one policy step
    def step(self):
        sim = self.sim
        while self.keys:
            self._handle(self.keys.popleft())
        if self.reaching:
            if self._sync_targets():
                self.obs = sim.observations()
        self.obs = sim.step(self.obs)
        if not self.reaching:
            self._place_idle_markers()
        self.vel_window.append(np.r_[self._lin_vel_b(), sim.d.qvel[5]])
        if sim.fell():
            self.fallen_since = self.fallen_since or time.time()
            if time.time() - self.fallen_since > RESET_AFTER_FALL_S:
                self.reset()
        else:
            self.fallen_since = None

    def _lin_vel_b(self) -> np.ndarray:
        q = self.sim.root()[1]
        return apply(np.zeros(3), qconj(q), np.r_[self.sim.d.qvel[0:3], 1.0, 0, 0, 0])[:2]

    def overlay(self) -> tuple[str, str]:
        sim = self.sim
        avg = np.mean(self.vel_window, axis=0) if self.vel_window else np.zeros(3)
        rows = [("mode", "REACHING" if self.reaching else "walking"),
                ("command vx vy wz", "{:+.2f} {:+.2f} {:+.2f}".format(*sim.vel_cmd)),
                ("pelvis vx vy wz (gait mean)", "{:+.2f} {:+.2f} {:+.2f}".format(*avg)),
                ("pelvis height", f"{sim.d.qpos[2]:.3f} m")]
        if self.reaching:
            pe, re_ = sim.true_errors()
            drift = sim.log_step()["cmd_drift"]
            rows += [("keys move", ("both", "left", "right")[self.which] + " target(s)"),
                     ("left wrist error", f"{100 * pe[0]:.1f} cm  {re_[0]:.2f} rad"),
                     ("right wrist error", f"{100 * pe[1]:.1f} cm  {re_[1]:.2f} rad"),
                     ("odometry drift", f"{100 * drift:.1f} cm ({self.odometry})")]
        if self.fallen_since:
            rows.append(("FELL", "resetting"))
        rows.append(("keys", "Enter mode, arrows/PgUp/PgDn move, End stop, Home reset"))
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--physics", choices=["robocasa", "isaac"], default="robocasa",
                   help="robocasa: the MJCF's joint dynamics, as GOLEM deploys; isaac: the training plant's")
    p.add_argument("--odometry", choices=["learned", "true"], default="learned")
    a = p.parse_args()

    checkpoint = str(Path(a.checkpoint).resolve())
    rest = np.array(json.loads((C.GOALS_DIR / "tables_meta.json").read_text())["rest_pose_b"], dtype=float)
    estimator = _estimator_for(checkpoint) if a.odometry == "learned" else None
    actors = load_actors(checkpoint)
    if "whole" in actors:
        sys.exit("human_control drives the two-agent policy (legs and arms); this checkpoint is the whole-body baseline")
    sim = HumanControlSim(actors, load_estimator(estimator), physics=a.physics, rest_pose_b=rest,
                          model=build_model(C.DEFAULT_ASSETS),
                          target_limits=C.read_target_limits(Path(checkpoint).parent.parent))
    print(f"[human_control] {checkpoint}\n  estimator {estimator}\n  physics {a.physics}, odometry {a.odometry}",
          flush=True)
    ctl = Controller(sim, a.odometry)

    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=ctl.on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.pelvis
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15.0, 150.0
        try:
            run(viewer, ctl)
        except KeyboardInterrupt:
            pass


def run(viewer, ctl: Controller):
    """The policy at 50 Hz in real time until the window closes."""
    next_t = time.perf_counter()
    while viewer.is_running():
        with viewer.lock():
            ctl.step()
            left, right = ctl.overlay()
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
        viewer.sync()
        next_t += C.POLICY_DT
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()  # running behind: don't try to catch up


if __name__ == "__main__":
    main()
