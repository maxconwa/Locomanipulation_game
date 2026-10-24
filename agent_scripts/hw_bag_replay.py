"""Bag replay: a real robot bag's sensor streams through exported policies, offline, as GOLEM's marl_deploy_node fed
them. Do the real streams push the LSTM where simulation never did?

GOLEM lmg_deploy's own MarlController, LivoxImu and LivoxAttitude are imported (read only, from --golem) and stepped
at the times the node actually published (/safety/lowcmd_lower_in), on the latest /lowstate, /livox/imu, /Odometry
and /cmd_vel before each. The engage tick is found from the recorded targets: the pre-pose ramp ends on the default
pose, and the first tick after it whose leg targets leave it is the policy's first step.

Every --policy runs on the same recorded states. Only the one that was deployed (--deployed) saw its own actions
come back as those states; the others are open loop (what they would have read, not what they would have done).
For the deployed one, the replayed leg targets are checked against the recorded ones: near zero means the replay
is the node.

Per policy and actor: the cell state c per unit and step, and the observation as the network normalizes it,
z = (obs - mean) / std (clamped at +-5 inside the network). With --sim_seeds, the same is logged in MuJoCo
(hw_stress_mujoco.StressMujoco, nominal plant) driven by the bag's own velocity commands as the reference range, and
the report counts the real steps outside it.

    python agent_scripts/hw_bag_replay.py --bag <bag dir> --deployed mid360v2 \
        --policy mid360v2=<GOLEM>/core_ws/src/lmg_deploy/policies/mid360v2 [--policy name=<export dir> ...] \
        [--sim_seeds 4] [--out logs/hw_bag_replay/<bag>]

Any python with numpy, torch, pinocchio, pyyaml, mujoco (for --sim_seeds) and rosbags. Run from the repo root.
Writes <out>/summary.md, summary.json and replay.npz.
"""

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
LEGS_GROUPS = [("acc", 3), ("gyro", 3), ("gravity", 3), ("cmd", 3), ("arm_goal", 1), ("ee_targets", 14),
               ("q", 27), ("dq", 27), ("gait", 4), ("last_action", 12)]
ARMS_GROUPS = LEGS_GROUPS[:-1] + [("last_action", 14), ("wrist_pose", 14), ("wrist_error", 12)]
NUM_LEGS = 12

parser = argparse.ArgumentParser()
parser.add_argument("--bag", required=True)
parser.add_argument("--policy", action="append", required=True, help="name=dir: a deploy folder (<name>.yaml + "
                    "export.yaml + legs.pt / arms.pt) or an export_marl.py output (the template is added)")
parser.add_argument("--deployed", default=None, help="the --policy name the bag ran, for the target check")
parser.add_argument("--golem", default=os.path.expanduser("~/GOLEM"))
parser.add_argument("--urdf", default=str(REPO / "third_party/CL_Assets/ros_assets/h1_2_magpie.urdf"))
parser.add_argument("--template", default=str(REPO / "agent_scripts/hw_deploy_template.yaml"))
parser.add_argument("--imu", choices=("livox", "lowstate"), default="livox",
                    help="the deployed policy's IMU. lowstate: LowState.imu_state, as the node did up to GOLEM 8d1eddd "
                         "(2026-10-16 bags); the other policies always read /livox/imu as today's node does")
parser.add_argument("--attitude", choices=("fastlio", "filter"), default="fastlio")
parser.add_argument("--odom_timeout", type=float, default=1.0)
parser.add_argument("--imu_trim_deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW"),
                    help="the deployed node's imu_offset_*_deg (Explicit500 on 10-16: 0.5 -2.5 0)")
parser.add_argument("--steps_offset", type=int, default=-1,
                    help="joined mid-episode: the node's gait-clock step count at the first tick (unknown from the bag); "
                         "-1 picks the one whose replayed leg targets match the recording best over --fit_seconds")
parser.add_argument("--fit_seconds", type=float, default=4.0)
parser.add_argument("--lowstate_lag", type=int, default=1,
                    help="step on the /lowstate this many messages before the latest one the recorder had at the tick (1 fits "
                         "policy_bag_walk best: the node reads LowState a sample ahead of the recorder)")
parser.add_argument("--free_running", action="store_true",
                    help="the deployed policy's legs observe the replay's own last target, not the recorded one")
parser.add_argument("--max_seconds", type=float, default=0.0, help="stop this long after engage (0: the whole bag)")
parser.add_argument("--sim_seeds", type=int, default=0, help="MuJoCo reference rollouts on the bag's commands (0: none)")
parser.add_argument("--out", default=None)
args = parser.parse_args()

sys.path.insert(0, os.path.join(args.golem, "core_ws/src/lmg_deploy"))
from lmg_deploy.controller import ARM_START, NUM_JOINTS, MarlController, RobotState  # noqa: E402
from lmg_deploy.kinematics import ArmKinematics  # noqa: E402
from lmg_deploy.livox_attitude import LivoxAttitude, in_imu_frame  # noqa: E402
from lmg_deploy.livox_imu import LIVOX_IMU_FRAME, LivoxImu, livox_to_imu_rotation  # noqa: E402
from lmg_deploy.networks import load_network  # noqa: E402
from lmg_deploy.rotations import imu_mount_quat, quat_apply, quat_conjugate, quat_mul, xyzw_to_wxyz  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402


def load_cfg(folder: str) -> dict:
    yamls = [p for p in glob.glob(os.path.join(folder, "*.yaml")) if not p.endswith("export.yaml")]
    cfg = yaml.safe_load(open(yamls[0] if yamls else args.template))
    cfg.update(yaml.safe_load(open(os.path.join(folder, "export.yaml"))) or {})
    cfg.setdefault("legs", {}).setdefault("policy", "legs.pt")
    cfg.setdefault("arms", {}).setdefault("policy", "arms.pt")
    return cfg


class Probe:
    """A deploy network that logs its cell state and its normalized input after each call."""

    def __init__(self, path: str, num_obs: int, num_actions: int):
        self.net = load_network(path, num_obs, num_actions)
        module = self.net.module
        self.recurrent = hasattr(module, "hidden_size")
        self.mean = module.mean.numpy().astype(np.float64)
        self.std = module.std.numpy().astype(np.float64)
        self.eps = float(getattr(module, "epsilon", 1e-8))
        self.c, self.z = [], []

    def reset(self):
        self.net.reset()

    def __call__(self, x):
        out = self.net(x)
        self.z.append((x - self.mean) / (self.std + self.eps))
        self.c.append(self.net._c[-1, 0].numpy().copy() if self.recurrent else np.zeros(0))
        return out


class Replayer:
    """One policy's controller, fed what the node would have fed it."""

    def __init__(self, name: str, folder: str):
        self.name = name
        self.cfg = cfg = load_cfg(folder)
        self.imu_frame = (cfg.get("imu") or {}).get("frame", "imu_link")
        self.kin = ArmKinematics(args.urdf, joint_names=cfg["joint_names"], ee_frames=cfg["arms"]["ee_frames"],
                                 arm_joint_names=[cfg["joint_names"][ARM_START:ARM_START + 7],
                                                  cfg["joint_names"][ARM_START + 7:]],
                                 extra_frames=list(dict.fromkeys([cfg["slam_frame"], LIVOX_IMU_FRAME, self.imu_frame])))
        self.kin.update(np.zeros(NUM_JOINTS))
        self.r_imu_livox = livox_to_imu_rotation(self.kin, self.imu_frame)
        self.livox = LivoxImu(self.r_imu_livox, tau_s=0.5,
                              gyro_lowpass_hz=float((cfg.get("imu") or {}).get("gyro_lowpass_hz", 0.0)))
        self.attitude = LivoxAttitude() if args.attitude == "fastlio" else None
        self.probes = {a: Probe(os.path.join(folder, cfg[a]["policy"]), cfg[a]["num_obs"], cfg[a]["num_actions"])
                       for a in ("legs", "arms")}
        self.ctrl = MarlController(cfg, self.kin, self.probes["legs"], self.probes["arms"])
        # --imu lowstate and the trim were the deployed node's settings; the others read today's node (livox, no trim)
        self.deployed = name == args.deployed
        trim = np.radians(args.imu_trim_deg) if self.deployed else np.zeros(3)
        self.mount = imu_mount_quat(*trim) if np.any(trim) else None
        self.slam_time = None
        self.targets, self.attitude_source, self.times = [], [], []
        # a policy at another rate than the node's ticks (the sport-mode runs' 100 Hz) steps on its own clock, on
        # the latest /lowstate (replay)
        self.dt = float(cfg["control_dt"])
        self.own_clock = False
        self.next_t: float | None = None

    def on_imu(self, stamp, gyro, acc):
        self.livox.update(stamp, gyro, acc)
        if self.attitude is not None:
            self.attitude.add_imu(stamp, gyro, acc)

    def on_odom(self, t, stamp, pos, quat):
        self.ctrl.set_slam_pose(pos, quat)
        self.slam_time = t
        if self.attitude is not None:
            self.attitude.add_pose(stamp, quat)

    def robot_state(self, msg, now: float) -> RobotState:
        motors = msg.motor_state
        if args.imu == "lowstate" and self.deployed:
            # the node before 2026-10-16 14:50 (GOLEM 8d1eddd): the torso IMU's LowState.imu_state as it came
            imu = msg.imu_state
            quat, gyro, acc, source = (np.asarray(imu.quaternion, dtype=np.float64),
                                       np.asarray(imu.gyroscope, dtype=np.float64),
                                       np.asarray(imu.accelerometer, dtype=np.float64), "lowstate")
        elif self.livox.ready:
            gyro, acc = self.livox.gyro.copy(), self.livox.acc.copy()
            fresh = self.slam_time is not None and now - self.slam_time <= args.odom_timeout
            if self.attitude is not None and self.attitude.calibrated and fresh:
                quat, source = in_imu_frame(self.attitude.quat(), self.r_imu_livox), "fastlio"
            else:
                quat, source = self.livox.quat(), "filter"
        else:
            quat, gyro, acc, source = np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), "none"
        if self.mount is not None:
            quat = quat_mul(quat, quat_conjugate(self.mount))
            gyro, acc = quat_apply(self.mount, gyro), quat_apply(self.mount, acc)
        self.attitude_source.append(source)
        return RobotState(q=np.array([motors[i].q for i in range(NUM_JOINTS)], dtype=np.float64),
                          dq=np.array([motors[i].dq for i in range(NUM_JOINTS)], dtype=np.float64),
                          tau=np.array([motors[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float64),
                          quat=quat, gyro=gyro, acc=acc)

    def engage(self, msg, now):
        if self.attitude is not None and not self.attitude.calibrated:
            why = self.attitude.calibrate()
            if why is not None:
                print(f"[replay] {self.name}: FAST-LIO attitude not levelled at engage ({why}); filter attitude")
        self.attitude_source.clear()
        self.ctrl.reset(self.robot_state(msg, now))
        self.ctrl.set_velocity_command(np.zeros(3))
        self.attitude_source.clear()

    def step(self, msg, now):
        if self.attitude is not None and not self.attitude.calibrated:
            self.attitude.calibrate()   # joined mid-episode: level at the first still second
        self.targets.append(self.ctrl.step(self.robot_state(msg, now)).q)
        self.times.append(now)


def read_bag(bag: str):
    store = get_typestore(Stores.ROS2_HUMBLE)
    msgs = Path(args.golem) / "core_ws/src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg/msg"
    types = {}
    for p in msgs.glob("*.msg"):
        types.update(get_types_from_msg(p.read_text(), f"unitree_hg/msg/{p.stem}"))
    store.register(types)
    wanted = {"/lowstate", "/livox/imu", "/Odometry", "/cmd_vel", "/safety/lowcmd_lower_in", "/marl_deploy_node/state",
              "/estop/triggered"}
    with AnyReader([Path(bag)], default_typestore=store) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, t, raw in reader.messages(connections=conns):
            yield conn.topic, t * 1e-9, reader.deserialize(raw, conn.msgtype)


def stamp(msg) -> float:
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def replay(players: list[Replayer], steps_offset: int = 0, max_seconds: float = 0.0):
    """Steps every player at the node's publish ticks after engage; a player whose control_dt differs from the
    node's (the deployed policy's, else the first player's) steps on its own clock instead, from the engage tick
    on, at the latest /lowstate. Returns the tick times, the recorded leg targets per tick, the bag's command per
    tick, its events and whether it joined mid-episode."""
    lowstate, ticks, recorded, cmds, events = None, [], [], [], []
    node_dt = next((p.dt for p in players if p.name == args.deployed), players[0].dt)
    for p in players:
        p.own_clock = abs(p.dt - node_dt) > 1e-9
    default = np.asarray(players[0].cfg["default_joint_pos"])[:NUM_LEGS]
    engaged, cmd, last_target, mid_episode, last_estop = False, np.zeros(3), None, False, None
    history = collections.deque(maxlen=args.lowstate_lag + 1)
    for topic, t, msg in read_bag(args.bag):
        if topic == "/lowstate":
            history.append(msg)
            lowstate = history[max(0, len(history) - 1 - args.lowstate_lag)]
            for p in players:
                while engaged and p.own_clock and p.next_t is not None and t >= p.next_t:
                    p.step(lowstate, t)
                    p.next_t += p.dt
        elif topic == "/livox/imu":
            w, a = msg.angular_velocity, msg.linear_acceleration
            for p in players:
                p.on_imu(stamp(msg), np.array([w.x, w.y, w.z]), np.array([a.x, a.y, a.z]))
        elif topic == "/Odometry":
            o, pos = msg.pose.pose.orientation, msg.pose.pose.position
            q = xyzw_to_wxyz(np.array([o.x, o.y, o.z, o.w]))
            for p in players:
                p.on_odom(t, stamp(msg), np.array([pos.x, pos.y, pos.z]), q)
        elif topic == "/cmd_vel":
            cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
            if engaged:
                for p in players:
                    p.ctrl.set_velocity_command(cmd)
        elif topic in ("/marl_deploy_node/state", "/estop/triggered"):
            value = str(getattr(msg, "data", ""))
            if topic == "/estop/triggered" and last_estop == value:
                continue
            if topic == "/estop/triggered":
                last_estop = value
            events.append((t, topic, value))
            if topic == "/marl_deploy_node/state" and str(msg.data).startswith("fault"):
                break   # latched silent: the node published nothing after this
            if not engaged and msg.data in ("nav", "settle_to_arm", "arm_goal", "settle_to_nav") and not ticks \
                    and last_target is None:
                # the node was engaged before the recording began: start at the first tick from a zero LSTM state
                mid_episode = True
        elif topic == "/safety/lowcmd_lower_in" and lowstate is not None:
            target = np.array([msg.motor_cmd[i].q for i in range(NUM_LEGS)])
            if not engaged and mid_episode:
                engaged = True
                events.append((t, "replay", "joined mid-episode: LSTM state starts at zero"))
                for p in players:
                    p.engage(lowstate, t)
                    p.ctrl.set_velocity_command(cmd)
                    p.ctrl.steps = steps_offset
            if max_seconds > 0 and ticks and t - ticks[0] > max_seconds:
                break
            if not engaged:
                # the pre-pose's last ticks sit on the default pose; the first tick off it after that is the policy's
                if last_target is not None and np.allclose(last_target, default, atol=1e-6) \
                        and not np.allclose(target, default, atol=1e-6):
                    engaged = True
                    events.append((t, "replay", "engaged"))
                    for p in players:
                        p.engage(prev_lowstate, prev_t)
                else:
                    last_target, prev_lowstate, prev_t = target, lowstate, t
                    continue
            for p in players:
                if p.own_clock:
                    if p.next_t is None:   # its first step, on the engage tick
                        p.step(lowstate, t)
                        p.next_t = t + p.dt
                    continue
                p.step(lowstate, t)
                if p.name == args.deployed and not args.free_running:
                    # the legs observe the target the node sent (mdp.applied_action), not the replay's: the inputs
                    # stay the robot's even where the replay's action differs
                    p.ctrl.legs_last = (target - p.ctrl.default[:NUM_LEGS]) / p.ctrl.legs_scale
            ticks.append(t)
            recorded.append(target)
            cmds.append(players[0].ctrl.velocity_command())
            last_target = target
    if not engaged:
        raise SystemExit("never engaged: no pre-pose ending on the default pose in /safety/lowcmd_lower_in")
    return np.array(ticks), np.array(recorded), np.array(cmds), events, mid_episode


def sim_reference(folder: str, cmds: np.ndarray, seeds: int, tick_t: np.ndarray):
    """c and z in MuJoCo (nominal plant, Isaac joint dynamics) driven by the bag's own velocity command (cmds at the
    tick times tick_t, s from engage, held between ticks; the sim steps at the policy's own control_dt), from a zero
    LSTM state on its feet. Also returns each seed's fall time (None: stood)."""
    sys.path.insert(0, str(REPO / "agent_scripts"))
    sys.path.insert(0, str(REPO))
    import hw_stress_mujoco as H

    cfg = load_cfg(folder)
    out = {"legs": {"c": [], "z": []}, "arms": {"c": [], "z": []}}

    class SimProbe:
        recurrent = True

        def __init__(self, path):
            self.net = torch.jit.load(path, map_location="cpu").eval()
            self.mean, self.std = self.net.mean.numpy(), self.net.std.numpy()
            n = int(self.net.hidden_size)
            self.h, self.c = torch.zeros(1, 1, n), torch.zeros(1, 1, n)
            self.cs, self.zs = [], []

        def __call__(self, x):
            a, self.h, self.c = self.net(x, self.h, self.c)
            self.zs.append((x[0].numpy() - self.mean) / (self.std + 1e-8))
            self.cs.append(self.c[0, 0].numpy().copy())
            return a

        def reset_state(self):
            self.h.zero_(), self.c.zero_()

        def state_norm(self):
            return float(self.c.abs().mean())

    falls = []
    for seed in range(seeds):
        actors = {a: SimProbe(os.path.join(folder, cfg[a]["policy"])) for a in ("legs", "arms")}
        rng = np.random.default_rng(seed)
        sim = H.StressMujoco(actors, dict(H.NOMINAL), cfg, "isaac", rng)
        sim.reset(rng)
        obs = sim.observations()
        fell = None
        dt = sim.policy_dt
        with torch.no_grad():
            for k in range(int(tick_t[-1] / dt) + 1):
                tick = max(int(np.searchsorted(tick_t, k * dt, side="right")) - 1, 0)
                sim.vel_cmd = np.asarray(cmds[tick], dtype=np.float64)
                obs = sim.step(obs)
                if sim.tilt() > H.FALL_TILT or sim.root()[0][2] < H.FALL_HEIGHT:
                    fell = k * dt
                    break
        falls.append(fell)
        print(f"[replay] sim seed {seed}: {len(actors['legs'].cs)} steps, "
              f"{'fell at %.1f s' % fell if fell is not None else 'stood'}")
        for a, p in actors.items():
            out[a]["c"].append(np.array(p.cs))
            out[a]["z"].append(np.array(p.zs))
    ref = {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in out.items()}
    ref["falls"] = falls
    return ref


def group_of(n: int, groups) -> list[str]:
    names = [g for g, k in groups for _ in range(k)]
    if len(names) == n + 2:
        # no gait_phase_cos (runs before Tune3Track2): the gait phase is each leg's sin alone
        names = [g for g, k in groups for _ in range(2 if g == "gait" else k)]
    return names if len(names) == n else [f"in{i}" for i in range(n)]


def main():
    out = Path(args.out or REPO / "logs/hw_bag_replay" / Path(args.bag).name)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    specs = [spec.split("=", 1) for spec in args.policy]
    offset = max(args.steps_offset, 0)
    if args.steps_offset < 0 and args.deployed:
        # the gait clock's phase at the first tick, if the bag joins mid-episode: the one that replays best
        folder = dict(specs)[args.deployed]
        cfg = load_cfg(folder)
        period = int(round(float(cfg["gait_period"]) / float(cfg["control_dt"])))
        # the clock only shows while a walk is commanded: fit over the first fit_seconds of that
        probe = Replayer(args.deployed, folder)
        ticks, _, cmds, _, mid = replay([probe], 0)
        walking = np.linalg.norm(cmds, axis=1) >= float(cfg["zero_cmd_eps"])
        fits = []
        if mid and walking.any():
            t_walk = ticks[walking] - ticks[0]
            window = t_walk[0] + args.fit_seconds
            keep = walking[: int(np.searchsorted(ticks - ticks[0], window, side="right"))]
            for k in range(period):
                fit = Replayer(args.deployed, folder)
                _, rec, _, _, _ = replay([fit], k, window)
                n = min(len(rec), len(keep))
                err = np.abs(np.array(fit.targets)[:n, :NUM_LEGS] - rec[:n])[keep[:n]]
                fits.append(float(np.median(err)))
        if fits:
            offset = int(np.argmin(fits))
            print(f"[replay] joined mid-episode: gait clock offset {offset} steps (median |diff| over "
                  f"{args.fit_seconds:g} s of commanded walking from {t_walk[0]:.1f} s: {fits[offset]:.2e} rad; "
                  f"the worst offset {max(fits):.2e})")
    players = [Replayer(*spec) for spec in specs]
    ticks, recorded, cmds, events, mid_episode = replay(players, offset, args.max_seconds)
    t = ticks - ticks[0]
    print(f"[replay] {len(ticks)} policy steps over {t[-1]:.1f} s after engage")
    summary = {"bag": os.path.abspath(args.bag), "steps": len(ticks), "seconds": float(t[-1]),
               "events": [(float(e[0] - ticks[0]), e[1], e[2]) for e in events], "policies": {}}
    npz = {"t": t, "recorded_legs_target": recorded, "cmd": cmds}
    md = [f"# Bag replay: {Path(args.bag).name}", "",
          f"{len(ticks)} policy steps ({t[-1]:.1f} s) after engage; attitude {args.attitude}, IMU trim "
          f"{tuple(args.imu_trim_deg)} deg. Open loop for every policy but the deployed one ({args.deployed}).", ""]
    for p in players:
        info = {"attitude": {s: p.attitude_source.count(s) for s in set(p.attitude_source)}}
        targets = np.array(p.targets)
        npz[f"{p.name}_legs_target"] = targets[:, :NUM_LEGS]
        pt = np.array(p.times) - ticks[0]   # this policy's step times (the ticks', unless on its own clock)
        npz[f"{p.name}_t"] = pt
        if p.name == args.deployed:
            err = np.abs(targets[:, :NUM_LEGS] - recorded)
            info["target_check"] = {}
            # from engage, and from 10 s on (a replay joined mid-episode starts from a zero LSTM state)
            for label, keep in (("all", t >= 0), ("after_10s", t >= 10.0)):
                if keep.any():
                    e = err[keep]
                    info["target_check"][label] = {"max": float(e.max()), "median": float(np.median(e)),
                                                   "p99": float(np.percentile(e, 99))}
                    md += [f"**Target check ({p.name}, {label}):** replayed vs recorded leg targets, |diff| median "
                           f"{np.median(e):.2e}, p99 {np.percentile(e, 99):.2e}, max {e.max():.2e} rad.", ""]
        ref = None
        if args.sim_seeds > 0:
            ref = sim_reference(dict(specs)[p.name], cmds, args.sim_seeds, t)
            info["sim_falls"] = ref["falls"]
            md += [f"MuJoCo on the bag's commands, {args.sim_seeds} seeds: "
                   + ", ".join("stood" if f is None else f"fell at {f:.1f} s" for f in ref["falls"]) + ".", ""]
        # a replay joined mid-episode starts from a zero LSTM state: judge c only once it has had 10 s to fill
        judged = pt >= (10.0 if mid_episode else 0.0)
        md += [f"## {p.name}", "", "| actor | max \\|c\\| real | max \\|c\\| sim | steps with a unit outside sim range "
               "| max \\|z\\| real (group) | steps with any \\|z\\| > 5 | max \\|z\\| sim | steps with an input past "
               "sim's |", "|---|---|---|---|---|---|---|---|"]
        for actor, groups in (("legs", LEGS_GROUPS), ("arms", ARMS_GROUPS)):
            probe = p.probes[actor]
            c, z = np.array(probe.c), np.array(probe.z)
            npz[f"{p.name}_{actor}_c"], npz[f"{p.name}_{actor}_z"] = c, z
            names = group_of(z.shape[1], groups)
            zmax = np.abs(z).max(axis=0)
            worst = int(zmax.argmax())
            row = {"max_abs_c": float(np.abs(c).max()) if c.size else None,
                   "mean_abs_c_last_5s": float(np.abs(c[pt >= pt[-1] - 5]).mean()) if c.size else None,
                   "max_abs_z": float(zmax.max()), "max_abs_z_input": [worst, names[worst]],
                   "steps_any_z_over_5": float((np.abs(z) > 5).any(axis=1).mean()),
                   "z_over_5_by_group": {g: float((np.abs(z[:, [i for i, n in enumerate(names) if n == g]]) > 5)
                                                  .any(axis=1).mean()) for g in dict(groups)}}
            sim_c, sim_z, outside, z_past = "-", "-", "-", "-"
            if ref is not None and c.size:
                rc, rz = ref[actor]["c"], ref[actor]["z"]
                lo, hi = rc.min(axis=0), rc.max(axis=0)
                span = np.maximum(hi - lo, 1e-3)
                out_of = ((c < lo - 0.1 * span) | (c > hi + 0.1 * span))[judged]
                # inputs: past the largest |z| the sim produced on that input (by 10%)
                zlim = 1.1 * np.abs(rz).max(axis=0) + 1e-3
                past = np.abs(z) > zlim
                by_group = {g: float(past[:, [i for i, n in enumerate(names) if n == g]].any(axis=1).mean())
                            for g in dict(groups)}
                row.update(sim_max_abs_c=float(np.abs(rc).max()), sim_max_abs_z=float(np.abs(rz).max()),
                           steps_outside_sim_c=float(out_of.any(axis=1).mean()),
                           units_outside_sim_c=np.flatnonzero(out_of.any(axis=0)).tolist(),
                           steps_input_past_sim=float(past.any(axis=1).mean()), input_past_sim_by_group=by_group)
                npz[f"{p.name}_{actor}_sim_c_lo"], npz[f"{p.name}_{actor}_sim_c_hi"] = lo, hi
                npz[f"{p.name}_{actor}_sim_zlim"] = zlim
                sim_c, sim_z = f"{np.abs(rc).max():.2f}", f"{np.abs(rz).max():.1f}"
                outside = f"{100 * row['steps_outside_sim_c']:.1f}% ({len(row['units_outside_sim_c'])} units)"
                top = sorted(by_group.items(), key=lambda kv: -kv[1])[:3]
                z_past = f"{100 * row['steps_input_past_sim']:.1f}% (" + ", ".join(
                    f"{g} {100 * v:.0f}%" for g, v in top if v > 0) + ")"
            info[actor] = row
            md.append(f"| {actor} | {row['max_abs_c']:.2f} | {sim_c} | {outside} | {zmax.max():.1f} ({names[worst]}) "
                      f"| {100 * row['steps_any_z_over_5']:.1f}% | {sim_z} | {z_past} |")
        md.append("")
        summary["policies"][p.name] = info
    md += ["Events (s after engage): " + ", ".join(f"{e[0]:.2f} {e[1]} {e[2]}" for e in summary["events"]), ""]
    np.savez_compressed(out / "replay.npz", **npz)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"[replay] wrote {out}")


if __name__ == "__main__":
    main()
