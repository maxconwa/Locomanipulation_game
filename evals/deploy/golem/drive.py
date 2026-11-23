"""The deploy test's driver, in GOLEM's ros container (run.sh starts it): confirm the engage as the operator does,
wait for the policy node's release of the sim's elastic band, then send THE command program (evals/common/program.py,
its moves and stops from "standing free" on) on /cmd_vel at 50 Hz of sim time, and, if the export takes arm goals,
ARMS_ONLY_GOALS standing goals on /marl_deploy_node/arm_goal from its goal table. Every segment is marked in
events.json (with its sim time); GOLEM's stack-test driver records /cmd_vel, /lowcmd's writers and SLAM's streams
(drive.npz). Python 3.10 + rclpy, numpy, yaml (the ros image's).

    python3 drive.py --out /out
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, "/home/code/tests/sim")
sys.path.insert(0, "/home/code/tests/g1")
import test_sim_bringup as bringup_test  # noqa: E402  (GOLEM's test helpers, mounted read-only)
import test_sim_stack as stack_test  # noqa: E402

from evals.common import program  # noqa: E402

POLICY = "/home/code/core_ws/src/lmg_deploy/policies/h12"
PUBLISH_DT = 0.02   # s of sim time between /cmd_vel messages
WALL_LIMIT = 900.0  # s: the program must finish within this wall time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from geometry_msgs.msg import Pose, PoseArray, Twist
    from std_msgs.msg import Float32MultiArray, String
    from std_srvs.srv import Trigger
    settings = yaml.safe_load(open(os.path.join(POLICY, "export.yaml")))
    r = settings.get("command_ranges") or {}
    full = tuple(float(max(abs(v[0]), abs(v[1]))) for v in (r.get("lin_vel_x", [-0.55, 0.55]),
                                                            r.get("lin_vel_y", [-0.54, 0.54]),
                                                            r.get("ang_vel_z", [-0.97, 0.97])))
    d = stack_test.Driver(a.out)
    d.name = "lmg_evals"
    cmd_pub = d.node.create_publisher(Twist, "/cmd_vel", 10)
    goal_pub = d.node.create_publisher(PoseArray, "/marl_deploy_node/arm_goal", 10)
    end_goal = d.node.create_client(Trigger, "/marl_deploy_node/end_arm_goal")
    d.info["program_full"] = list(full)
    wrist, states = [], []   # (sim t, the node's wrist errors), (sim t, its state on a change)
    d.node.create_subscription(Float32MultiArray, "/marl_deploy_node/wrist_error",
                               lambda m: wrist.append([d._sim()] + list(m.data)), 10)

    def on_state(m):
        if not states or states[-1][1] != m.data:
            states.append((d._sim(), m.data))
            d.mark("node_state", state=m.data)
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    d.node.create_subscription(String, "/marl_deploy_node/state", on_state, QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        d.mark("drive_start")
        if not d.wait_band_released(timeout=300.0, refusals=bringup_test.LAUNCH_REFUSALS, confirm=True,
                                    confirm_service="/marl_deploy_node/confirm_engage"):
            d.mark("not_engaged")
            return 1
        segs = [s for s in program.segments(True, full) if s.kind in ("stand", "move", "stop")]
        t_off = segs[0].start
        end = program.duration(segs)
        start, give_up = d.clock, time.time() + WALL_LIMIT
        current, next_pub = None, 0.0
        while True:
            d.rclpy.spin_once(d.node, timeout_sec=0.005)
            t = d.clock - start + t_off
            if t >= end:
                break
            if time.time() > give_up:
                d.mark("wall_limit")
                return 1
            seg = program.at(segs, t)
            if seg is not current:
                current = seg
                d.mark(f"segment:{seg.name}", kind=seg.kind, vx=seg.cmd[0], vy=seg.cmd[1], wz=seg.cmd[2])
            if d.clock >= next_pub:
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.angular.z = seg.cmd
                cmd_pub.publish(msg)
                next_pub = d.clock + PUBLISH_DT
            d.samples(time.time())
        cmd_pub.publish(Twist())
        d.mark("program_end")
        if settings.get("arm_goals") and settings.get("arm_seed_table"):
            table = np.load(os.path.join(POLICY, os.path.basename(settings["arm_seed_table"])))
            rows = [table[f"poses_{i}"] for i in range(2)]
            rng = np.random.default_rng(0)
            for g in range(program.ARMS_ONLY_GOALS):
                pa = PoseArray()
                pa.header.frame_id = "pelvis"
                for arm in range(2):
                    row = rows[arm][rng.integers(len(rows[arm]))]
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = (float(v) for v in row[:3])
                    p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = (float(v) for v in row[3:7])
                    pa.poses.append(p)
                goal_pub.publish(pa)
                d.mark("arm_goal", index=g)
                d.spin_for(program.GOAL_EVERY_S)
            if end_goal.wait_for_service(timeout_sec=2.0):
                end_goal.call_async(Trigger.Request())
            d.mark("arms_end")
            d.spin_for(2.0)
        d.spin_for(stack_test.SETTLE_S)
        return 0
    finally:
        n = max((len(w) for w in wrist), default=1)
        np.savez(os.path.join(a.out, "lmg_evals.npz"),
                 wrist=np.array([w + [math.nan] * (n - len(w)) for w in wrist], float).reshape(-1, n))
        try:
            d.finish()
        finally:
            d.close()


if __name__ == "__main__":
    sys.exit(main())
