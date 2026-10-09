"""GOLEM's arm IK on a standing robot, scored on the fixed goal grid: the split stack's non-learned reach.

    python3 paper/reach/tools/ik_baselines.py [--goals logs/reach/goals/eval_goals_v1.csv] [--workers 8] [--limit N]

It reads the same goals as the policies, in the standing frame (pelvis level, standing_height above the ground).

golem_ik  GOLEM's upper-body IK (core_ws/src/h12_ros2_controller, IKSolver.solve_ik_reduced), imported directly
          without ROS: the 14 arm joints on a pelvis fixed at standing height (legs and torso frozen), pink QP steps
          with the URDF position, velocity and acceleration limits and GOLEM's sphere self-collision barrier
          (d_min 0.02 m), seeds home / t_pose / arms_front_45 / current, 1 s per seed, alpha 0.1: the settings
          UpperController.plan_to_ik_target deploys. The robot cannot crouch, so this is the standing reach of
          the deployed split stack's arm controller.
Success: each wrist within POS_TOL and ROT_TOL of its goal (no hold time: there are no dynamics).
Writes logs/reach/results/golem_ik/kinematic/{trials.csv, meta.json, solutions.npz}. Runs in a Python with pinocchio and
pink (on this workstation the base python3: pinocchio 4.1, pink 4.1).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

GOLEM_IK = Path(os.environ.get("GOLEM_IK", "/home/humanoid/Programs/Humanoid_Simulation/core_ws/src/h12_ros2_controller"))
ROS_ASSETS = C.DEFAULT_ASSETS / "ros_assets"
URDF = ROS_ASSETS / "h1_2_magpie.urdf"
URDF_SPHERE = ROS_ASSETS / "h1_2_magpie_sphere.urdf"
SRDF_SPHERE = ROS_ASSETS / "h1_2_magpie_sphere_collision.srdf"

_W = {}   # per-worker state


def _se3(pose):
    import pinocchio as pin
    x, y, z, qw, qx, qy, qz = pose
    return pin.SE3(pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix(), np.array([x, y, z]))


def _pose_error(m_actual, m_target) -> tuple[float, float]:
    import pinocchio as pin
    return (float(np.linalg.norm(m_actual.translation - m_target.translation)),
            float(np.linalg.norm(pin.log3(m_target.rotation.T @ m_actual.rotation))))


# ---------------------------------------------------------------- golem_ik

def _init_golem():
    sys.path[:0] = [str(GOLEM_IK), str(GOLEM_IK / "submodules" / "unitree_sdk2_python")]
    import pinocchio as pin
    from h12_ros2_controller.core.ik_solver import IKSolver
    from h12_ros2_controller.core.robot_model import RobotModel
    from h12_ros2_controller.utility.joint_definition import ENABLED_JOINTS
    rm = RobotModel(str(URDF))
    rm.init_reduced_model(ENABLED_JOINTS)
    rm.init_collision_model(str(URDF_SPHERE), str(SRDF_SPHERE))
    ik = IKSolver(rm, dt=0.02, d_min=0.02)
    ik.add_frame_task("left", "left_wrist_yaw_link")
    ik.add_frame_task("right", "right_wrist_yaw_link")
    _W.update(pin=pin, rm=rm, ik=ik)


def _solve_golem(goal: dict) -> dict:
    pin, rm, ik = _W["pin"], _W["rm"], _W["ik"]
    targets = {arm: _se3(C.goal_pose(goal, arm)) for arm in C.ARMS}
    ik.frame_tasks["left"].set_target(targets["left"])
    ik.frame_tasks["right"].set_target(targets["right"])
    t0 = time.time()
    # IKSolver.solve_ik_reduced, seed by seed, with its own attempt function and settings. One change: pink raises
    # NoSolutionFound when an integration step leaves a joint a hair past its limit (the wrist pitch at
    # +-0.4625 rad); solve_ik_reduced then aborts every remaining seed. Here a seed that raises is scored by the
    # configuration it had reached, and the next seed runs.
    from pink.exceptions import NoSolutionFound
    attempts, qp_failures = [], 0
    for seed_name, q_seed in ik._seed_list(True, None):
        try:
            att = ik._solve_ik_reduced_attempt(seed_name, q_seed, 0.1, 1.0, 1e-3, 1e-2)
        except NoSolutionFound:
            qp_failures += 1
            q_full = ik._reduced_q_to_full(ik.configuration_reduced.q)
            lin, ang = ik._frame_task_errors(ik.configuration_reduced)
            from h12_ros2_controller.core.ik_solver import IKAttemptResult
            att = IKAttemptResult(q=q_full, success=False, linear_error=lin, angular_error=ang, seed_name=seed_name,
                                  iterations=-1, elapsed_time=0.0)
        attempts.append(att)
        if att.success:
            break
    res = ik._result_from_attempts(attempts, 1e-3, 1e-2)
    model, data = rm.model_body, rm.data_body
    pin.forwardKinematics(model, data, res.q)
    pin.updateFramePlacements(model, data)
    errs = {arm: _pose_error(data.oMf[model.getFrameId(f"{arm}_wrist_yaw_link")], targets[arm]) for arm in C.ARMS}
    # GOLEM's own collision check on the result (sphere model, reduced to the arms)
    q_red = res.q[rm.reduced_mask]
    pin.computeCollisions(rm.model_body_reduced, rm.data_body_reduced, rm.collision_model_body_reduced,
                          rm.collision_data_body_reduced, q_red, False)
    dists = pin.computeDistances(rm.model_body_reduced, rm.data_body_reduced, rm.collision_model_body_reduced,
                                 rm.collision_data_body_reduced, q_red)
    min_dist = min((r.min_distance for r in rm.collision_data_body_reduced.distanceResults), default=math.nan)
    return {"errs": errs, "q": np.asarray(res.q), "seed_name": res.seed_name, "solve_s": time.time() - t0,
            "qp_failures": qp_failures,
            "pelvis_h": C.STANDING_PELVIS_HEIGHT, "min_dist": float(min_dist), "feasible": True,
            "com_margin": math.nan, "pitch_deg": 0.0}


# ---------------------------------------------------------------- driver

def _worker_init(method):
    _init_golem()


def _work(args):
    method, goal = args
    try:
        r = _solve_golem(goal)
    except Exception as e:  # keep the sweep going; the row records the failure
        r = {"errs": {a: (math.nan, math.nan) for a in C.ARMS}, "q": None, "feasible": False, "error": repr(e),
             "pelvis_h": math.nan, "com_margin": math.nan, "pitch_deg": math.nan, "solve_s": math.nan,
             "seed_name": "", "min_dist": math.nan}
    return goal, r


def row_for(method, goal, r) -> dict:
    ok = {a: r["feasible"] and r["errs"][a][0] < C.POS_TOL and r["errs"][a][1] < C.ROT_TOL for a in C.ARMS}
    nan = math.nan
    row = {"label": method, "sim": "kinematic", "kind": method, "lambda_share": nan, "checkpoint": "", "seed": 0,
           "goal_id": goal["goal_id"], "base_id": goal["base_id"], "depth": goal["depth"], "ext": goal["ext"],
           "target_l_x": goal["l_x"], "target_l_y": goal["l_y"], "target_l_z": goal["l_z"],
           "target_r_x": goal["r_x"], "target_r_y": goal["r_y"], "target_r_z": goal["r_z"],
           "target_min_height": goal["min_height"],
           "needs_crouch": min(goal["l_z"], goal["r_z"]) < -0.109,
           "success": ok["left"] and ok["right"], "success_l": ok["left"], "success_r": ok["right"],
           "t_success": nan, "closest": float(np.mean([r["errs"][a][0] for a in C.ARMS])),
           "err_final_l": r["errs"]["left"][0], "err_final_r": r["errs"]["right"][0],
           "rot_final_l": r["errs"]["left"][1], "rot_final_r": r["errs"]["right"][1],
           "err_last1s": float(np.mean([r["errs"][a][0] for a in C.ARMS])),
           "rot_last1s": float(np.mean([r["errs"][a][1] for a in C.ARMS])),
           "pelvis_h_start": C.STANDING_PELVIS_HEIGHT, "pelvis_h_min": r["pelvis_h"], "pelvis_h_last1s": r["pelvis_h"],
           "pelvis_drop_last1s": C.STANDING_PELVIS_HEIGHT - r["pelvis_h"], "fell": False, "t_fall": nan,
           "com_margin_min": r["com_margin"], "com_margin_last1s": r["com_margin"], "pitch_last1s_deg": r["pitch_deg"],
           "feasible": r["feasible"], "solve_s": r["solve_s"], "seed_name": r.get("seed_name", ""),
           "min_dist": r.get("min_dist", nan), "error": r.get("error", ""), "qp_failures": r.get("qp_failures", ""),
           }
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["golem_ik"], default="golem_ik")
    p.add_argument("--goals", default=str(C.GOALS_DIR / "eval_goals_v1.csv"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    goals = C.read_goals(Path(a.goals))
    if a.limit:
        goals = goals[: a.limit]
    for method in [a.method]:
        t0 = time.time()
        out = C.RESULTS_DIR / method / "kinematic"
        rows, qs = [], {}
        ctx = mp.get_context("fork")
        with ctx.Pool(a.workers, initializer=_worker_init, initargs=(method,)) as pool:
            for i, (goal, r) in enumerate(pool.imap(_work, [(method, g) for g in goals], chunksize=4)):
                rows.append(row_for(method, goal, r))
                if r.get("q") is not None:
                    qs[goal["goal_id"]] = np.asarray(r["q"])
                if (i + 1) % 200 == 0 or i + 1 == len(goals):
                    C.write_csv(out / "trials.csv", rows, C.TRIAL_COLUMNS + [
                        "feasible", "solve_s", "seed_name", "min_dist", "error", "qp_failures"])
                    rate = np.mean([x["success"] for x in rows])
                    print(f"[ik_baselines] {method}: {i + 1}/{len(goals)} goals, success {100 * rate:.1f}%,"
                          f" {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out / "solutions.npz", goal_id=np.array(list(qs)), q=np.stack(list(qs.values())))
        C.write_json(out / "meta.json", {
            "method": method, "goals": str(Path(a.goals).resolve()), "goals_sha256": C.sha256(Path(a.goals)),
            "urdf": str(URDF), "urdf_sphere": str(URDF_SPHERE), "srdf_sphere": str(SRDF_SPHERE),
            "golem_ik": {"path": str(GOLEM_IK), **C.git_info(GOLEM_IK)}, "workers": a.workers,
            "wall_s": round(time.time() - t0, 1), "n": len(rows),
        })


if __name__ == "__main__":
    main()
