"""Is a plant's explicit PD damping stable at its physics step? The eigenvalues of ``dt * M^-1 * Kd``.

An explicit PD's damping torque -kd qd is computed from the last step's velocity; over one step it removes
``dt M^-1 Kd`` of each velocity mode. At 1.0 a mode is stopped in one step and above it the damping overshoots,
flipping the mode's sign every step (the first PULSE run's coupled arm mode was at 1.28). M is the articulation's
generalized mass matrix at the plant's default pose (floating base: the joint block of its inverse, which keeps the
base's coupling), Kd the plant's damping over the 27 policy joints, times ``--kd-scale`` for the gain draws.

    python -m pretraining.scripts.pd_stability --plant main [--kd-scale 1.2]

Prints the largest eigenvalues with the joints that carry most of each eigenvector; exit 0 if the largest is below
1.0, else 1.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from pretraining.constants import JOINT_NAMES
from pretraining.plants import Plant, get_plant

THRESHOLD = 1.0
_PRIM = "/World/PdStabilityRobot"


def inverse_mass_matrix(plant: Plant) -> np.ndarray:
    """M^-1 over the policy joints (``JOINT_NAMES`` order) at ``plant``'s default pose; needs the Isaac app."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation

    from pretraining.robot import robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=plant.sim_dt, device="cpu", gravity=(0.0, 0.0, 0.0)))
    try:
        robot = Articulation(robot_cfg(plant).replace(prim_path=_PRIM))
        sim.reset()
        joint_pos = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        sim.forward()
        mass = robot.root_physx_view.get_generalized_mass_matrices()[0].double().cpu().numpy()
        n = robot.num_joints
        # a floating base adds its 6 DOFs ahead of the joints; the joint block of M^-1 includes their coupling
        base = mass.shape[0] - n
        if base not in (0, 6):
            raise RuntimeError(f"mass matrix {mass.shape} for {n} joints")
        # PhysX's generalized mass matrix leaves out the joint armature (it reads the same at armature 0.01 and
        # 1.0), which the solver adds to each joint's inertia; add the armature it was given
        armature = robot.root_physx_view.get_dof_armatures()[0].double().cpu().numpy()
        mass[base:, base:] += np.diag(armature)
        minv = np.linalg.inv(mass)[base:, base:]
        ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
        return minv[np.ix_(ids, ids)]
    finally:
        # run it in a process of its own: cleared like this, the app's next simulation still hangs (tests/test_plants)
        sim_utils.delete_prim(_PRIM)
        sim.clear_instance()


def damping_eigenvalues(plant: Plant, minv: np.ndarray, kd_scale: float = 1.0, joints_per_mode: int = 3):
    """[(eigenvalue, [joint, ...]), ...] of ``sim_dt * M^-1 * Kd``, largest first, each with the joints of the
    largest eigenvector entries (weighted by Kd^1/2, so the joints that carry the mode's damping energy)."""
    kd = kd_scale * np.array([plant.kd[j] for j in JOINT_NAMES])
    # dt M^-1 Kd is similar to the symmetric dt Kd^1/2 M^-1 Kd^1/2: real eigenvalues, orthogonal eigenvectors
    root = np.sqrt(kd)
    values, vectors = np.linalg.eigh(plant.sim_dt * root[:, None] * minv * root[None, :])
    out = []
    for k in np.argsort(values)[::-1]:
        top = np.argsort(np.abs(vectors[:, k]))[::-1][:joints_per_mode]
        out.append((float(values[k]), [JOINT_NAMES[i] for i in top]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plant", default="main")
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=6, help="eigenvalues to print")
    args = parser.parse_args()
    plant = get_plant(args.plant)

    from isaaclab.app import AppLauncher

    sys.argv = sys.argv[:1]  # Kit reads sys.argv
    AppLauncher(headless=True)
    modes = damping_eigenvalues(plant, inverse_mass_matrix(plant), args.kd_scale)
    print(f"plant {plant.name}: dt {plant.sim_dt} s, kd x{args.kd_scale:g}; largest eigenvalues of dt M^-1 Kd:")
    for value, joints in modes[:args.top]:
        print(f"  {value:.4f}  {', '.join(joints)}")
    stable = modes[0][0] < THRESHOLD
    print(f"max {modes[0][0]:.4f} {'<' if stable else '>='} {THRESHOLD}: {'stable' if stable else 'UNSTABLE'}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if stable else 1)  # app.close() has hung in this environment


if __name__ == "__main__":
    main()
