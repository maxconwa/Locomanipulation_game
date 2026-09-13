"""Spawn the H1-2 magpie on a ground plane and print config diagnostics.

No RL machinery: this only checks that the articulation loads, that the joint
regexes matched what you intended, and that the base is free.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4000, help="Physics steps to simulate.")
parser.add_argument("--fall-height", type=float, default=0.6, help="Pelvis z below this counts as a fall.")
parser.add_argument("--randomize", action="store_true", help="Randomize the reset state.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0])

ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/ground", ground_cfg)

light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
light_cfg.func("/World/light", light_cfg)

robot = Articulation(H1_2_MAGPIE_CFG.replace(prim_path="/World/Robot"))

# Physics must be running before robot.data is populated.
sim.reset()

print(f"\nfixed base : {robot.is_fixed_base}")
print(f"joints     : {robot.num_joints}")
print(f"bodies     : {robot.num_bodies}")

soft = robot.data.soft_joint_pos_limits[0]
print(f"\n{'joint':28s} {'default':>8s} {'kp':>8s} {'kd':>6s} {'soft lo':>9s} {'soft hi':>9s}")
for i, name in enumerate(robot.joint_names):
    print(
        f"{name:28s} "
        f"{robot.data.default_joint_pos[0, i].item():8.2f} "
        f"{robot.data.joint_stiffness[0, i].item():8.1f} "
        f"{robot.data.joint_damping[0, i].item():6.1f} "
        f"{soft[i, 0].item():9.2f} {soft[i, 1].item():9.2f}"
    )

print("\nbodies:", robot.body_names, "\n")

FOOT_IDX = robot.body_names.index("left_ankle_roll_link")

def reset(env_ids: torch.Tensor) -> None:
    """Restore the default state, optionally perturbed.

    Standalone Articulation, so default_root_state is already in world terms.
    Inside an InteractiveScene you would add scene.env_origins to the position.
    """
    root = robot.data.default_root_state[env_ids].clone()
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    if args.randomize:
        n = len(env_ids)
        root[:, 0:2] += torch.empty(n, 2, device=root.device).uniform_(-0.5, 0.5)
        root[:, 2] += torch.empty(n, device=root.device).uniform_(0.0, 0.05)
        root[:, 7:13] += torch.empty(n, 6, device=root.device).uniform_(-0.5, 0.5)
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.2, 0.2)
        joint_vel += torch.empty_like(joint_vel).uniform_(-0.5, 0.5)
        joint_pos = joint_pos.clamp(soft[:, 0], soft[:, 1])

    robot.write_root_pose_to_sim(root[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.reset(env_ids)


all_ids = torch.arange(robot.num_instances, device=robot.device)
episode_step = torch.zeros(robot.num_instances, dtype=torch.long, device=robot.device)
episode = 0


step = 0
while simulation_app.is_running() and step < args.steps:
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    episode_step += 1

    fell = robot.data.root_pos_w[:, 2] < args.fall_height
    if fell.any():
        done_ids = all_ids[fell]
        for i in done_ids.tolist():
            episode += 1
            survived = episode_step[i].item() * sim.get_physics_dt()
            print(f"episode {episode:3d}  fell after {survived:5.2f} s  (step {step})")
        reset(done_ids)
        episode_step[fell] = 0

    if step % 100 == 0:
        z = robot.data.root_pos_w[0, 2].item()
        foot_z = robot.data.body_pos_w[0, FOOT_IDX, 2].item()
        print(f"step {step:5d}  pelvis z = {z:.3f}  L ankle z = {foot_z:.3f}  (delta = {z - foot_z:.4f})")
    step += 1

simulation_app.close()