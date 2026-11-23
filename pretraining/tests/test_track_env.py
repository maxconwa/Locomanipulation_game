"""The Isaac Lab tracking env: registration, observation layout, reset, kinematic replay, action clip, motion end."""

import gymnasium as gym
import pytest
import torch

import pretraining.tasks  # noqa: F401  (registers the gym ids; a bare `import pretraining` from the repo root
# finds the outer pretraining/ directory as a namespace package and skips __init__)
from pretraining.constants import DEFAULT_JOINT_POS, JOINT_NAMES, TARGET_LIMITS
from pretraining.supermimic.goal_window import GOAL_REL_ORI_DIM, GOAL_REL_POS_DIM, NUM_GOAL_FRAMES
from standins import write_clips

TASK = "LocoManip-Pretrain-Track-v0"
PLAY_TASK = "LocoManip-Pretrain-Track-Play-v0"
NUM_ENVS = 16
FRAMES = 120
ERROR_METRICS = [
    f"error_{name}" for name in (
        "anchor_pos", "anchor_rot", "anchor_lin_vel", "anchor_ang_vel", "body_pos", "body_rot",
        "body_lin_vel", "body_ang_vel", "joint_pos", "joint_vel")
]


def test_registered_and_listed():
    for task_id, cfg_name in ((TASK, "TrackEnvCfg"), (PLAY_TASK, "TrackEnvCfg_PLAY")):
        assert task_id in gym.registry
        assert "LocoManip-" in task_id  # scripts/list_envs.py lists only these
        spec = gym.registry[task_id]
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] == f"pretraining.tasks.track_env_cfg:{cfg_name}"


@pytest.fixture(scope="module")
def env(sim_app, tmp_path_factory):
    """16 envs on stand-in clips, every reset on frame 0 of the first clip with no perturbation."""
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.utils import load_cfg_from_registry

    sim_utils.create_new_stage()  # a fresh stage: an earlier sim test may have left its prims

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = NUM_ENVS
    motion = cfg.commands.motion
    motion.motion_file = str(tmp_path_factory.mktemp("clips"))
    write_clips(motion.motion_file, n=3, frames=FRAMES)
    motion.sampling_mode = "start"
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    env = gym.make(TASK, cfg=cfg).unwrapped
    yield env
    env.close()


def _ids(env):
    return torch.arange(env.num_envs, device=env.device)


def _joint_ids(env):
    return env.scene["robot"].find_joints(list(JOINT_NAMES), preserve_order=True)[0]


@pytest.mark.sim
def test_observation_shapes(env):
    obs, _ = env.reset()
    assert obs["actor"].shape == (NUM_ENVS, 87)
    assert obs["full_state"].shape == (NUM_ENVS, 3 + 4 + 27 + 3 + 3 + 27)
    assert obs["goal"].shape == (NUM_ENVS, NUM_GOAL_FRAMES * 2 * 27 + GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM)
    assert obs["critic"].shape == (NUM_ENVS, 54 + 3 + 6 + 14 * 3 + 14 * 6 + 3 + 3 + 27 + 27 + 27)


@pytest.mark.sim
def test_reset_puts_robot_on_the_clip_frame(env):
    env.reset()
    robot, cmd = env.scene["robot"], env.command_manager.get_term("motion")
    assert (cmd.time_steps == 0).all()
    motion = cmd.motion
    torch.testing.assert_close(robot.data.joint_pos[:, _joint_ids(env)], motion.joint_pos[0].expand(NUM_ENVS, -1),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(robot.data.root_link_pos_w - env.scene.env_origins,
                               motion.root_pos_w[0].expand(NUM_ENVS, -1), atol=1e-4, rtol=0)
    torch.testing.assert_close(robot.data.root_link_quat_w, motion.root_quat_w[0].expand(NUM_ENVS, -1),
                               atol=1e-4, rtol=0)


@pytest.mark.sim
def test_kinematic_replay_reads_zero_error(env):
    """Each step the clip's state at the command's frame is written to the sim: every tracking error reads zero."""
    env.reset()
    cmd = env.command_manager.get_term("motion")
    actions = torch.zeros(env.num_envs, 27, device=env.device)
    for _ in range(40):
        env.step(actions)
        cmd.reset_to_frame(_ids(env), int(cmd.time_steps[0]))
        cmd.refresh_robot_state()
        cmd.update_relative_body_poses()
        cmd._update_metrics()
        for name in ERROR_METRICS:
            assert cmd.metrics[name].abs().max().item() < 1e-3, name
    log = env.command_manager.reset(_ids(env))
    for name in ERROR_METRICS:
        assert log[f"Metrics/motion/{name}"] < 1e-3, name


@pytest.mark.sim
def test_actor_frame_matches_h12_train_layout(env):
    env.reset()
    robot, cmd = env.scene["robot"], env.command_manager.get_term("motion")
    default = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], device=env.device).expand(NUM_ENVS, -1)
    root = torch.zeros(NUM_ENVS, 13, device=env.device)
    root[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, 1.0], device=env.device)
    root[:, 3] = 1.0
    robot.write_joint_state_to_sim(default, torch.zeros_like(default), joint_ids=_joint_ids(env))
    robot.write_root_link_state_to_sim(root)
    cmd.refresh_robot_state()
    frame = env.observation_manager.compute_group("actor")
    gravity, gyro, q, dq = frame[:, 0:3], frame[:, 3:6], frame[:, 6:33], frame[:, 33:60]
    torch.testing.assert_close(gravity, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(NUM_ENVS, -1),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(gyro, torch.zeros_like(gyro), atol=1e-4, rtol=0)
    # q - default is zero up to each env's encoder bias, drawn once at start-up in (-0.01, 0.01)
    assert env.encoder_bias.shape == (NUM_ENVS, 27)
    assert env.encoder_bias.abs().max() <= 0.01 and env.encoder_bias.abs().max() > 0
    torch.testing.assert_close(q, env.encoder_bias, atol=1e-5, rtol=0)
    torch.testing.assert_close(dq, torch.zeros_like(dq), atol=1e-4, rtol=0)
    action = torch.rand(NUM_ENVS, 27, device=env.device) * 0.2 - 0.1
    obs, *_ = env.step(action)
    torch.testing.assert_close(obs["actor"][:, 60:87], action)


@pytest.mark.sim
def test_targets_are_clipped_to_golem_limits(env):
    env.reset()
    # +1000, not the plan's +100: the smallest action scale (0.03 rad) times 100 stays inside the shoulder yaw range
    env.step(torch.full((NUM_ENVS, 27), 1000.0, device=env.device))
    target = env.action_manager.get_term("joint_pos").processed_actions
    upper = torch.tensor([TARGET_LIMITS[j][1] for j in JOINT_NAMES], device=env.device).expand(NUM_ENVS, -1)
    torch.testing.assert_close(target, upper)


@pytest.mark.sim
def test_motion_end_terminates_and_resets(env):
    env.reset()
    cmd = env.command_manager.get_term("motion")
    cmd.set_time_steps(_ids(env), torch.full((NUM_ENVS,), FRAMES - 1, device=env.device, dtype=torch.long))
    _, _, terminated, truncated, extras = env.step(torch.zeros(NUM_ENVS, 27, device=env.device))
    assert env.termination_manager.get_term("motion_end").all()
    assert truncated.all() and not terminated.any()  # motion end is a time-out
    assert (cmd.time_steps == 1).all()  # reset to frame 0, then advanced by the step's command update
    for name in ERROR_METRICS:
        assert f"Metrics/motion/{name}" in extras["log"]
    robot = env.scene["robot"]
    torch.testing.assert_close(robot.data.joint_pos[:, _joint_ids(env)],
                               cmd.motion.joint_pos[0].expand(NUM_ENVS, -1), atol=1e-4, rtol=0)
