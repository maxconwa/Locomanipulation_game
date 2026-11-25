"""LocoManip-Marl-Torso-v0: the cooperative task with the arms agent driving the torso joint too, as GOLEM's controller
does (the torso's action first of the arms'). Off by default, so the existing tasks keep their cfgs."""

import pytest

TASK = "LocoManip-Marl-Torso-v0"


@pytest.mark.sim
def test_torso_off_by_default(sim_app):
    from locomanipulation_game.tasks.locomanip_marl.marl_direct_env_cfg import MarlDirectEnvCfg

    cfg = MarlDirectEnvCfg()
    assert cfg.arms_drive_torso is False
    assert not hasattr(cfg.actions, "torso_pos")
    assert cfg.agent_action_terms == {"legs": ["joint_pos"], "arms": ["arm_pos"]}
    assert not hasattr(cfg.observations.arms, "torso_action")
    assert not hasattr(cfg.observations.critic, "torso_action")
    assert not hasattr(cfg.rewards.arms, "torso_action_rate")


@pytest.mark.sim
def test_torso_cfg_is_stable_under_sync_derived(sim_app):
    """train.py and the env call sync_derived again: the torso term stays one term, between the legs' and the arms'."""
    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import TORSO_ACTION, TORSO_ACTION_SCALE
    from locomanipulation_game.tasks.locomanip_marl.marl_torso_env_cfg import MarlTorsoEnvCfg

    cfg = MarlTorsoEnvCfg()
    cfg.sync_derived()
    cfg.sync_derived()
    assert cfg.arms_drive_torso is True
    assert [k for k in vars(cfg.actions) if not k.startswith("_")] == ["joint_pos", TORSO_ACTION, "arm_pos"]
    torso = getattr(cfg.actions, TORSO_ACTION)
    assert torso.joint_names == ["torso_joint"] and torso.scale == TORSO_ACTION_SCALE and torso.use_default_offset
    assert torso.joint_limits is cfg.actions.joint_pos.joint_limits
    assert cfg.agent_action_terms == {"legs": ["joint_pos"], "arms": [TORSO_ACTION, "arm_pos"]}
    assert cfg.observations.arms.torso_action.params == {"action_name": TORSO_ACTION}
    assert cfg.observations.critic.torso_action.params == {"action_name": TORSO_ACTION}
    rate = cfg.rewards.arms.torso_action_rate
    assert rate.params == {"action_name": TORSO_ACTION} and rate.weight == cfg.rewards.arms.action_rate.weight


@pytest.mark.sim
def test_torso_task_arms_action_is_15_with_torso_first(sim_app):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from locomanipulation_game.tasks.locomanip_marl.golem_safety import GOLEM_TARGET_CLIP
    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import TORSO_ACTION, TORSO_ACTION_SCALE

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = 2
    cfg.action_delay_range = None  # the target reaches the robot on the step it is set
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        assert env.cfg.action_spaces == {"legs": 12, "arms": 15}
        assert list(env.action_manager.active_terms) == ["joint_pos", TORSO_ACTION, "arm_pos"]
        robot = env.scene["robot"]
        torso_id = robot.find_joints(["torso_joint"])[0][0]
        arms = torch.zeros(2, 15, device=env.device)
        arms[:, 0] = 1.0
        env.step({"legs": torch.zeros(2, 12, device=env.device), "arms": arms})
        default = robot.data.default_joint_pos[:, torso_id]
        target = env.action_manager.get_term(TORSO_ACTION).processed_actions[:, 0]
        assert torch.allclose(target, default + TORSO_ACTION_SCALE, atol=GOLEM_TARGET_CLIP), (target, default)
        assert torch.allclose(robot.data.joint_pos_target[:, torso_id], target)
    finally:
        env.close()
