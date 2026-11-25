"""Main's rewards and actor observation on world-model state (``marl/main_rewards.py``, ``marl/main_obs.py``).

The CPU cases run on a stand-in of LocoManip-Marl-Pulse-v0's resolved cfg (the term names, functions, weights, stds
and noise it has after ``sync_derived``). The ``sim`` ones compare against the live Pulse env, term by term.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pretraining.constants import DEFAULT_JOINT_POS, ESTOP_LIMITS, JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.marl.config import GRASP_OFFSET, EstopPenaltyCfg
from pretraining.marl.main_obs import MainActorObs
from pretraining.marl.main_rewards import (
    WRISTS,
    MainRewardCfg,
    WMState,
    cycle_init,
    cycle_push,
    main_rewards,
    wrist_poses_in_pelvis,
)
from pretraining.math import quat_apply, quat_from_euler_xyz

REPO = Path(__file__).resolve().parents[2]
STAGE1_RUN = REPO / "data" / "pretraining" / "runs" / "isaac_s1_golem_aware"


# -- a stand-in of the Pulse env's resolved cfg ------------------------------------------------------------------------


def _fn(name):
    def f(*args, **kwargs):
        raise AssertionError("stand-in")

    f.__name__ = name
    return f


def _term(func, weight=1.0, noise=None, **params):
    return SimpleNamespace(func=_fn(func), weight=weight, params=params, noise=noise)


def _uniform(x):
    return SimpleNamespace(n_min=-x, n_max=x)


def _joints():
    return SimpleNamespace(joint_names=list(JOINT_NAMES))


def _arm_terms(scale: float) -> dict:
    terms = {}
    for arm, side in enumerate(("left", "right")):
        terms[f"{side}_ee_pos"] = _term("arm_target_pos_exp", scale, arm=arm, std=0.25)
        terms[f"{side}_ee_pos_fine"] = _term("arm_target_pos_exp_annealed", 0.0, arm=arm)
        terms[f"{side}_ee_quat"] = _term("arm_target_quat_exp_annealed", 0.0, arm=arm)
        for k, std in (("k20", 0.02), ("k5", 0.005), ("k2", 0.002)):
            terms[f"{side}_ee_pos_{k}"] = _term("arm_target_pos_exp", scale, arm=arm, std=std)
        for k, std in (("k200", 0.2), ("k50", 0.05)):
            terms[f"{side}_ee_quat_{k}"] = _term("arm_target_quat_exp", scale, arm=arm, std=std)
        terms[f"{side}_ee_pos_linear"] = _term("arm_target_pos_capped", -20.0 * scale, arm=arm, cap=0.02)
    terms["goal_reached"] = _term("arm_goal_reached", scale * 5.0 / 0.02)
    return terms


def pulse_cfg_standin():
    legs_rew = {"track_lin_vel_xy": _term("track_velocity_cycle_exp", 2.0, component="lin_xy"),
                "track_ang_vel_z": _term("track_velocity_cycle_exp", 1.5, component="ang_z"),
                "torques": _term("joint_torques_l2", -1e-5),
                "orientation": _term("flat_orientation_l2", 0.0),
                "ankle_action_rate": None,
                **{f"arms_{k}": v for k, v in _arm_terms(0.5).items()}}
    arms_rew = {**_arm_terms(1.0), "hold_still": _term("arm_hold_still", 2.0),
                "legs_track_lin_vel_xy": _term("track_velocity_cycle_exp", 1.0, component="lin_xy"),
                "legs_track_ang_vel_z": _term("track_velocity_cycle_exp", 0.75, component="ang_z")}
    speed = SimpleNamespace(floor=[0.01] * 27, per_speed=0.6)
    legs_obs = SimpleNamespace(
        base_ang_vel=_term("lowstate_gyro"), projected_gravity=_term("lowstate_projected_gravity"),
        velocity_commands=_term("generated_commands", command_name="base_velocity"), arm_goal=None, ee_targets=None,
        joint_pos=_term("joint_pos_rel_measured", noise=_uniform(0.003), asset_cfg=_joints()),
        joint_vel=_term("joint_vel_rel", noise=speed, asset_cfg=_joints()),
        gait_phase=_term("gait_phase_sin"), gait_phase_cos=_term("gait_phase_cos"),
        actions=_term("applied_action", action_name="legs_latent"), enable_corruption=True)
    wrists = SimpleNamespace(body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"])
    arms_obs = SimpleNamespace(
        base_ang_vel=None, projected_gravity=None, velocity_commands=None, arm_goal=None,
        ee_targets=_term("arm_targets_in_root_xyzw"),
        joint_pos=_term("joint_pos_rel_measured", noise=_uniform(0.003), asset_cfg=_joints()),
        joint_vel=_term("joint_vel_rel", noise=speed, asset_cfg=_joints()),
        gait_phase=None, gait_phase_cos=None, actions=_term("applied_action", action_name="arms_latent"),
        wrist_poses=_term("body_pose_in_root_xyzw", asset_cfg=wrists),
        wrist_errors=_term("arm_target_error_in_root", noise=_uniform(0.005)), enable_corruption=True)
    imu = SimpleNamespace(gyro_bias=(0.005, 0.005, 0.02), gyro_noise=0.03, quat_noise_deg=0.05,
                          quat_bias_deg={"roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)})
    return SimpleNamespace(
        possible_agents=["legs", "arms"], decimation=10, sim=SimpleNamespace(dt=0.002), gait_period=0.7,
        zero_cmd_eps=0.03, gyro_lowpass_hz=8.0, lowstate_imu=imu,
        encoder_offset_rad={r".*_ankle_pitch_joint": 0.04, r".*_(hip_.*|knee|ankle_roll)_joint": 0.02},
        scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace(
            joint_pos={**DEFAULT_JOINT_POS, "[lr]g_.*_hinge_.*": 0.0}))),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(lin_vel_stds=(0.5, 0.4, 0.3), ang_vel_stds=(1.5, 1.0, 0.7)),
            arm_targets=SimpleNamespace(pelvis_fixed_goals=True, fine_pos_stds=(0.1, 0.07), rot_stds=(0.5, 0.4))),
        rewards=SimpleNamespace(legs=SimpleNamespace(**legs_rew), arms=SimpleNamespace(**arms_rew)),
        observations=SimpleNamespace(legs=legs_obs, arms=arms_obs))


# -- states --------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fk():
    return ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), TRACKED_BODIES, "cpu")


DEFAULT = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES])


def _state(fk, n=4, seed=0, requires_grad=False, spread=0.1):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    root_pos = torch.tensor([0.0, 0.0, 0.99]) + 0.01 * r(n, 3)
    root_quat = quat_from_euler_xyz(0.05 * r(n), 0.05 * r(n), r(n))
    leaves = [root_pos, root_quat, 0.3 * r(n, 3), 0.3 * r(n, 3), DEFAULT + spread * r(n, 27), 0.5 * r(n, 27)]
    if requires_grad:
        leaves = [x.clone().requires_grad_(True) for x in leaves]
    return WMState.from_root(fk, *leaves), leaves


def _commands(n=4, seed=1):
    gen = torch.Generator().manual_seed(seed)
    arm = torch.cat([torch.tensor([0.3, 0.2, 0.2]) + 0.05 * torch.randn(n, 2, 3, generator=gen),
                     quat_from_euler_xyz(*(0.5 * torch.randn(3, n * 2, generator=gen))).view(n, 2, 4)], -1)
    return {"velocity": 0.3 * torch.randn(n, 3, generator=gen), "arm": arm}


def _inst(state):
    """A one-sample cycle mean: the state's own (vx, vy, wz)."""
    return cycle_push(cycle_init(state.joint_pos.shape[0], 35), state)[1]


# -- rewards -------------------------------------------------------------------------------------------------------


def test_reward_cfg_reads_main_terms():
    cfg = MainRewardCfg.from_env_cfg(pulse_cfg_standin(), lin_stage=1, ang_stage=2, arm_stage=1)
    assert cfg.dt == pytest.approx(0.02)
    terms = {(t.agent, t.name): t for t in cfg.terms}
    assert terms["legs", "track_lin_vel_xy"].weight == 2.0 and terms["legs", "track_lin_vel_xy"].std == 0.4
    assert terms["legs", "track_ang_vel_z"].std == 0.7
    assert terms["arms", "legs_track_lin_vel_xy"].weight == 1.0
    assert terms["arms", "left_ee_pos_k5"].std == 0.005 and terms["arms", "right_ee_pos_k5"].arm == 1
    assert terms["legs", "arms_left_ee_pos_k20"].weight == 0.5
    linear = terms["arms", "left_ee_pos_linear"]
    assert linear.kind == "ee_pos_capped" and linear.cap == 0.02
    assert terms["arms", "goal_reached"].kind == "goal_reached"
    # weight 0 and None terms are skipped; nonzero ones main has and this does not model are listed
    assert ("arms", "left_ee_pos_fine") not in terms and ("legs", "orientation") not in terms
    assert set(cfg.left_out) == {"legs/torques", "arms/hold_still"}
    # an annealed term takes the arm command's stage
    standin = pulse_cfg_standin()
    standin.rewards.arms.left_ee_quat.weight = 1.0
    t = {(t.agent, t.name): t for t in MainRewardCfg.from_env_cfg(standin, arm_stage=1).terms}["arms", "left_ee_quat"]
    assert t.std == 0.4 and t.kind == "ee_quat"
    with pytest.raises(ValueError, match="cycle_mean"):
        main_rewards(None, SimpleNamespace(joint_pos=torch.zeros(1, 27)), _commands(1), cfg, cycle_mean=None)
    standin.commands.arm_targets.pelvis_fixed_goals = False
    with pytest.raises(ValueError, match="pelvis-fixed"):
        MainRewardCfg.from_env_cfg(standin)


def test_wrist_terms_are_on_main_wrist_frame_without_grasp_offset(fk):
    """Review Focus 3: main's goals are *_wrist_yaw_link poses (locomanip_marl_env_cfg.py:67-68); stage 1's tool frame
    (GRASP_OFFSET) is not."""
    assert [TRACKED_BODIES[i] for i in WRISTS] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    state, _ = _state(fk)
    cfg = MainRewardCfg.from_env_cfg(pulse_cfg_standin())
    pos, quat = wrist_poses_in_pelvis(state)
    # the wrist_yaw_link frames from the FK's own world poses, into the pelvis frame
    rel = state.body_pos[:, list(WRISTS)] - state.root_pos[:, None]
    from pretraining.math import quat_apply_inverse
    assert torch.allclose(pos, quat_apply_inverse(state.root_quat[:, None].expand(-1, 2, 4), rel), atol=1e-6)
    on_wrist = {"velocity": torch.zeros(4, 3), "arm": torch.cat([pos, quat], -1)}
    out = main_rewards(state, state, on_wrist, cfg, cycle_mean=_inst(state))
    for side in ("left", "right"):
        assert torch.allclose(out[f"arms/{side}_ee_pos_k2"], torch.ones(4), atol=1e-4)
        assert torch.allclose(out[f"arms/{side}_ee_quat_k50"], torch.ones(4), atol=1e-4)
        assert torch.allclose(out[f"arms/{side}_ee_pos_linear"], torch.zeros(4), atol=1e-5)
    # a goal at stage 1's tool frame sits |GRASP_OFFSET| = 0.232 m off main's wrist frame: the capped term saturates
    offset = torch.tensor(GRASP_OFFSET)
    tool = pos + quat_apply(quat, offset.expand(4, 2, 3))
    out = main_rewards(state, state, {"velocity": torch.zeros(4, 3), "arm": torch.cat([tool, quat], -1)}, cfg,
                       cycle_mean=_inst(state))
    assert torch.allclose(out["arms/left_ee_pos_linear"], torch.full((4,), 0.02))
    assert torch.allclose(out["arms/left_ee_pos"], torch.full((4,), math.exp(-offset.norm().item() ** 2 / 0.25**2)),
                          atol=1e-4)


def test_velocity_terms_on_the_gait_cycle_mean(fk):
    cfg = MainRewardCfg.from_env_cfg(pulse_cfg_standin())
    state, _ = _state(fk)
    cmd = {"velocity": torch.cat([state.lin_vel_b[:, :2], state.ang_vel_b[:, 2:]], -1), "arm": _commands()["arm"]}
    out = main_rewards(state, state, cmd, cfg, cycle_mean=_inst(state))
    assert torch.allclose(out["legs/track_lin_vel_xy"], torch.ones(4))
    assert torch.allclose(out["arms/legs_track_ang_vel_z"], torch.ones(4))
    # the window: a cycle of 35 steps at 50 Hz; the mean over the samples so far
    cycle = cycle_init(4, 35)
    other, _ = _state(fk, seed=7)
    cycle, mean = cycle_push(cycle, state)
    cycle, mean = cycle_push(cycle, other)
    want = 0.5 * (torch.cat([state.lin_vel_b[:, :2], state.ang_vel_b[:, 2:]], -1)
                  + torch.cat([other.lin_vel_b[:, :2], other.ang_vel_b[:, 2:]], -1))
    assert torch.allclose(mean, want)
    out = main_rewards(state, other, cmd, cfg, cycle_mean=mean)
    err = (cmd["velocity"][:, :2] - want[:, :2]).square().sum(-1)
    assert torch.allclose(out["legs/track_lin_vel_xy"], torch.exp(-err / 0.5**2))
    # totals: dt x (weights x kernels + e-stop)
    total = sum(t.weight * out[f"legs/{t.name}"] for t in cfg.terms if t.agent == "legs") + out["legs/estop"]
    assert torch.allclose(out["legs"], cfg.dt * total)


def test_estop_penalty_on_a_joint_past_its_limit(fk):
    cfg = MainRewardCfg.from_env_cfg(pulse_cfg_standin(), estop=EstopPenaltyCfg(position=-20.0))
    state, leaves = _state(fk, spread=0.0)
    q = leaves[4].clone()
    q[:, 3] = ESTOP_LIMITS[JOINT_NAMES[3]].q_high + 0.01  # a knee past its e-stop range: the legs' penalty
    past = WMState.from_root(fk, leaves[0], leaves[1], leaves[2], leaves[3], q, torch.zeros(4, 27))
    out = main_rewards(state, past, _commands(), cfg, cycle_mean=_inst(past))
    assert (out["legs/estop"] < 0).all() and torch.allclose(out["arms/estop"], torch.zeros(4))


def test_main_rewards_are_differentiable(fk):
    cfg = MainRewardCfg.from_env_cfg(pulse_cfg_standin(), estop=EstopPenaltyCfg(position=-20.0))
    state, _ = _state(fk, seed=3)
    nxt, leaves = _state(fk, seed=4, requires_grad=True)
    target = (nxt.joint_pos + 0.05).detach()
    out = main_rewards(state, nxt, _commands(), cfg, cycle_mean=_inst(nxt), target=target)
    (out["legs"] + out["arms"]).sum().backward()
    grads = [x.grad for x in leaves]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    # the wrist terms reach the arm joints, the velocity terms the root twist
    assert leaves[4].grad[:, 13:].abs().sum() > 0 and leaves[2].grad.abs().sum() > 0


# -- observations -----------------------------------------------------------------------------------------------------


def test_main_obs_widths_order_and_noiseless_terms(fk):
    obs = MainActorObs(pulse_cfg_standin(), "cpu", corrupt=False)
    assert obs.widths() == {"legs": 3 + 3 + 3 + 27 + 27 + 2 + 2 + 32, "arms": 14 + 27 + 27 + 32 + 14 + 12}
    obs.imu.gyro_noise, obs.imu.quat_noise_deg = 0.0, 0.0
    obs.reset(4, torch.Generator().manual_seed(0))
    obs.zero_randomisation()
    state, _ = _state(fk)
    commands = _commands()
    latent = {"legs": torch.randn(4, 32), "arms": torch.randn(4, 32)}
    t = torch.full((4,), 0.35)
    out = obs(state, commands, latent, t)
    legs, arms = out["legs"], out["arms"]
    assert legs.shape == (4, 99) and arms.shape == (4, 126)
    from pretraining.math import quat_apply_inverse
    torso = TRACKED_BODIES.index("torso_link")
    q_torso = state.body_quat[:, torso]
    # the first gyro sample starts the low-pass at its input
    assert torch.allclose(legs[:, 0:3], quat_apply_inverse(q_torso, state.body_ang_vel[:, torso]), atol=1e-6)
    assert torch.allclose(legs[:, 3:6], quat_apply_inverse(q_torso, torch.tensor([0.0, 0, -1]).expand(4, 3)), atol=1e-6)
    assert torch.equal(legs[:, 6:9], commands["velocity"])
    assert torch.allclose(legs[:, 9:36], state.joint_pos - DEFAULT, atol=1e-6)
    assert torch.equal(legs[:, 36:63], state.joint_vel)
    # half a stride in: the left leg at phase 0.5, the right at 0 (unless standing)
    moving = commands["velocity"].norm(dim=-1) >= 0.03
    assert torch.allclose(legs[moving, 63:65], torch.tensor([0.0, 0.0]).expand(int(moving.sum()), 2), atol=1e-5)
    assert torch.allclose(legs[moving, 65:67], torch.tensor([-1.0, 1.0]).expand(int(moving.sum()), 2), atol=1e-5)
    assert torch.equal(legs[:, 67:], latent["legs"])
    # arms: goals xyzw, joints, latent, wrist poses, errors
    arm = commands["arm"]
    assert torch.allclose(arms[:, :14].view(4, 2, 7)[..., 3:], torch.cat([arm[..., 4:], arm[..., 3:4]], -1))
    assert torch.equal(arms[:, 68:100], latent["arms"])
    pos, quat = wrist_poses_in_pelvis(state)
    assert torch.allclose(arms[:, 100:114].view(4, 2, 7)[..., :3], pos, atol=1e-6)
    assert torch.allclose(arms[:, 114:].view(4, 2, 6)[..., :3], arm[..., :3] - pos, atol=1e-6)
    # zero error when the goal is the wrist pose
    on_goal = obs(state, {**commands, "arm": torch.cat([pos, quat], -1)}, latent, t)["arms"][:, 114:]
    assert torch.allclose(on_goal, torch.zeros_like(on_goal), atol=1e-4)


def test_main_obs_gyro_lowpass_settles_and_noise_and_offsets_drawn(fk):
    obs = MainActorObs(pulse_cfg_standin(), "cpu")
    obs.reset(4, torch.Generator().manual_seed(1))
    # the encoder offsets as main's ranges: ankle pitches +-0.04, the other leg joints +-0.02, the rest none
    assert (obs.encoder_offset[:, [4, 10]].abs() <= 0.04).all() and (obs.encoder_offset[:, 12:] == 0).all()
    assert (obs.encoder_offset[:, [0, 1, 2, 3, 5]].abs() <= 0.02).all() and obs.encoder_offset[:, 4].abs().sum() > 0
    assert (obs.gyro_bias.abs() <= torch.tensor([0.005, 0.005, 0.02])).all()
    state, _ = _state(fk)
    obs.imu.gyro_noise = 0.0
    latent = {"legs": torch.zeros(4, 32), "arms": torch.zeros(4, 32)}
    first = obs(state, _commands(), latent, torch.zeros(4))["legs"][:, :3]
    for _ in range(100):  # 2 s at 50 Hz: an 8 Hz filter has settled on a constant input
        settled = obs(state, _commands(), latent, torch.zeros(4))["legs"][:, :3]
    assert torch.allclose(settled, first, atol=1e-5)
    # the joint positions carry the encoder offsets and the uniform noise
    legs = obs(state, _commands(), latent, torch.zeros(4))["legs"]
    assert ((legs[:, 9:36] - (state.joint_pos - DEFAULT + obs.encoder_offset)).abs() <= 0.003 + 1e-6).all()


# -- against the live env (deferred while the GPU is busy) -------------------------------------------------------------


def _pulse_env(sim_app, noiseless: bool):
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401

    if not (STAGE1_RUN / "prior.pt").exists():
        pytest.skip(f"no stage-1 run at {STAGE1_RUN}")
    task = "LocoManip-Marl-Pulse-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    if noiseless:
        for agent in cfg.possible_agents:
            getattr(cfg.observations, agent).enable_corruption = False
        imu = cfg.lowstate_imu
        imu.gyro_bias, imu.gyro_noise, imu.quat_noise_deg, imu.quat_lag_s = (0.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0)
        imu.quat_bias_deg = {k: (0.0, 0.0) for k in ("roll", "pitch", "yaw")}
        imu.vibration_gain = (0.0, 0.0)
        cfg.encoder_offset_rad = None
        cfg.sync_derived()
    return cfg, gym.make(task, cfg=cfg).unwrapped


def _env_state(env) -> WMState:
    robot = env.scene["robot"]
    d = robot.data
    ids = robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0]
    bodies = robot.find_bodies(list(TRACKED_BODIES), preserve_order=True)[0]
    fields = (d.root_pos_w, d.root_quat_w, d.root_lin_vel_b, d.root_ang_vel_b, d.joint_pos[:, ids],
              d.joint_vel[:, ids], d.body_pos_w[:, bodies], d.body_quat_w[:, bodies], d.body_lin_vel_w[:, bodies],
              d.body_ang_vel_w[:, bodies])
    return WMState(*(x.clone() for x in fields))  # the sim's buffers change in place


def _env_commands(env) -> dict:
    vel = env.command_manager.get_term("base_velocity")
    arm = env.command_manager.get_term("arm_targets")
    return {"velocity": vel.command.clone(), "arm": arm.believed_b.clone(), "arm_goal": arm.arm_mode.float()}


@pytest.mark.sim
def test_main_obs_matches_env_obs_without_noise(sim_app):
    cfg, env = _pulse_env(sim_app, noiseless=True)
    try:
        env.reset()
        obs_model = MainActorObs(cfg, env.device)
        obs_model.reset(env.num_envs, torch.Generator(device=env.device).manual_seed(0))
        obs_model.zero_randomisation()
        assert obs_model.widths() == {a: env.observation_manager.group_obs_dim[a][0] for a in cfg.possible_agents}
        latents = {a: torch.zeros(env.num_envs, 32, device=env.device) for a in cfg.possible_agents}
        for _ in range(20):
            env_obs, *_ = env.step(latents)
            last = {a: env.action_manager.get_term(f"{a}_latent").applied_actions for a in cfg.possible_agents}
            ours = obs_model(_env_state(env), _env_commands(env), last, env.episode_length_buf * env.step_dt)
        for agent, terms in obs_model.groups.items():
            names = env.observation_manager.active_terms[agent]
            dims = env.observation_manager.group_obs_term_dim[agent]
            assert names == [name for name, _, _ in terms], agent
            start = 0
            for name, (width,) in zip(names, dims):
                a, b = env_obs[agent][:, start:start + width], ours[agent][:, start:start + width]
                # the gyro: main samples it at every physics substep, this model once per policy step
                tol = 0.05 if name == "base_ang_vel" else 1e-4
                assert torch.allclose(a, b, atol=tol), f"{agent}/{name}: {(a - b).abs().max().item()}"
                start += width
    finally:
        env.close()


@pytest.mark.sim
def test_main_rewards_match_env_terms(sim_app):
    cfg, env = _pulse_env(sim_app, noiseless=False)
    try:
        env.reset()
        reward_cfg = MainRewardCfg.from_env_cfg(cfg)
        cycle = cycle_init(env.num_envs, 35, env.device)
        latents = {a: torch.zeros(env.num_envs, 32, device=env.device) for a in cfg.possible_agents}
        state = _env_state(env)
        for _ in range(20):
            # the step's rewards read the commands and the reached flag the last observation's command update left
            # (the env updates its commands in _get_observations, after the rewards)
            commands = _env_commands(env)
            just_reached = env.command_manager.get_term("arm_targets").just_reached.clone()
            env.step(latents)
            nxt = _env_state(env)
            cycle, mean = cycle_push(cycle, nxt)
            out = main_rewards(state, nxt, commands, reward_cfg, cycle_mean=mean, just_reached=just_reached)
            state = nxt
        assert (env.episode_length_buf == 20).all(), "an env reset during the comparison"
        for t in reward_cfg.terms:
            manager = env.reward_managers[t.agent]
            value = manager._step_reward[:, manager._term_names.index(t.name)] / t.weight
            assert torch.allclose(value, out[f"{t.agent}/{t.name}"], atol=1e-5), (
                f"{t.agent}/{t.name}: {(value - out[f'{t.agent}/{t.name}']).abs().max().item()}")
    finally:
        env.close()


def test_main_obs_refuses_an_action_term_it_does_not_model():
    """LocoManip-Marl-Torso-v0's arms act in joint space with a torso_action term: not a latent."""
    standin = pulse_cfg_standin()
    standin.observations.arms.torso_action = _term("applied_action", action_name="torso_pos")
    with pytest.raises(ValueError, match="arms/torso_action.*torso_pos.*LocoManip-Marl-Torso-v0"):
        MainActorObs(standin, "cpu")
    standin = pulse_cfg_standin()
    standin.observations.arms.actions = _term("applied_action", action_name="arm_pos")
    with pytest.raises(ValueError, match="arm_pos"):
        MainActorObs(standin, "cpu")
