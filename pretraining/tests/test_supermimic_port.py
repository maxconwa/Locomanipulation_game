"""The SuperMimic port: dynamics, replay buffer, GenCoord representation, goal window, learners, sim context."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from pretraining.constants import DEFAULT_JOINT_POS, JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.motions import MotionLoader
from standins import write_clips

pytestmark = pytest.mark.filterwarnings("ignore:TensorFloat32 tensor cores:UserWarning")  # inductor's TF32 hint

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_JOINTS = len(JOINT_NAMES)
FULL_STATE_DIM = 13 + 2 * NUM_JOINTS
DT = 0.02


@pytest.fixture(scope="module")
def motion(tmp_path_factory):
    fk = ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), TRACKED_BODIES, DEVICE)
    clips = tmp_path_factory.mktemp("clips")
    write_clips(clips, n=2, frames=120)
    return MotionLoader(clips, fk, DEVICE)


def _default_joint_pos():
    return torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], device=DEVICE)


def _representation(actor_obs_mode="rep_state"):
    from pretraining.supermimic.representations import create_representation

    return create_representation(
        "gencoord", FULL_STATE_DIM, NUM_JOINTS, DT, 4, _default_joint_pos(), actor_obs_mode=actor_obs_mode
    ).to(DEVICE)


def _full_state(motion, frames):
    return torch.cat(
        [
            motion.root_pos_w[frames],
            motion.root_quat_w[frames],
            motion.joint_pos[frames],
            motion.root_lin_vel_w[frames],
            motion.root_ang_vel_w[frames],
            motion.joint_vel[frames],
        ],
        dim=-1,
    )


def test_dynamics_forward_shapes():
    from pretraining.supermimic.models import Dynamics

    rep = _representation("proprio")
    dyn = Dynamics(
        rep, NUM_JOINTS, 31, torch.full((1, NUM_JOINTS), 0.25), _default_joint_pos()[None].cpu(), hidden_dims=(64, 32)
    ).to(DEVICE)
    full = torch.randn(5, FULL_STATE_DIM, device=DEVICE)
    full[:, 3:7] = torch.nn.functional.normalize(full[:, 3:7], dim=-1)
    rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
    assert rep_state.shape == (5, 10 + 2 * NUM_JOINTS) and aux.shape == (5, 7)
    next_aux, next_rep = dyn(rep_state, torch.randn(5, NUM_JOINTS, device=DEVICE), aux, torch.zeros(5, 31, device=DEVICE))
    assert next_aux.shape == (5, 7) and next_rep.shape == rep_state.shape
    assert torch.isfinite(next_rep).all()
    torch.testing.assert_close(next_aux[:, 3:7].norm(dim=-1), torch.ones(5, device=DEVICE))
    # the policy contract: 87-wide proprio frame, its noise half-widths
    assert rep.proprio_obs_dim(NUM_JOINTS) == 87
    bounds = rep._proprio_noise_bounds
    assert bounds[:3].eq(0.05).all() and bounds[3:6].eq(0.2).all()
    assert bounds[6:33].eq(0.01).all() and bounds[33:60].eq(0.5).all() and bounds[60:].eq(0.0).all()


def test_replay_buffer_causal_sampling_never_crosses_a_done():
    from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer

    num_envs, size = 4, 16
    buf = ReplayBuffer(
        {
            "episode": FieldSpec(dtype=torch.long),
            "t": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        },
        size,
        num_envs,
        DEVICE,
    )
    gen = torch.Generator().manual_seed(0)
    episode, t = torch.zeros(num_envs, dtype=torch.long), 0
    for step in range(40):  # wraps the ring twice
        done = torch.rand(num_envs, generator=gen) < 0.2
        term = done & (torch.rand(num_envs, generator=gen) < 0.5)
        buf.insert(
            TensorDict(
                {"episode": episode.clone(), "t": torch.full((num_envs,), step), "terminated": term,
                 "truncated": done & ~term},
                batch_size=[num_envs],
            ).to(DEVICE)
        )
        episode = episode + done.long()  # the frame that stores a done is the episode's last
        t += 1
    window, padded = buf.sample_causal(6, 4096)
    ep, ts = window["episode"], window["t"]
    # every real slot is the newest frame's episode and consecutive in time
    newest = ep[:, -1:]
    assert (ep == newest).all(), "a window crossed a done"
    real = ~padded
    assert real[:, -1].all()
    steps = ts[:, 1:] - ts[:, :-1]
    assert (steps[real[:, :-1]] == 1).all()
    # padded slots repeat the first real frame
    first = torch.where(padded, ts.new_full(ts.shape, 10**9), ts).amin(-1, keepdim=True)
    assert (ts[padded] == first.expand_as(ts)[padded]).all()
    assert padded.any() and real[:, 0].any()  # both cases occur: a window that reaches back past a reset, one that doesn't

    seq = buf.sample(5, 4096)
    ended = seq["terminated"] | seq["truncated"]
    assert not ended[:, :-1].any(), "sample() put a done inside a window"
    assert ((seq["t"][:, 1:] - seq["t"][:, :-1]) == 1).all()


def test_gencoord_roundtrip(motion):
    rep = _representation()
    rep.fk = motion.fk
    frames = torch.arange(0, motion.time_step_total, 7, device=DEVICE)
    full = _full_state(motion, frames)
    rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
    torch.testing.assert_close(rep.reconstruct_full_state(aux, rep_state), full, atol=1e-5, rtol=0)
    pos, rot, lin, ang = rep.world_frame_bodies(aux, rep_state)
    torch.testing.assert_close(pos, motion.body_pos_w[frames], atol=1e-5, rtol=0)
    torch.testing.assert_close(lin, motion.body_lin_vel_w[frames], atol=1e-4, rtol=0)
    torch.testing.assert_close(ang, motion.body_ang_vel_w[frames], atol=1e-4, rtol=0)
    from pretraining.math import matrix_from_quat

    torch.testing.assert_close(rot, matrix_from_quat(motion.body_quat_w[frames]), atol=1e-5, rtol=0)
    torch.testing.assert_close(rep.joint_state(rep_state), torch.cat([full[:, 7:34], full[:, 40:]], -1))
    # the world model's reference bodies are the clip's
    for a, b in zip(motion.body_state(frames), (motion.body_pos_w, motion.body_quat_w, motion.body_lin_vel_w,
                                                motion.body_ang_vel_w)):
        torch.testing.assert_close(a, b[frames], atol=1e-5, rtol=0)


def test_goal_window_indices_clamp_to_the_clip():
    from pretraining.supermimic.representations import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES, goal_window_indices

    assert (NUM_GOAL_FRAMES, GOAL_FRAME_STRIDE) == (10, 5)
    clips = SimpleNamespace(motion_ends=torch.tensor([20, 40]))
    idx = goal_window_indices(clips, torch.tensor([0, 7, 19, 20, 33, 39]))
    assert idx.shape == (6, 10)
    assert idx[0].tolist() == [0, 5, 10, 15, 19, 19, 19, 19, 19, 19]
    assert idx[1].tolist() == [7, 12, 17] + [19] * 7
    assert idx[2].tolist() == [19] * 10
    assert idx[3].tolist() == [20, 25, 30, 35] + [39] * 6
    assert idx[4].tolist() == [33, 38] + [39] * 8
    assert idx[5].tolist() == [39] * 10


def _linear_system_batches(rep, num_traj, horizon, gen):
    """Trajectories whose prediction is a fixed linear map of the action, integrated by the representation."""
    torch.manual_seed(0)
    weight = torch.randn(rep.prediction_dim, NUM_JOINTS, generator=gen).to(DEVICE) * 0.05
    full = torch.zeros(num_traj, FULL_STATE_DIM, device=DEVICE)
    full[:, 2] = 1.0
    full[:, 3] = 1.0
    full[:, 7:34] = _default_joint_pos()
    aux, rep_state = rep.extract_aux_state(full), rep.represent(full)
    actions = torch.randn(num_traj, horizon, NUM_JOINTS, generator=gen).to(DEVICE)
    states = [full]
    for k in range(horizon - 1):
        aux, rep_state = rep.apply_prediction(aux, rep_state, actions[:, k] @ weight.T, rep.dt)
        states.append(rep.reconstruct_full_state(aux, rep_state))
    return torch.stack(states, 1), actions


def test_world_model_learner_reduces_loss_on_a_linear_system():
    from pretraining.supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
    from pretraining.supermimic.models import Dynamics

    rep = _representation()
    gen = torch.Generator().manual_seed(1)
    states, actions = _linear_system_batches(rep, 512, 8, gen)
    dyn = Dynamics(rep, NUM_JOINTS, 1, torch.ones(1, NUM_JOINTS), torch.zeros(1, NUM_JOINTS),
                   hidden_dims=(128, 128)).to(DEVICE)
    cfg = WorldModelLearnerCfg(lr=1e-3, num_batches=1, batch_size=128, horizon=8)
    learner = WorldModelLearner(dyn, rep, cfg, device=DEVICE)
    learner.update_normalization(states.flatten(0, 1), actions.flatten(0, 1))
    sim_ctx = torch.zeros(512, 8, 1, device=DEVICE)

    def batch():
        i = torch.randint(512, (cfg.batch_size,), generator=gen).to(DEVICE)
        return TensorDict({"state": states[i], "action": actions[i], "sim_ctx": sim_ctx[i]}, batch_size=[cfg.batch_size, 8])

    losses = [float(learner.update([batch()])["Loss/state"]) for _ in range(200)]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first / 10, (first, last)


class _PdDynamics(torch.nn.Module):
    """A known world model: the joints accelerate toward the commanded target, the root stays put."""

    def __init__(self, rep, kp=20.0):
        super().__init__()
        self.rep, self.kp = rep, kp
        self.register_buffer("offset", _default_joint_pos())

    def forward(self, rep_state, action, aux_state, sim_ctx):
        del sim_ctx
        n = NUM_JOINTS
        qj, vj = rep_state[..., 10 : 10 + n], rep_state[..., 10 + n :]
        target = 0.5 * action + self.offset
        dvj = (self.kp * (target - qj) - 2.0 * vj) * self.rep.dt
        prediction = torch.cat([torch.zeros_like(rep_state[..., :6]), dvj], -1)
        return self.rep.apply_prediction(aux_state, rep_state, prediction, self.rep.dt)


def _stand_in_wm_env(motion, rep, buf, batch_size):
    from pretraining.supermimic.envs.motion_tracking_env import MotionTrackingEnv

    rep.fk = motion.fk
    rep._soft_joint_pos_limits = torch.tensor([[-3.0, 3.0]] * NUM_JOINTS, device=DEVICE)
    term = lambda w: SimpleNamespace(weight=w)  # noqa: E731
    cfg = SimpleNamespace(
        # the tracking env's: anchor pos/ori 0.5, body pos/ori/lin vel/ang vel 1.0, action rate -0.1, a disabled term
        rewards=SimpleNamespace(
            anchor_pos=term(0.5), anchor_ori=term(0.5), body_pos=term(1.0), body_ori=term(1.0), body_lin_vel=term(1.0),
            body_ang_vel=term(1.0), action_rate=term(-0.1), off=None,
        ),
        terminations=SimpleNamespace(
            anchor_pos=SimpleNamespace(params={"threshold": 0.25}),
            anchor_ori=SimpleNamespace(params={"threshold": 0.8}),
        ),
    )
    cmd = SimpleNamespace(motion=motion, motion_anchor_body_index=0)
    unwrapped = SimpleNamespace(step_dt=DT, command_manager=SimpleNamespace(get_term=lambda name: cmd))
    env = SimpleNamespace(unwrapped=unwrapped, cfg=cfg, num_actions=NUM_JOINTS)
    return MotionTrackingEnv(_PdDynamics(rep).to(DEVICE), rep, env, batch_size, buf, 1, 0.05, device=DEVICE)


def test_bptt_policy_learner_improves_return_in_a_known_world_model(motion):
    from pretraining.supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
    from pretraining.supermimic.models import build_mlp
    from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer

    rep = _representation()
    total = motion.time_step_total
    buf = ReplayBuffer(
        {
            "state": FieldSpec(FULL_STATE_DIM),
            "action": FieldSpec(NUM_JOINTS),
            "prev_action": FieldSpec(NUM_JOINTS),
            "motion_time": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
            "sim_ctx": FieldSpec(1),
        },
        buffer_size=total,
        num_envs=1,
        device=DEVICE,
    )
    for f in range(total):
        frame = torch.tensor([f], device=DEVICE)
        buf.insert(
            TensorDict(
                {
                    "state": _full_state(motion, frame),
                    "action": torch.zeros(1, NUM_JOINTS, device=DEVICE),
                    "prev_action": torch.zeros(1, NUM_JOINTS, device=DEVICE),
                    "motion_time": frame,
                    "terminated": torch.zeros(1, dtype=torch.bool, device=DEVICE),
                    "truncated": (frame + 1 == motion.motion_ends).any().view(1),
                    "sim_ctx": torch.zeros(1, 1, device=DEVICE),
                },
                batch_size=[1],
            )
        )
    wm_env = _stand_in_wm_env(motion, rep, buf, batch_size=256)
    obs_dim = wm_env.reset()["actor"].shape[-1]

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = build_mlp(obs_dim, (128, 128), NUM_JOINTS)

        def forward(self, obs):
            return self.mlp(obs["actor"])

    torch.manual_seed(0)
    actor = Actor().to(DEVICE)
    cfg = BpttPolicyLearnerCfg(lr=3e-4, batch_size=256, horizon=16)
    learner = BpttPolicyLearner(actor, wm_env, cfg, device=DEVICE)
    start = wm_env.sample_initial_states().clone()

    @torch.no_grad()
    def mean_reward():
        """Mean per-step reward (dt-free) of the actor's rollout from the same start states."""
        actor.eval()
        obs, total = wm_env.reset(start.clone()), 0.0
        for _ in range(cfg.horizon):
            obs, reward, _, _ = wm_env.step(actor(obs))
            total += float(reward.mean())
        return total / cfg.horizon / DT

    best = wm_env.max_step_reward / DT  # every tracking kernel at 1, no penalty
    before = mean_reward()
    for _ in range(100):
        learner.update()
    after = mean_reward()
    assert best - after < 0.5 * (best - before), (before, after, best)


_FALLBACK_SCRIPT = """
import logging, sys
import torch
from tensordict import TensorDict
from pretraining.supermimic.learners import learner_utils
from pretraining.supermimic.learners.learner_utils import CapturedUpdate

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="%(message)s")
learner_utils.CUDA_GRAPHS = True
weight = torch.zeros(3, device="cuda")
if sys.argv[1] == "rng_intact":  # capture refused before it begins: the RNG is untouched
    def refuse(*args, **kwargs):
        raise RuntimeError("capture refused")
    torch.cuda.graph = refuse

def update(inputs):  # a host sync: legal eagerly, illegal under capture
    weight.add_(inputs["x"])
    return {"sum": torch.tensor(float(weight.sum()), device="cuda")}

captured = CapturedUpdate(update, lambda: [weight], warmup=1)
sums = []
try:
    for _ in range(4):
        sums.append(float(captured(TensorDict({"x": torch.ones(3, device="cuda")}))["sum"]))
except RuntimeError as e:
    print("RAISED", e, "CAUSE", type(e.__cause__).__name__)
print("SUMS", sums, "FLAG", learner_utils.CUDA_GRAPHS)
"""


def _run_fallback(case):
    import subprocess
    import sys
    from pathlib import Path

    out = subprocess.run(
        [sys.executable, "-c", _FALLBACK_SCRIPT, case], capture_output=True, text=True, timeout=300,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


# In their own processes: on torch 2.7 a failed capture leaves the CUDA RNG unusable for the rest of the process.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph capture needs a GPU")
def test_captured_update_falls_back_to_eager_when_capture_fails_and_the_rng_survives():
    out = _run_fallback("rng_intact")
    assert "SUMS [3.0, 6.0, 9.0, 12.0] FLAG False" in out  # every call updated, eagerly after the failure
    assert out.count("CUDA-graph capture failed") == 1 and "RAISED" not in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph capture needs a GPU")
def test_captured_update_fails_fast_when_the_failed_capture_breaks_the_rng():
    out = _run_fallback("rng_broken")
    assert "RAISED CUDA-graph capture failed" in out and "PRETRAINING_CUDA_GRAPHS=0" in out
    assert "CAUSE RuntimeError" in out  # the capture error, chained
    assert "SUMS [3.0] FLAG False" in out  # the warm-up update ran; nothing after the failed capture


# --- simulator ---------------------------------------------------------------------------------------------------

TASK = "LocoManip-Pretrain-Track-v0"
NUM_ENVS = 16


@pytest.fixture(scope="module")
def env(sim_app, tmp_path_factory):
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.utils import load_cfg_from_registry

    import pretraining.tasks  # noqa: F401  (registers the gym ids)

    sim_utils.create_new_stage()
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = NUM_ENVS
    motion_cfg = cfg.commands.motion
    motion_cfg.motion_file = str(tmp_path_factory.mktemp("sim_clips"))
    write_clips(motion_cfg.motion_file, n=3, frames=120)
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    env = gym.make(TASK, cfg=cfg).unwrapped
    yield env
    env.close()


@pytest.mark.sim
def test_sim_context_binds_and_reads_normalised_values(env):
    from pretraining.supermimic.sim_context import SimContext

    ctx = SimContext.from_env(env)
    assert [s.name for s in ctx.specs] == ["foot_friction", "base_com", "encoder_bias"]
    assert ctx.output_dim == 1 + 3 + NUM_JOINTS
    env.reset()
    values = ctx.read(env)
    assert values.shape == (NUM_ENVS, ctx.output_dim)
    assert torch.isfinite(values).all()
    assert values.abs().max() <= 1.0
    assert values.std(0).min() > 0, "a column is constant across envs: it does not read the randomized value"
    # the encoder-bias columns are the event's bias, denormalised
    torch.testing.assert_close(ctx.denormalize(values, ctx.find("encoder_bias")), env.encoder_bias)
    # friction and COM offset are cached at bind: they are PhysX's values
    robot = env.scene["robot"]
    friction = ctx.denormalize(values, ctx.find("foot_friction"))
    assert friction.device == env.encoder_bias.device
    materials = robot.root_physx_view.get_material_properties()
    assert any(torch.equal(friction[:, 0].cpu(), materials[:, k, 0]) for k in range(materials.shape[1]))
    torso = robot.body_names.index("torso_link")
    com_offset = ctx.denormalize(values, ctx.find("base_com")).cpu()
    coms = robot.root_physx_view.get_coms()[:, torso, :3]
    torch.testing.assert_close(coms - com_offset, (coms[:1] - com_offset[:1]).expand(NUM_ENVS, 3), atol=1e-6, rtol=0)  # one USD COM
    # the cache needs start-up events
    for name in ("foot_friction", "base_com"):
        event = getattr(env.cfg.events, name)
        event.mode = "reset"
        try:
            with pytest.raises(RuntimeError, match="startup"):
                SimContext.from_env(env)
        finally:
            event.mode = "startup"


@pytest.mark.sim
def test_gencoord_binds_to_the_env_and_its_bodies_match_the_motion_command(env):
    from pretraining.supermimic.representations import create_representation
    from pretraining.supermimic.sim_context import SimContext

    obs, _ = env.reset()
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")
    rep = create_representation(
        "gencoord", obs["full_state"].shape[-1], NUM_JOINTS, env.step_dt, env.cfg.decimation,
        robot.data.default_joint_pos[0, cmd.joint_ids], actor_obs_mode="proprio",
    ).to(env.device)
    rep.bind(env, SimContext.from_env(env))
    assert rep.fk is cmd.motion.fk
    # the policy joints' soft limits, in JOINT_NAMES order (the articulation also has the gripper hinges)
    torch.testing.assert_close(rep._soft_joint_pos_limits, robot.data.soft_joint_pos_limits[0, cmd.joint_ids])
    full = obs["full_state"]
    pos, *_ = rep.world_frame_bodies(rep.extract_aux_state(full), rep.represent(full))
    torch.testing.assert_close(pos + env.scene.env_origins[:, None], cmd.body_pos_w, atol=1e-4, rtol=0)


@pytest.mark.sim
def test_env_adapter_and_world_model_on_the_env(env):
    from pretraining.supermimic.envs.motion_tracking_env import MotionTrackingEnv
    from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer
    from pretraining.supermimic.runners.base_runner import BaseRunnerCfg, EnvAdapter, build_world_model

    adapter = EnvAdapter(env)
    assert (adapter.num_envs, adapter.num_actions, adapter.unwrapped) == (NUM_ENVS, NUM_JOINTS, env)
    assert adapter.max_episode_length == env.max_episode_length and adapter.cfg is env.cfg
    obs = adapter.get_observations()
    assert obs.batch_size == torch.Size([NUM_ENVS]) and obs["actor"].shape == (NUM_ENVS, 87)
    next_obs, reward, dones, extras = adapter.step(torch.zeros(NUM_ENVS, NUM_JOINTS, device=env.device))
    assert next_obs["full_state"].shape == (NUM_ENVS, FULL_STATE_DIM) and reward.shape == (NUM_ENVS,)
    assert dones.dtype == torch.long and extras["time_outs"].shape == (NUM_ENVS,)

    wm = build_world_model(BaseRunnerCfg(actor_obs="proprio", dynamics_hidden_dims=(64,)), adapter, str(env.device))
    assert (wm.num_actions, wm.representation.full_state_dim) == (NUM_JOINTS, FULL_STATE_DIM)
    assert wm.sim_context.output_dim == 1 + 3 + NUM_JOINTS
    # the action map is the env's: target = action * scale + default joint position, in JOINT_NAMES order
    cmd = env.command_manager.get_term("motion")
    torch.testing.assert_close(wm.dynamics.action_offset[0], env.scene["robot"].data.default_joint_pos[0, cmd.joint_ids])
    buf = ReplayBuffer(
        {"state": FieldSpec(FULL_STATE_DIM), "terminated": FieldSpec(dtype=torch.bool),
         "truncated": FieldSpec(dtype=torch.bool)}, 4, NUM_ENVS, str(env.device),
    )
    wm_env = MotionTrackingEnv(wm.dynamics, wm.representation, adapter, NUM_ENVS, buf, 1, 0.05, str(env.device))
    weights = [env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms]
    assert wm_env.max_step_reward == pytest.approx(env.step_dt * sum(w for w in weights if w > 0))
    assert (wm_env._anchor_pos_threshold, wm_env._anchor_ori_threshold) == (
        env.cfg.terminations.anchor_pos.params["threshold"], env.cfg.terminations.anchor_ori.params["threshold"])
