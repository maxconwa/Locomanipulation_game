"""Latent policies in MuJoCo (evals/mujoco/latent.py) and LocoManip-Marl-Torso-v0's torso action in the sim.

In the python with mujoco and torch (CPU), from the repository root:

    python -m evals.mujoco.test_latent            (or python -m pytest evals/mujoco/test_latent.py where pytest is)

The decoders here are small stand-ins scripted with torch alone (export_decoders' real ones are tested against their
eager originals in pretraining/tests/test_latent_evals.py). The frame's fixture is the Isaac env's own frame, recorded
by test_record_the_isaac_proprio_frame_fixture (a sim test); without it that case says so and is skipped.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evals.mujoco import latent as LM  # noqa: E402
from evals.mujoco.sim import ALL_NAMES, MarlMujoco  # noqa: E402

FIXTURE = REPO / "pretraining" / "tests" / "fixtures" / "pulse_frame.json"
TEMPLATE = REPO / "evals" / "deploy" / "hw_deploy_template.yaml"


class _Skip(Exception):
    pass


def settings(**extra) -> dict:
    s = yaml.safe_load(TEMPLATE.read_text())
    s.update(control_dt=0.01, rest_pose=[[0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]],
             imu={"frame": "imu_link", "source": "lowstate", "orientation": "calibrated",
                  "residual_deg": {"roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0]}},
             clip_actions=5.0, **extra)
    return s


class _Decoder(nn.Module):
    def __init__(self, out: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(32 + 87, out)
        nn.init.uniform_(self.lin.weight, -0.02, 0.02)

    def forward(self, z: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.cat([z, frame], -1))


class _Encoder:
    """An skrl export stand-in: obs[None] -> a fixed u[None]."""

    def __init__(self, u):
        self.u, self.calls = torch.as_tensor(u, dtype=torch.float32), 0

    def __call__(self, x):
        self.calls += 1
        return self.u[None]

    def reset_state(self):
        pass


def fake_export(d: Path, default: list[float]) -> Path:
    torch.jit.script(_Decoder(12, 0)).save(str(d / "legs_decoder.pt"))
    torch.jit.script(_Decoder(15, 1)).save(str(d / "upper_decoder.pt"))
    info = {"plant": "main", "step_dt": 0.01, "latent_dim": 32, "proprio_dim": 87, "joint_names": list(ALL_NAMES),
            "default_joint_pos": default, "action_scale": [0.25] * 27, "target_limits": [[-3.0, 3.0]] * 27,
            "agent_joints": {"legs": [0, 12], "arms": [12, 27]}, "stage1": "x", "prior": None}
    (d / "plant.json").write_text(json.dumps(info))
    return d


def latent_sim(d: Path, u_legs, u_arms, **kw):
    s = settings()
    pol = LM.LatentMujocoPolicy(fake_export(d, s["default_joint_pos"]),
                                {"legs": _Encoder(u_legs), "arms": _Encoder(u_arms)})
    return LM.LatentMarlMujoco(pol, s, physics="isaac", **kw), pol


def test_frame_at_the_default_pose_standing_still():
    """Upright, still, at the default pose: [0, 0, -1, gyro 0, q - default 0, dq 0, last action 0]."""
    with tempfile.TemporaryDirectory() as tmp:
        sim, pol = latent_sim(Path(tmp), np.zeros(32), np.zeros(32), noise=False)
        sim.reset(np.random.default_rng(0))
        import mujoco
        sim.d.qpos[3:7], sim.d.qvel[:] = (1.0, 0.0, 0.0, 0.0), 0.0
        sim.d.qpos[sim.qadr] = sim.default
        mujoco.mj_forward(sim.m, sim.d)
        frame = pol.frame(sim, noise=False)
        assert frame.shape == (87,)
        assert np.allclose(frame[:3], [0.0, 0.0, -1.0], atol=1e-6) and np.allclose(frame[3:], 0.0, atol=1e-9)


def test_frame_equals_the_isaac_frame_of_the_same_state():
    """The fixture's state (root orientation and angular velocity, joint angles and speeds) set in MuJoCo gives the
    Isaac frame: gravity and gyro to 1e-3 (another engine's IMU site), the rest to 1e-5."""
    if not FIXTURE.is_file():
        raise _Skip(f"no {FIXTURE.relative_to(REPO)}: record it with the sim test"
                    " test_record_the_isaac_proprio_frame_fixture (pretraining/tests/test_latent_evals.py)")
    rec = json.loads(FIXTURE.read_text())
    assert rec["joint_names"] == list(ALL_NAMES)
    import mujoco
    with tempfile.TemporaryDirectory() as tmp:
        s = settings(default_joint_pos=rec["default_joint_pos"])
        pol = LM.LatentMujocoPolicy(fake_export(Path(tmp), rec["default_joint_pos"]),
                                    {"legs": _Encoder(np.zeros(32)), "arms": _Encoder(np.zeros(32))})
        sim = LM.LatentMarlMujoco(pol, s, physics="isaac", noise=False)
        sim.reset(np.random.default_rng(0))
        q = np.array(rec["root_quat_wxyz"])
        sim.d.qpos[3:7], sim.d.qpos[sim.qadr] = q, rec["joint_pos"]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        sim.d.qvel[:] = 0.0
        sim.d.qvel[0:3] = rec["root_lin_vel_w"]
        sim.d.qvel[3:6] = R.reshape(3, 3).T @ np.array(rec["root_ang_vel_w"])  # a free joint's: in the body frame
        sim.d.qvel[sim.vadr] = rec["joint_vel"]
        mujoco.mj_forward(sim.m, sim.d)
        pol.last_action = np.array(rec["frame"][60:])
        frame, want = pol.frame(sim, noise=False), np.array(rec["frame"])
        assert np.allclose(frame[:6], want[:6], atol=1e-3), (frame[:6], want[:6])
        assert np.allclose(frame[6:], want[6:], atol=1e-5)


def test_a_step_decodes_both_latents_on_one_frame_and_moves_the_last_action_on():
    rng = np.random.default_rng(1)
    u_l, u_a = rng.uniform(-6, 6, 32), rng.uniform(-1, 1, 32)
    with tempfile.TemporaryDirectory() as tmp:
        sim, pol = latent_sim(Path(tmp), u_l, u_a, noise=False)
        sim.reset(np.random.default_rng(0))
        frame0 = pol.frame(sim, noise=False)
        obs = sim.observations()
        sim.step(obs)
        dec = {a: torch.jit.load(str(Path(tmp) / n)) for a, n in (("legs", "legs_decoder.pt"),
                                                                    ("arms", "upper_decoder.pt"))}
        f = torch.as_tensor(frame0, dtype=torch.float32)[None]
        raw = np.concatenate([dec[a](torch.tanh(torch.as_tensor(np.clip(u, -5, 5), dtype=torch.float32))[None], f)[0]
                              .detach().numpy() for a, u in (("legs", u_l), ("arms", u_a))])
        assert np.allclose(pol.last_action, raw, atol=1e-6)
        want = np.clip(pol.default + 0.25 * raw, sim.lower, sim.upper)
        assert np.allclose(sim.applied, want, atol=1e-6)
        # what the agents observe of their action: the latent as clamped
        legs, arms = sim.applied_actions()
        assert np.allclose(legs, np.clip(u_l, -5, 5)) and np.allclose(arms, u_a)
        assert np.allclose(pol.frame(sim, noise=False)[60:], raw)
        sim.reset(np.random.default_rng(0))
        assert not pol.last_action.any()


def test_the_command_encoders_read_the_velocity_and_the_goals_xyzw():
    sim = type("S", (), {})()
    sim.observed_cmd = lambda: np.array([0.1, 0.2, 0.3])
    sim.believed = np.array([[1, 2, 3, 0.5, 0.1, 0.2, 0.3], [4, 5, 6, 0.9, 0.4, 0.5, 0.6]])
    assert np.allclose(LM.command_vector(sim, "legs"), [0.1, 0.2, 0.3])
    assert np.allclose(LM.command_vector(sim, "arms"), [1, 2, 3, 0.1, 0.2, 0.3, 0.5, 4, 5, 6, 0.4, 0.5, 0.6, 0.9])


def test_command_encoder_carries_its_state_and_hands_atanh_z():
    class Enc(nn.Module):
        def forward(self, obs: torch.Tensor, command: torch.Tensor, hc: torch.Tensor):
            return torch.tanh(obs[:, :32] + command.sum()), hc + 1.0

    with tempfile.TemporaryDirectory() as tmp:
        torch.jit.script(Enc()).save(str(Path(tmp) / "e.pt"))
        e = LM.CommandEncoder(Path(tmp) / "e.pt", lstm=8)
        obs = np.linspace(-2, 2, 40)
        u = e(obs, np.zeros(3))
        assert np.allclose(u, obs[:32], atol=1e-4) and float(e.hc.mean()) == 1.0
        e(obs, np.zeros(3))
        assert float(e.hc.mean()) == 2.0
        e.reset_state()
        assert e.hc is None
        assert np.isfinite(e(np.full(40, 50.0), np.zeros(3))).all()


def test_the_torso_action_drives_the_torso_joint():
    """LocoManip-Marl-Torso-v0 (export.yaml torso_action_scale): the arms' first action, target default + 0.25 a,
    observed as the arms' torso_action term; without the key the torso is held at 0 as before."""
    s = settings(torso_action_scale=0.25)
    s["arms_obs_terms"] = ["base_ang_vel", "projected_gravity", "actions", "torso_action"]
    sim = MarlMujoco({"legs": _Encoder(np.zeros(12)), "arms": _Encoder(np.zeros(15))}, s, physics="isaac")
    sim.reset(np.random.default_rng(0))
    a_arms = np.r_[0.4, np.zeros(14)]
    sim.process_actions(np.zeros(12), a_arms)
    assert math.isclose(sim.applied[12], sim.default[12] + 0.1, abs_tol=1e-9)
    assert math.isclose(sim.torso_action, 0.4, abs_tol=1e-9)
    assert math.isclose(sim.observations()["arms"][-1], 0.4, abs_tol=1e-9)
    plain = MarlMujoco({"legs": _Encoder(np.zeros(12)), "arms": _Encoder(np.zeros(14))}, settings(), physics="isaac")
    plain.reset(np.random.default_rng(0))
    plain.process_actions(np.zeros(12), np.zeros(14))
    assert plain.applied[12] == 0.0


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except _Skip as skip:
            print(f"SKIP {name}: {skip}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {err!r}")
    sys.exit(1 if failed else 0)
