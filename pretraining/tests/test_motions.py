import numpy as np
import pytest
import torch

from pretraining.constants import JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.motions import MotionLoader, check_motion_dir, resample_clip
from standins import write_clips


@pytest.fixture(scope="module")
def fk():
    return ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), TRACKED_BODIES, "cpu")


def test_loads_clips_and_concatenates(tmp_path, fk):
    write_clips(tmp_path, n=3, frames=120)
    (tmp_path / "notes.txt").write_text("not a clip")
    motion = MotionLoader(tmp_path, fk, "cpu")
    assert motion.time_step_total == 360
    assert motion.motion_ends.tolist() == [120, 240, 360]
    assert motion.motion_starts.tolist() == [0, 120, 240]
    assert motion.fps == 50.0
    assert [p.name for p in motion.files] == ["standin_000.npz", "standin_001.npz", "standin_002.npz"]
    assert motion.joint_pos.shape == (360, 27)
    assert motion.body_pos_w.shape == (360, len(TRACKED_BODIES), 3)
    assert motion.body_quat_w.shape == (360, len(TRACKED_BODIES), 4)
    assert motion.body_lin_vel_w.shape == motion.body_ang_vel_w.shape == (360, len(TRACKED_BODIES), 3)
    # the second clip's first frame is the file's first frame, and its bodies are FK of it
    clip = np.load(tmp_path / "standin_001.npz")
    torch.testing.assert_close(motion.joint_pos[120], torch.from_numpy(clip["joint_pos"][0]))
    pos, *_ = fk(motion.root_pos_w[120], motion.root_quat_w[120], motion.joint_pos[120])
    torch.testing.assert_close(motion.body_pos_w[120], pos)
    torch.testing.assert_close(motion.body_pos_w[120, 0], motion.root_pos_w[120])  # pelvis is the root


def test_refuses_empty_or_missing_dir(tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(ValueError, match=str(missing)):
        check_motion_dir(missing)
    with pytest.raises(ValueError, match=f"{tmp_path}.*no .npz"):
        check_motion_dir(tmp_path)


def test_refuses_wrong_fps_or_joint_order(tmp_path):
    (bad,) = write_clips(tmp_path / "fps", n=1)
    clip = dict(np.load(bad))
    np.savez(bad, **{**clip, "fps": np.array([30.0])})
    with pytest.raises(ValueError, match=f"{bad.name}.*30 Hz.*50"):
        check_motion_dir(tmp_path / "fps")

    (bad,) = write_clips(tmp_path / "order", n=1)
    clip = dict(np.load(bad))
    np.savez(bad, **{**clip, "joint_names": clip["joint_names"][::-1]})
    with pytest.raises(ValueError, match=f"{bad.name}.*joint_names"):
        check_motion_dir(tmp_path / "order")


def _direct(motion, fk, frames):
    return fk(motion.root_pos_w[frames], motion.root_quat_w[frames], motion.joint_pos[frames],
              motion.root_lin_vel_w[frames], motion.root_ang_vel_w[frames], motion.joint_vel[frames])


def test_lazy_body_attributes_equal_direct_fk(tmp_path, fk):
    write_clips(tmp_path, n=3, frames=40)
    motion = MotionLoader(tmp_path, fk, "cpu")
    g = torch.Generator().manual_seed(0)
    boundaries = torch.tensor([0, 39, 40, 79, 80, 119])  # first and last frames of each clip
    attrs = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    for frames in (boundaries, torch.randint(0, 120, (5, 3), generator=g), torch.randint(0, 120, (2, 3, 4), generator=g),
                   torch.tensor(7), 7):
        direct = _direct(motion, fk, frames)
        for name, want in zip(attrs, direct):
            torch.testing.assert_close(getattr(motion, name)[frames], want, atol=0, rtol=0)
            for body in (0, 5, torch.tensor(13)):
                torch.testing.assert_close(getattr(motion, name)[frames, body], want[..., body, :], atol=0, rtol=0)
        for got, want in zip(motion.body_state(frames), direct):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert motion.body_pos_w.shape == (120, len(TRACKED_BODIES), 3)
    for bad in (slice(0, 3), (boundaries, 0, 1), ..., (boundaries, slice(None)), torch.tensor([True, False])):
        with pytest.raises(TypeError, match="body_pos_w"):
            motion.body_pos_w[bad]


def test_loader_holds_nothing_frames_by_bodies(tmp_path, fk):
    write_clips(tmp_path, n=3, frames=40)
    motion = MotionLoader(tmp_path, fk, "cpu")
    for name, value in vars(motion).items():
        if isinstance(value, torch.Tensor):
            assert value.dim() < 2 or value.shape[:2] != (120, len(TRACKED_BODIES)), name


def _clip(quats: np.ndarray) -> dict[str, np.ndarray]:
    """A clip of len(quats) frames: ramps in every vector field, the given root quaternions."""
    n = len(quats)
    ramp = np.arange(n, dtype=np.float32)[:, None]
    return {
        "fps": np.array([50.0]), "joint_names": np.array(JOINT_NAMES),
        "joint_pos": ramp * np.linspace(0.1, 1.0, 27, dtype=np.float32), "joint_vel": ramp * np.ones(27, np.float32),
        "root_pos_w": ramp * np.array([1.0, 2.0, 0.0], np.float32), "root_quat_w": quats.astype(np.float32),
        "root_lin_vel_w": ramp * np.ones(3, np.float32), "root_ang_vel_w": -ramp * np.ones(3, np.float32),
    }


def test_resample_doubles_frames_and_keeps_originals(tmp_path):
    [path] = write_clips(tmp_path, n=1, frames=30)
    clip = dict(np.load(path))
    out = resample_clip(clip, 2)
    assert float(out["fps"][0]) == 100.0
    assert out["joint_names"].tolist() == clip["joint_names"].tolist()
    for name in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"):
        assert out[name].shape == (59, *clip[name].shape[1:]) and out[name].dtype == clip[name].dtype
        np.testing.assert_array_equal(out[name][::2], clip[name])
    # odd frames of linear fields are the midpoints
    np.testing.assert_allclose(out["joint_pos"][1::2], (clip["joint_pos"][:-1] + clip["joint_pos"][1:]) / 2,
                               atol=1e-6)


def test_resample_takes_the_short_arc_and_keeps_edges():
    q0 = np.array([np.cos(0.2), 0.0, np.sin(0.2), 0.0])
    neighbour = np.array([np.cos(0.35), 0.0, np.sin(0.35) * 0.8, np.sin(0.35) * 0.6])
    out = resample_clip(_clip(np.stack([q0, -neighbour])), 2)  # q and -q: the long arc would swing past 300 deg
    assert out["root_quat_w"].shape == (3, 4)
    mid = out["root_quat_w"][1].astype(np.float64)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-6
    short = (q0 + neighbour) / np.linalg.norm(q0 + neighbour)  # slerp at t = 1/2 on the short arc
    assert min(np.abs(mid - short).max(), np.abs(mid + short).max()) < 1e-6
    np.testing.assert_allclose(out["root_pos_w"][1], [0.5, 1.0, 0.0], atol=1e-7)
    one = resample_clip(_clip(q0[None]), 2)
    assert one["root_quat_w"].shape == (1, 4) and float(one["fps"][0]) == 100.0


def test_loader_resamples_to_an_integer_multiple(tmp_path, fk):
    write_clips(tmp_path, n=2, frames=40)
    motion = MotionLoader(tmp_path, fk, "cpu", fps=100.0)
    assert motion.fps == 100.0 and motion.time_step_total == 2 * 79
    assert motion.motion_ends.tolist() == [79, 158]


def test_loader_refuses_non_integer_ratio(tmp_path, fk):
    write_clips(tmp_path, n=1, frames=10)
    with pytest.raises(ValueError, match=r"75.*50|50.*75"):
        MotionLoader(tmp_path, fk, "cpu", fps=75.0)
