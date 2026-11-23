import numpy as np
import pytest
import torch

from pretraining.constants import JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.motions import MotionLoader, check_motion_dir
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
