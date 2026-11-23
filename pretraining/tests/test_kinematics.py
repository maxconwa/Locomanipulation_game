import pytest
import torch

from pretraining.constants import DEFAULT_JOINT_POS, JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.math import quat_error_magnitude, quat_from_angle_axis, quat_mul, quat_conjugate, axis_angle_from_quat

DEFAULT = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], dtype=torch.float64)


@pytest.fixture(scope="module")
def tree():
    return UrdfTree.from_file(URDF, JOINT_NAMES)


def _fk(tree, bodies=TRACKED_BODIES):
    return ForwardKinematics(tree, bodies, "cpu", dtype=torch.float64)


def test_tree_order_and_joints(tree):
    assert tree.body_names[0] == "pelvis" and tree.parent[0] == -1
    assert all(p < i for i, p in enumerate(tree.parent[1:], 1))  # parents before children
    assert sorted(j for j in tree.joint_index if j >= 0) == list(range(27))


def test_default_pose_wrists_mirror_in_y(tree):
    fk = _fk(tree, ("left_wrist_yaw_link", "right_wrist_yaw_link"))
    pos, *_ = fk(torch.zeros(3, dtype=torch.float64), torch.tensor([1.0, 0, 0, 0], dtype=torch.float64), DEFAULT)
    left, right = pos
    assert left[1] > 0.1
    torch.testing.assert_close(left * torch.tensor([1, -1, 1]), right, atol=1e-4, rtol=0)


def test_velocities_match_finite_differences(tree):
    fk = _fk(tree)
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64) * 2 - 1  # noqa: E731
    root_pos, root_quat = r(3), quat_from_angle_axis(r(1)[0] * 3, r(3))
    q, v_root, w_root, qd = DEFAULT + 0.3 * r(27), r(3), r(3), r(27)
    h = 1e-3
    _, _, lin, ang = fk(root_pos, root_quat, q, v_root, w_root, qd)

    def stepped(s):  # the state moved by s along its velocity; the root turns about w_root (world frame)
        turn = quat_from_angle_axis(w_root.norm() * s, w_root)
        return fk(root_pos + s * v_root, quat_mul(turn, root_quat), q + s * qd)[:2]

    (pos_b, quat_b), (pos_f, quat_f) = stepped(-h), stepped(h)  # central differences
    torch.testing.assert_close((pos_f - pos_b) / (2 * h), lin, atol=1e-3, rtol=0)
    ang_fd = axis_angle_from_quat(quat_mul(quat_f, quat_conjugate(quat_b))) / (2 * h)
    torch.testing.assert_close(ang_fd, ang, atol=1e-3, rtol=0)


def test_batch_shapes(tree):
    fk = ForwardKinematics(tree, TRACKED_BODIES, "cpu")
    assert fk.body_names == list(TRACKED_BODIES)
    out = fk(torch.zeros(4, 7, 3), torch.tensor([1.0, 0, 0, 0]).expand(4, 7, 4), torch.zeros(4, 7, 27),
             torch.zeros(4, 7, 3), torch.zeros(4, 7, 3), torch.zeros(4, 7, 27))
    assert [o.shape for o in out] == [(4, 7, 14, 3), (4, 7, 14, 4), (4, 7, 14, 3), (4, 7, 14, 3)]


@pytest.mark.sim
def test_matches_isaac_body_poses(sim_app, tree):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from pretraining.robot import H1_2_PRETRAIN_CFG

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu", gravity=(0.0, 0.0, 0.0)))
    robot = Articulation(H1_2_PRETRAIN_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
    g = torch.Generator().manual_seed(1)
    q = DEFAULT + 0.4 * (torch.rand(27, generator=g, dtype=torch.float64) * 2 - 1)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[0, ids] = q.float()
    root_pose = torch.tensor([[0.1, -0.2, 1.1, *quat_from_angle_axis(torch.tensor(0.7), torch.tensor([0.3, 0.2, 1.0]))]])
    robot.write_root_pose_to_sim(root_pose)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    sim.forward()
    robot.update(0.0)
    body_ids = [robot.body_names.index(b) for b in TRACKED_BODIES]
    pos, quat, *_ = _fk(tree)(root_pose[0, :3].double(), root_pose[0, 3:].double(), q)
    torch.testing.assert_close(robot.data.body_pos_w[0, body_ids].double(), pos, atol=1e-4, rtol=0)
    err = quat_error_magnitude(robot.data.body_quat_w[0, body_ids].double(), quat)
    assert err.max() < 1e-4, dict(zip(TRACKED_BODIES, err.tolist()))
    sim.clear_instance()
