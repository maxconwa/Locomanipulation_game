"""Each estimator against the synthetic system, whose answers are known in closed form (synthetic.py)."""

import numpy as np

from evals.isaac.arms.steerability import estimators as est
from evals.isaac.arms.steerability import se3
from evals.isaac.arms.steerability.closed_loop import OuterLoop, run_closed_loop
from evals.isaac.arms.steerability.runner import run
from evals.isaac.arms.steerability.schedules import repeat_schedule, step_schedule
from evals.isaac.arms.steerability.synthetic import SyntheticArms, SyntheticParams

MM, DEG = 1e-3, np.pi / 180


def make_bases(n: int, seed: int, rotate: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pose = se3.identity((n, 2))
    pose[..., :3] = rng.uniform(-0.2, 0.2, (n, 2, 3))
    if rotate:
        pose[..., 3:] = se3.quat_from_rotvec(rng.normal(size=(n, 2, 3)) * 0.7)
    return pose


def test_se3_round_trips():
    rng = np.random.default_rng(0)
    a, b = make_bases(16, 1), make_bases(16, 2)
    back = se3.perturb(a, se3.relative(a, b))
    assert np.allclose(back[..., :3], b[..., :3])
    assert np.allclose(np.abs((back[..., 3:] * b[..., 3:]).sum(-1)), 1.0)
    d = rng.normal(size=(16, 2, 6)) * 0.3
    assert np.allclose(se3.relative(a, se3.perturb(a, d)), d)


def test_local_jacobian_recovers_gain():
    rng = np.random.default_rng(3)
    gain = np.eye(6) + 0.1 * rng.normal(size=(6, 6))
    arms = SyntheticArms(4, SyntheticParams(gain=gain), seed=0)
    plan = step_schedule(make_bases(4, 0), np.array([[0.5 * MM, 0.5 * DEG], [5 * MM, 5 * DEG]]), repeats=1,
                         sham_repeats=1, settle_s=1.5, approach_s=2.0, goal_mode=False, seed=0)
    resp = est.step_responses(plan, run(arms, plan.segments))
    assert resp["flat"].all()
    J = est.local_jacobians(plan, resp)
    assert np.allclose(J, gain, atol=1e-3)


def test_jacobian_includes_the_bias_gradient():
    params = SyntheticParams(bias_amp=[1 * MM, 0, 0, 0, 0, 0], bias_wavelength=0.1)
    arms = SyntheticArms(8, params, seed=0)
    bases = make_bases(8, 4, rotate=False)
    plan = step_schedule(bases, np.array([[0.2 * MM, 0.2 * DEG]]), repeats=1, sham_repeats=0, settle_s=1.5,
                         approach_s=2.0, goal_mode=False, seed=1)
    J = est.local_jacobians(plan, est.step_responses(plan, run(arms, plan.segments)))
    expected = 1.0 + arms.bias_gradient_x(bases[..., :3])[..., 0]
    assert np.allclose(J[:, :, 0, 0, 0], expected, rtol=0.02)


def test_jacobian_metrics():
    J = np.eye(6)
    J[:3, :3] = np.diag([1.0, 2.0, 4.0])
    J[0, 3] = 0.01
    m = est.jacobian_metrics(J)
    assert np.allclose(m["trans_gain"], [1, 2, 4])
    assert np.isclose(m["trans_cond"], 4.0)
    assert np.isclose(m["trans_coupling"], 0.0)
    assert np.isclose(m["rot_identity_dev"], 0.0)
    assert np.isclose(m["rot_to_trans"][0, 0], 0.01)
    assert np.isnan(est.jacobian_metrics(np.full((6, 6), np.nan))["trans_cond"])


def test_repeatability_matches_goal_noise():
    std = np.array([0.3 * MM] * 3 + [0.2 * DEG] * 3)
    arms = SyntheticArms(256, SyntheticParams(goal_std=std), seed=0)
    bases = make_bases(256, 5)
    starts = np.stack([make_bases(256, 10 + k) for k in range(8)])
    plan = repeat_schedule(bases, starts, settle_s=1.5, approach_s=1.5)
    rep = est.repeatability(plan, run(arms, plan.segments))
    assert np.allclose(np.sqrt((rep["std"] ** 2).mean(axis=(0, 1))), std, rtol=0.05)


def test_hysteresis_gap_equals_backlash():
    arms = SyntheticArms(8, SyntheticParams(backlash=[1 * MM] * 3 + [0.0] * 3), seed=0)
    plan = step_schedule(make_bases(8, 6), np.array([[2 * MM, 2 * DEG]]), repeats=1, sham_repeats=0, settle_s=1.5,
                         approach_s=2.0, goal_mode=False, seed=2)
    resp = est.step_responses(plan, run(arms, plan.segments))
    trans = plan.axis < 3
    assert np.allclose(resp["gap"][trans], 1 * MM, atol=1e-5)
    assert np.allclose(resp["gap"][~trans], 0.0, atol=1e-5)  # what 1.5 s leaves of a 2 deg step's settling


def test_resolution_from_white_noise():
    # settled over 0.5 s: 50 samples of 0.2 mm noise, so a response's noise is sqrt(2) x 0.2 / sqrt(50) = 0.04 mm and
    # 90% correct signs need a step above 1.28 x 0.04 = 0.05 mm
    arms = SyntheticArms(64, SyntheticParams(white_std=0.2 * MM), seed=0)
    scales = np.array([[s * MM, s * DEG] for s in (0.01, 0.02, 0.05, 0.1, 0.2)])
    plan = step_schedule(make_bases(64, 7), scales, repeats=2, sham_repeats=2, settle_s=1.0, approach_s=2.0,
                         goal_mode=False, seed=3)
    resp = est.step_responses(plan, run(arms, plan.segments))
    trans = plan.axis < 3
    pooled = [np.concatenate([resp["resp_plus"][trans & (plan.scale == s)], resp["resp_minus"][trans & (plan.scale == s)]])
              for s in range(len(scales))]
    sham = np.concatenate([resp["resp_plus"][trans & (plan.scale < 0)], resp["resp_minus"][trans & (plan.scale < 0)]])
    assert est.resolution(pooled, sham, scales[:, 0]) in (0.05 * MM, 0.1 * MM)


def test_flatness_flags_drift():
    arms = SyntheticArms(4, SyntheticParams(drift=[1 * MM] * 3 + [1 * DEG] * 3), seed=0)  # per s
    plan = step_schedule(make_bases(4, 8), np.array([[1 * MM, 1 * DEG]]), repeats=1, sham_repeats=1, settle_s=1.0,
                         approach_s=1.0, goal_mode=False, seed=4)
    resp = est.step_responses(plan, run(arms, plan.segments))
    assert not resp["flat"].any()


def test_step_dynamics_of_an_underdamped_system():
    zeta = 0.5
    arms = SyntheticArms(4, SyntheticParams(omega=2 * np.pi, zeta=zeta), seed=0)
    plan = step_schedule(make_bases(4, 9), np.array([[5 * MM, 5 * DEG]]), repeats=1, sham_repeats=0, settle_s=4.0,
                         approach_s=4.0, goal_mode=False, seed=5)
    result = run(arms, plan.segments, trace=True)
    m = 0  # the first item's + step
    trace = np.take_along_axis(result.traces[1], plan.axis[m][None, ..., None], -1)[..., 0]  # (T, N, arms)
    dyn = est.step_dynamics(trace.reshape(len(trace), -1), result.dt, band=0.02 * 5 * MM)
    expected = np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2))  # 0.163
    trans = (plan.axis[m] < 3).reshape(-1)
    assert np.allclose(dyn["overshoot"][trans], expected, atol=0.01)
    # 2% envelope exp(-zeta omega t) / sqrt(1 - zeta^2) < 0.02 at t = 1.29 s
    assert np.all((dyn["settle_time"][trans] > 0.8) & (dyn["settle_time"][trans] < 1.4))


def test_validity_radius_of_a_sine_bias():
    # bias_x = A sin(2 pi x / L): the RMS change over h is sqrt(2) A |sin(pi h / L)|, under 0.2 mm up to h = 4.5 mm
    params = SyntheticParams(bias_amp=[1 * MM, 0, 0, 0, 0, 0], bias_wavelength=0.1)
    arms = SyntheticArms(128, params, seed=0)
    scales = np.array([[s * MM, s * DEG] for s in (1, 2, 5, 10)])
    plan = step_schedule(make_bases(128, 11, rotate=False), scales, repeats=1, sham_repeats=1, settle_s=1.5,
                         approach_s=2.0, goal_mode=False, seed=6)
    result = run(arms, plan.segments)
    inc = est.field_increments(plan, result, est.step_responses(plan, result))
    along_x = plan.axis == 0
    levels, g_pos, _ = est.variogram(inc["h"][along_x], inc["delta"][along_x])
    assert np.allclose(np.sqrt(2 * g_pos[1:]), np.sqrt(2) * 1 * MM * np.abs(np.sin(np.pi * levels[1:] / 0.1)),
                       rtol=0.15)
    assert est.validity_radius(levels, g_pos, 0.2 * MM) == 2 * MM


def test_closed_loop_converges_at_the_predicted_rate_and_floor():
    g = 0.5
    params = SyntheticParams(gain=g * np.eye(6), goal_std=3 * MM)
    targets = make_bases(256, 12)
    loop = OuterLoop(gain=1.0, settle_s=1.5, corrections=6)
    err = run_closed_loop(SyntheticArms(256, params, seed=0), targets, loop, approach_s=2.0, seed=0)["error"]
    norm = np.linalg.norm(err[..., :3], axis=-1)
    assert np.allclose(np.median(norm[3] / norm[0]), (1 - g) ** 3, rtol=0.02)

    sigma = 0.1 * MM
    noisy = OuterLoop(gain=1.0, settle_s=1.5, corrections=16, noise_pos=sigma, noise_rot=sigma)
    err = run_closed_loop(SyntheticArms(256, params, seed=1), targets, noisy, approach_s=2.0, seed=1)["error"]
    floor = err[8:, ..., 0].std()
    cov = est.predicted_floor(g * np.eye(6), 1.0, sigma ** 2 * np.eye(6), np.zeros((6, 6)))
    assert np.isclose(floor, np.sqrt(cov[0, 0]), rtol=0.15)  # g sigma / sqrt(1 - (1 - g)^2) = 0.58 sigma
    assert np.isinf(est.predicted_floor(3.0 * np.eye(6), 1.0, np.eye(6), np.eye(6))).all()
