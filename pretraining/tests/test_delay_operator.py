"""The world model's effective target under the action delay (Dynamics.effective_target), against the env's substep
rule (tasks/delay.py, main's LocoManipMarlEnv._apply_action), on CPU."""

import math
from types import SimpleNamespace

import pytest
import torch

from pretraining.supermimic.models import Dynamics, delay_weights

A = 4


def _stub(decimation):
    """effective_target's and roll_targets' inputs: identity proc, the delay column 0 normalised as 5 +- 5."""
    return SimpleNamespace(proc=lambda x: x, delay=(decimation, 0, 5.0, 5.0))


def _ctx(d):
    return torch.tensor([[(float(x) - 5.0) / 5.0] for x in d])


def _brute(history, d, decimation):
    """Mean over the step's substeps of the applied target: substep k applies history[ceil(max(d-k,0)/D)]."""
    out = torch.zeros(A)
    for k in range(decimation):
        out += history[math.ceil(max(d - k, 0) / decimation)]
    return out / decimation


def _eff(stub, a, d, prev):
    return Dynamics.effective_target(stub, a, _ctx(d), prev)


@pytest.mark.parametrize("decimation", [5, 10])
def test_matches_the_substep_rule_over_an_episode_from_a_reset(decimation):
    torch.manual_seed(0)
    stub = _stub(decimation)
    for d in range(0, 11):
        measured = torch.randn(A)  # after a reset the history holds the pose the env starts in
        history = measured.expand(3, A).clone()
        prev = measured.expand(1, 2, A).clone()
        for _ in range(4):
            a = torch.randn(A)
            history = torch.cat([a[None], history[:2]])
            ours = _eff(stub, a[None], [d], prev)[0]
            torch.testing.assert_close(ours, _brute(history, d, decimation), msg=f"d={d}")
            prev = Dynamics.roll_targets(stub, a[None], prev)


def test_zero_delay_is_the_target_bit_for_bit_and_no_delay_is_unchanged():
    a, prev = torch.randn(3, A), torch.randn(3, 2, A)
    assert torch.equal(_eff(_stub(5), a, [0, 0, 0], prev), a)
    nominal = SimpleNamespace(proc=lambda x: x, delay=None)
    assert torch.equal(Dynamics.effective_target(nominal, a, _ctx([7] * 3), prev), a)
    assert torch.equal(_eff(_stub(5), a, [7] * 3, None), a)


@pytest.mark.parametrize("d", range(0, 11))
def test_gradient_shares(d):
    a = torch.randn(1, A, requires_grad=True)
    prev = torch.randn(1, 2, A, requires_grad=True)
    _eff(_stub(5), a, [d], prev).sum().backward()
    w0 = (5 - min(d, 5)) / 5
    w2 = max(d - 5, 0) / 5
    torch.testing.assert_close(a.grad, torch.full((1, A), w0))
    torch.testing.assert_close(prev.grad[0, 0], torch.full((A,), 1.0 - w0 - w2))
    torch.testing.assert_close(prev.grad[0, 1], torch.full((A,), w2))
    w = delay_weights(torch.tensor(float(d)), 5)
    assert sum(float(x) for x in w) == pytest.approx(1.0)


def test_wm_training_feeds_each_step_its_stored_previous_targets():
    from pretraining.supermimic.learners.wm_learner import WorldModelLearner

    seen = []

    def dynamics(rep, action, aux, ctx, prev=None):
        seen.append(prev)
        return aux, rep

    learner = SimpleNamespace(dynamics=dynamics)
    prev = torch.randn(2, 4, 2, A)
    WorldModelLearner._rollout(learner, torch.zeros(2, 3), torch.zeros(2, 4, A), torch.zeros(2, 1), torch.zeros(2, 1),
                               prev)
    assert len(seen) == 3 and all(torch.equal(s, prev[:, k]) for k, s in enumerate(seen))
