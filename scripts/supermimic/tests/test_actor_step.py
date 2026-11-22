"""actor_step reproduces the deployed actor: skrl's observation scaler (export_marl.RecurrentActor's formula) followed
by LSTMGaussianPolicy.compute's acting branch, over several recurrent steps."""

import pytest
import torch

from actors import StandInLSTMPolicy
from supermimic.shac import actor_step


@pytest.mark.parametrize("skip,cap,layers", [(True, 2.0, 2), (False, 0.0, 1), (True, 0.0, 1), (False, 0.5, 2)])
def test_actor_step_matches_reference(skip, cap, layers):
    torch.manual_seed(0)
    O, A, B = 9, 4, 33
    policy = StandInLSTMPolicy(O, A, hidden_size=12, num_layers=layers, skip_observations=skip, cell_cap=cap)
    mean, var = torch.randn(O), torch.rand(O) * 3 + 0.01
    h = torch.randn(layers, B, 12)
    c = torch.randn(layers, B, 12) * 3
    h_ref, c_ref = h.clone(), c.clone()
    for _ in range(4):
        obs = torch.randn(B, O) * 6  # some beyond the +-5 clip after scaling
        a, h, c = actor_step(policy, mean, var, obs, h, c)
        z = torch.clamp((obs - mean) / (torch.sqrt(var) + 1e-8), -5.0, 5.0)
        a_ref, out = policy.compute({"observations": z, "rnn": [h_ref, c_ref]})
        h_ref, c_ref = out["rnn"]
        assert torch.allclose(a, a_ref, atol=1e-6)
        assert torch.allclose(h, h_ref, atol=1e-6) and torch.allclose(c, c_ref, atol=1e-6)
    if cap > 0:
        assert c.abs().max() < cap
