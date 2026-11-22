"""ReplayBuffer: windows are contiguous in time across the ring's wrap and never straddle the write head; the alive /
next_valid masks match a brute-force reference on random done patterns; starts are stored entries."""

import torch

from supermimic.buffer import ReplayBuffer, flatten_lstm, standard_spec, unflatten_lstm


def _fill(capacity=13, num_envs=5, steps=31, seed=0, p_done=0.2):
    """A buffer whose 'state' is the global time step written (so contiguity is checkable) and whose done flags are
    random; returns the buffer and the full logs (steps, num_envs) for reference."""
    torch.manual_seed(seed)
    spec = {"state": 2, "action": 1, "terminated": 0, "truncated": 0, "reward/a": 0}
    buf = ReplayBuffer(spec, num_envs, capacity)
    term = torch.rand(steps, num_envs) < p_done / 2
    trunc = torch.rand(steps, num_envs) < p_done / 2
    for t in range(steps):
        stamp = torch.full((num_envs,), float(t))
        env = torch.arange(num_envs, dtype=torch.float32)
        buf.add({"state": torch.stack([stamp, env], -1), "action": stamp.unsqueeze(-1), "terminated": term[t],
                 "truncated": trunc[t], "reward/a": stamp})
    return buf, term, trunc, steps


def test_windows_contiguous_and_behind_the_head():
    buf, term, trunc, steps = _fill()
    assert buf.full and len(buf) == 13
    k = 6
    w = buf.sample_windows(4000, k)
    t = w["state"][..., 0]
    assert w["state"].shape == (4000, k + 1, 2) and w["action"].shape == (4000, k, 1)
    assert w["reward/a"].shape == (4000, k) and w["terminated"].dtype == torch.bool
    # contiguous in time (across the wrap: steps 31 > capacity 13) and within one env
    assert torch.all(t[:, 1:] - t[:, :-1] == 1)
    assert torch.all(w["state"][..., 1] == w["state"][:, :1, 1])
    # only the stored range [steps - capacity, steps - 1]: never straddles the write head
    assert t.min() >= steps - 13 and t.max() <= steps - 1
    # transition fields belong to the window's first k entries
    assert torch.all(w["action"][..., 0] == t[:, :k])
    # every start is reachable
    assert set(t[:, 0].long().tolist()) == set(range(steps - 13, steps - k))


def test_masks_match_brute_force():
    buf, term, trunc, steps = _fill(p_done=0.4, seed=3)
    k = 5
    w = buf.sample_windows(3000, k)
    t0 = w["state"][:, 0, 0].long()
    env = w["state"][:, 0, 1].long()
    for b in range(3000):
        alive = 1.0
        for j in range(k):
            done = bool(term[t0[b] + j, env[b]] or trunc[t0[b] + j, env[b]])
            assert w["alive"][b, j] == alive
            assert w["next_valid"][b, j] == (alive if not done else 0.0)
            if done:
                alive = 0.0


def test_not_full_and_starts():
    torch.manual_seed(1)
    spec = standard_spec(3, 2, 4, {"legs": 5, "arms": 6}, {"legs": 2, "arms": 1}, {"legs": (2, 8)})
    buf = ReplayBuffer(spec, num_envs=7, capacity=50)
    written = []
    for t in range(9):
        step = {name: torch.randn((7, dim) if dim else (7,)) for name, dim in spec.items()}
        step["terminated"] = torch.rand(7) < 0.1
        step["truncated"] = torch.zeros(7, dtype=torch.bool)
        h = torch.randn(2, 7, 8)
        step["h/legs"] = flatten_lstm(h)
        assert torch.equal(unflatten_lstm(step["h/legs"], 2), h)
        buf.add(step)
        written.append(step["state"])
    assert len(buf) == 9 and not buf.full
    w = buf.sample_windows(100, 8)  # needs all 9 entries: the only start is 0
    assert torch.equal(w["state"][:, 0], torch.stack(written)[0][_env_of(w["state"][:, 0], torch.stack(written)[0])])
    starts = buf.sample_starts(500)
    table = torch.stack(written).reshape(-1, 3)
    match = (starts["state"].unsqueeze(1) == table.unsqueeze(0)).all(-1).any(1)
    assert match.all()
    assert starts["h/legs"].shape == (500, 16) and starts["obs/arms"].shape == (500, 6)


def _env_of(rows, first):
    return (rows.unsqueeze(1) == first.unsqueeze(0)).all(-1).float().argmax(1)
