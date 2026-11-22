"""The replay buffer: a ring of the last `capacity` env steps of every env, one tensor per named field.

Entry t of an env holds what was true before acting at t (state, observations, goal states, sim context, the actors'
LSTM state), the action taken at t (all agents, concatenated, as executed: noise and clamp included) and what that
action produced (per-agent reward, terminated, truncated). If entry t is done, entry t + 1 is the first step of the
next episode: the env auto-resets, so the pre-reset successor of a done step is not stored, and the world model
learns that last transition's reward and termination from its own predicted successor (world_model.py).

Windows are k + 1 contiguous entries of one env that never straddle the write head, and may cross episode ends:
rather than rejecting those (which would bias the sample against short episodes and the falls that end them) each
window carries masks, `alive` (no done earlier in the window) and `next_valid` (alive and this step not done, so entry
j + 1 really is the successor of entry j), which the losses use.
"""

from __future__ import annotations

import torch

__all__ = ["ReplayBuffer", "flatten_lstm", "is_transition", "unflatten_lstm", "standard_spec"]

# fields of a transition (k per window) rather than of a state (k + 1 per window)
TRANSITION_FIELDS = ("action", "terminated", "truncated")
TRANSITION_PREFIXES = ("reward/", "trip/")
BOOL_FIELDS = ("terminated", "truncated")


def flatten_lstm(x: torch.Tensor) -> torch.Tensor:
    """(num_layers, N, hidden) -> (N, num_layers * hidden), the buffer's layout of an LSTM state."""
    return x.transpose(0, 1).reshape(x.shape[1], -1)


def unflatten_lstm(x: torch.Tensor, num_layers: int) -> torch.Tensor:
    """(N, num_layers * hidden) -> (num_layers, N, hidden), contiguous, as nn.LSTM takes it."""
    return x.reshape(x.shape[0], num_layers, -1).transpose(0, 1).contiguous()


def standard_spec(state_dim: int, ctx_dim: int, action_dim: int, obs_dims: dict[str, int], goal_dims: dict[str, int],
                  lstm_sizes: dict[str, tuple[int, int]], trips: bool = False) -> dict[str, int]:
    """The fields world_model.py and shac.py read. obs_dims / goal_dims: every agent's observation and raw goal-state
    widths; lstm_sizes: (num_layers, hidden_size) of each trained agent's actor; trips: a trip/<agent> field per agent
    (1 on the step a GOLEM e-stop trip of that agent's starts, as float; the world model's trip heads). Scalar fields
    have dim 0."""
    spec = {"state": state_dim, "ctx": ctx_dim, "action": action_dim, "terminated": 0, "truncated": 0}
    for agent in obs_dims:
        spec[f"obs/{agent}"] = obs_dims[agent]
        spec[f"goal/{agent}"] = goal_dims[agent]
        spec[f"reward/{agent}"] = 0
        if trips:
            spec[f"trip/{agent}"] = 0
    for agent, (layers, hidden) in lstm_sizes.items():
        spec[f"h/{agent}"] = spec[f"c/{agent}"] = layers * hidden
    return spec


def is_transition(name: str) -> bool:
    """Whether a field belongs to a transition (k per window) rather than a state (k + 1 per window)."""
    return name in TRANSITION_FIELDS or name.startswith(TRANSITION_PREFIXES)


class ReplayBuffer:
    """capacity steps x num_envs of every field in spec ({name: dim}, dim 0 = one scalar per env), on device.
    terminated / truncated are stored as bool, everything else as dtype."""

    def __init__(self, spec: dict[str, int], num_envs: int, capacity: int, device="cpu", dtype=torch.float32):
        for name in BOOL_FIELDS:
            if name not in spec:
                raise ValueError(f"the spec needs a '{name}' field (the window masks are built from it)")
        self.spec, self.num_envs, self.capacity, self.device = dict(spec), num_envs, capacity, torch.device(device)
        self.data: dict[str, torch.Tensor] = {}
        for name, dim in spec.items():
            shape = (capacity, num_envs) + ((dim,) if dim else ())
            self.data[name] = torch.zeros(shape, dtype=torch.bool if name in BOOL_FIELDS else dtype, device=self.device)
        self.ptr = 0     # next physical row to write
        self.size = 0    # rows stored

    def __len__(self) -> int:
        """Stored entries per env."""
        return self.size

    @property
    def full(self) -> bool:
        return self.size == self.capacity

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.data.values())

    def add(self, step: dict[str, torch.Tensor]) -> None:
        """Write one time step of every env: step[name] is (num_envs, dim) or (num_envs,) / (num_envs, 1) for scalars.
        Every field of the spec must be given (a silently stale field would poison every window through it)."""
        missing = set(self.spec) - set(step)
        if missing:
            raise KeyError(f"add() is missing fields {sorted(missing)}")
        for name, x in step.items():
            if name not in self.data:
                raise KeyError(f"'{name}' is not a field of this buffer")
            row = self.data[name][self.ptr]
            row.copy_(x.detach().reshape(row.shape))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _oldest(self) -> int:
        return self.ptr if self.full else 0

    def sample_windows(self, batch: int, k: int, fields=None) -> dict[str, torch.Tensor]:
        """batch windows of k steps, uniform over (env, start): (B, k + 1, ...) of state-like fields and (B, k, ...)
        of transition fields (action, reward/*, trip/*, terminated, truncated), plus float masks `alive` and `next_valid`
        (B, k). fields: the names to gather (default all; the masks are always built)."""
        if self.size < k + 1:
            raise ValueError(f"the buffer holds {self.size} steps per env, a window needs {k + 1}")
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        start = torch.randint(0, self.size - k, (batch,), device=self.device)  # chronological offset from the oldest
        steps = torch.arange(k + 1, device=self.device)
        rows = (self._oldest() + start.unsqueeze(1) + steps) % self.capacity    # (B, k + 1)
        cols = env.unsqueeze(1)
        out = {}
        for name in self.spec if fields is None else fields:
            if is_transition(name):
                out[name] = self.data[name][rows[:, :k], cols]
            else:
                out[name] = self.data[name][rows, cols]
        done = (self.data["terminated"][rows[:, :k], cols] | self.data["truncated"][rows[:, :k], cols]).float()
        not_done = 1.0 - done
        # alive_j = prod_{i < j} (1 - done_i)
        alive = torch.cat([torch.ones_like(done[:, :1]), torch.cumprod(not_done, dim=1)[:, :-1]], dim=1)
        out["alive"], out["next_valid"] = alive, alive * not_done
        return out

    def sample_starts(self, batch: int, fields=None) -> dict[str, torch.Tensor]:
        """batch stored entries, uniform over (env, entry): (B, ...) of every field (default all)."""
        if self.size == 0:
            raise ValueError("the buffer is empty")
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        rows = (self._oldest() + torch.randint(0, self.size, (batch,), device=self.device)) % self.capacity
        return {name: self.data[name][rows, env] for name in (self.spec if fields is None else fields)}
