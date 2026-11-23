# Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0), src/supermimic/sim_context.py. Changes: the
# specs read Isaac Lab's articulation (PhysX shape material, COM offset from the USD COM, env.encoder_bias) instead of
# env.sim.model; events are configclass attributes, not dict entries.
"""Per-env sim-context conditioning for the world model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRlEnv


_DEGENERATE_HALF_WIDTH = 1e-3
"""Half-widths at or below this are treated as constant. Float32 ULP for
O(1) values is ~1e-7; dividing ``(raw - mean)`` by ``half_width`` amplifies
that noise by ``1/half_width``. At 1e-3 the worst-case amplified noise is
~1e-4 — well below the [-1, 1] signal range. Below ~1e-6 the noise
overwhelms the signal entirely. All default DR ranges
(foot_friction ±0.45, base_com ±0.025–0.05, encoder_bias ±0.01) clear
this threshold."""


def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
    """Range → (mean, scale). Half-widths below the precision floor get
    scale=1.0 so ``(raw - mean) ≈ 0`` instead of amplifying float-ULP noise."""
    assert hi >= lo, f"empty range: lo={lo}, hi={hi}"
    half = (hi - lo) / 2.0
    return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


class SimContextSpec(ABC):
    name: str
    dim: int

    @abstractmethod
    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        """Resolve indices and ``dim``; return each column's DR range ``(lo, hi)``."""
        ...

    @abstractmethod
    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Return raw per-env values of shape ``(num_envs, dim)``."""
        ...

    @staticmethod
    def _body_ids(asset, body_names) -> list[int]:
        """Articulation body indices of ``body_names``, in their order (PhysX views index bodies this way)."""
        return asset.find_bodies(list(body_names), preserve_order=True)[0]


class FootFrictionSpec(SimContextSpec):
    """Per-env tangential foot friction read from one matching collision shape.

    The ``foot_friction`` event writes one value per env to every shape of the
    foot bodies (mjlab's ``shared_random=True``); we read from the first match.
    """

    name = "foot_friction"
    dim = 1

    def __init__(self) -> None:
        self._shape_idx: int | None = None
        self._asset_name: str | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = getattr(env.cfg.events, self.name, None)
        if event_cfg is None:
            raise RuntimeError(f"FootFrictionSpec: no event '{self.name}' on env.cfg.events")
        params = event_cfg.params
        ranges = params["friction_range"]

        asset_cfg = params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        if not asset_cfg.body_names:
            raise ValueError(f"FootFrictionSpec: '{self.name}' has empty body_names")
        from pretraining.tasks.events import _shape_slices  # Isaac Lab: imported with the env, not this module

        matches = _shape_slices(asset, self._body_ids(asset, asset_cfg.body_names))
        if not matches:
            raise ValueError(f"FootFrictionSpec: no shapes in '{asset_cfg.name}' on {asset_cfg.body_names!r}")
        self._asset_name = asset_cfg.name
        self._shape_idx = matches[0]
        return [ranges]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._shape_idx is None:
            raise RuntimeError("FootFrictionSpec.read() called before bind()")
        materials = env.scene[self._asset_name].root_physx_view.get_material_properties()  # (E, shapes, 3), cpu
        return materials[:, self._shape_idx : self._shape_idx + 1, 0].to(env.device)


class CoMOffsetSpec(SimContextSpec):
    """Per-env offset of the body's COM from its USD COM, in the body frame (mjlab's ``body_ipos``
    offset; Isaac Lab's ``randomize_rigid_body_com`` adds it to the PhysX COM)."""

    name = "base_com"
    dim = 3

    def __init__(self) -> None:
        self._body_id: int | None = None
        self._asset_name: str | None = None
        self._default_com: torch.Tensor | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = getattr(env.cfg.events, self.name, None)
        if event_cfg is None:
            raise RuntimeError(f"CoMOffsetSpec: no event '{self.name}' on env.cfg.events")
        params = event_cfg.params
        ranges = params["com_range"]
        asset_cfg = params["asset_cfg"]
        if len(asset_cfg.body_names) != 1:
            raise ValueError(f"CoMOffsetSpec: expected exactly 1 body_name, got {asset_cfg.body_names!r}")
        asset = env.scene[asset_cfg.name]
        self._asset_name = asset_cfg.name
        self._body_id = self._body_ids(asset, asset_cfg.body_names)[0]
        self._default_com = _usd_center_of_mass(asset.root_physx_view.link_paths[0][self._body_id])
        return [ranges.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._body_id is None:
            raise RuntimeError("CoMOffsetSpec.read() called before bind()")
        coms = env.scene[self._asset_name].root_physx_view.get_coms()  # (E, bodies, 7) pos + quat, cpu
        return (coms[:, self._body_id, :3] - self._default_com).to(env.device)


class EncoderBiasSpec(SimContextSpec):
    name = "encoder_bias"
    dim = 0  # set in bind() from env.encoder_bias

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = getattr(env.cfg.events, self.name, None)
        if event_cfg is None:
            raise RuntimeError(f"EncoderBiasSpec: no event '{self.name}' on env.cfg.events")
        params = event_cfg.params
        self.dim = _encoder_bias(env).shape[1]
        return [params["bias_range"]] * self.dim

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return _encoder_bias(env)


def _encoder_bias(env: ManagerBasedRlEnv) -> torch.Tensor:
    """``env.encoder_bias``, the ``encoder_bias`` event's draw (``pretraining.tasks.events``)."""
    from pretraining.tasks.events import encoder_bias  # Isaac Lab: imported with the env, not this module

    return encoder_bias(env)


def _usd_center_of_mass(prim_path: str) -> torch.Tensor:
    """The COM a body's USD authors (its ``UsdPhysics.MassAPI``), before any randomization."""
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    com = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
    if com is None or not all(abs(c) < 1e30 for c in com):  # unauthored: USD's sentinel is -inf
        raise RuntimeError(f"CoMOffsetSpec: {prim_path} authors no center of mass")
    return torch.tensor(list(com), dtype=torch.float32)


class SimContext(nn.Module):
    def __init__(self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]):
        super().__init__()
        self.specs = list(specs)
        stats = [_mean_scale(lo, hi) for lo, hi in ranges]
        self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
        self.register_buffer("scale", torch.tensor([s for _, s in stats]))

    @classmethod
    def from_env(cls, env: ManagerBasedRlEnv) -> SimContext:
        """Discover, build, bind, and validate from the env's active DR events.

        Spec order here is the column layout of the normalized context: the replay
        buffer's ``sim_ctx`` field and the Dynamics input.
        """
        candidates: tuple[type[SimContextSpec], ...] = (
            FootFrictionSpec,
            CoMOffsetSpec,
            EncoderBiasSpec,
        )
        specs = [c() for c in candidates if getattr(env.cfg.events, c.name, None) is not None]
        if not specs:
            raise RuntimeError(
                "No sim_ctx specs derived from env.cfg.events. "
                "Enable at least one DR event (foot_friction, base_com, encoder_bias) "
                "in the task cfg."
            )
        return cls(specs, [r for s in specs for r in s.bind(env)]).to(env.device)

    @property
    def output_dim(self) -> int:
        return sum(spec.dim for spec in self.specs)

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Normalized per-env context ``(num_envs, output_dim)``, in spec order."""
        return (torch.cat([s.read(env) for s in self.specs], -1) - self.mean) / self.scale

    def find(self, name: str) -> slice | None:
        """The columns spec ``name`` occupies in a normalized context."""
        offset = 0
        for spec in self.specs:
            if spec.name == name:
                return slice(offset, offset + spec.dim)
            offset += spec.dim
        return None

    def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
        """Raw values of ``columns`` in the normalized context ``sim_ctx``."""
        return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
