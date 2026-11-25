# Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0), src/supermimic/sim_context.py. Changes: the
# specs read Isaac Lab's articulation (PhysX shape material, COM offset from the USD COM, env.encoder_bias) instead of
# env.sim.model, the friction and COM once at bind (start-up events, checked) and cached; events are configclass
# attributes, not dict entries.
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
    def _startup_event(env: ManagerBasedRlEnv, name: str):
        """The event cfg ``name``, which must be start-up mode: its values are read once and cached."""
        event_cfg = getattr(env.cfg.events, name, None)
        if event_cfg is None:
            raise RuntimeError(f"{name}: no event '{name}' on env.cfg.events")
        if event_cfg.mode != "startup":
            raise RuntimeError(f"{name}: the event is mode={event_cfg.mode!r}; the cached read needs 'startup'")
        return event_cfg

    @staticmethod
    def _body_ids(asset, body_names) -> list[int]:
        """Articulation body indices of ``body_names``, in their order (PhysX views index bodies this way)."""
        return asset.find_bodies(list(body_names), preserve_order=True)[0]


class FootFrictionSpec(SimContextSpec):
    """Per-env tangential foot friction read from one matching collision shape.

    The ``foot_friction`` event writes one value per env to every shape of the
    foot bodies (mjlab's ``shared_random=True``); we read from the first match.
    Read once from PhysX in ``bind()`` and cached on the env device: this assumes
    a start-up-mode event (``bind()`` raises otherwise).
    """

    name = "foot_friction"
    dim = 1

    def __init__(self) -> None:
        self._shape_idx: int | None = None
        self._value: torch.Tensor | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = self._startup_event(env, self.name)
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
        self._shape_idx = matches[0]
        materials = asset.root_physx_view.get_material_properties()  # (E, shapes, 3), cpu
        self._value = materials[:, self._shape_idx : self._shape_idx + 1, 0].to(env.device)
        return [ranges]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("FootFrictionSpec.read() called before bind()")
        return self._value


class CoMOffsetSpec(SimContextSpec):
    """Per-env offset of the body's COM from its USD COM, in the body frame (mjlab's ``body_ipos``
    offset; Isaac Lab's ``randomize_rigid_body_com`` adds it to the PhysX COM). Read once from PhysX in ``bind()``
    and cached on the env device: this assumes a start-up-mode event (``bind()`` raises otherwise)."""

    name = "base_com"
    dim = 3

    def __init__(self) -> None:
        self._body_id: int | None = None
        self._value: torch.Tensor | None = None

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = self._startup_event(env, self.name)
        params = event_cfg.params
        ranges = params["com_range"]
        asset_cfg = params["asset_cfg"]
        if len(asset_cfg.body_names) != 1:
            raise ValueError(f"CoMOffsetSpec: expected exactly 1 body_name, got {asset_cfg.body_names!r}")
        asset = env.scene[asset_cfg.name]
        self._body_id = self._body_ids(asset, asset_cfg.body_names)[0]
        default_com = _usd_center_of_mass(asset.root_physx_view.link_paths[0][self._body_id])
        coms = asset.root_physx_view.get_coms()  # (E, bodies, 7) pos + quat, cpu
        self._value = (coms[:, self._body_id, :3] - default_com).to(env.device)
        return [ranges.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("CoMOffsetSpec.read() called before bind()")
        return self._value


class EncoderBiasSpec(SimContextSpec):
    name = "encoder_bias"
    dim = 0  # set in bind() from env.encoder_bias

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        event_cfg = getattr(env.cfg.events, self.name, None)
        if event_cfg is None:
            raise RuntimeError(f"EncoderBiasSpec: no event '{self.name}' on env.cfg.events")
        params = event_cfg.params
        self.dim = _encoder_bias(env).shape[1]
        if "half_widths" in params:  # main's per-joint offsets (events.sample_encoder_offsets)
            import re

            from pretraining.constants import JOINT_NAMES

            widths = [next((w for p, w in params["half_widths"].items() if re.fullmatch(p, j)), 0.0)
                      for j in JOINT_NAMES]
            return [(-w, w) for w in widths]
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


class HeldSpec(SimContextSpec):
    """A column block of a source run's layout whose event the current env does not have: it reads the source's
    mean (0 in normalised units), so the world model and critics see the nominal value they were trained around."""

    def __init__(self, name: str, dim: int, mean: torch.Tensor):
        self.name, self.dim, self._mean = name, dim, mean

    def bind(self, env):
        raise RuntimeError("HeldSpec is built bound")

    def read(self, env) -> torch.Tensor:
        return self._mean.to(env.device).expand(env.num_envs, -1)


_CANDIDATES = {"foot_friction": FootFrictionSpec, "base_com": CoMOffsetSpec, "encoder_bias": EncoderBiasSpec}


def layout_from_mean_scale(names_dims: list[tuple[str, int]], mean, scale) -> list[dict]:
    """A layout (``SimContext.layout``'s form) from spec names and widths and the normalisation buffers."""
    out, at = [], 0
    for name, dim in names_dims:
        m, sc = mean[at:at + dim].tolist(), scale[at:at + dim].tolist()
        out.append({"name": name, "dim": dim, "ranges": [[a - b, a + b] for a, b in zip(m, sc)]})
        at += dim
    return out


def layout_from_bundle(sim_context_state: dict, num_joints: int) -> list[dict]:
    """The layout of a saved world model's sim context (its ``mean`` / ``scale``) from before run.json recorded it:
    the specs are those of ``_CANDIDATES``, in that order, whose widths (1, 3, ``num_joints``) add up to its width."""
    mean, scale = sim_context_state["mean"].cpu(), sim_context_state["scale"].cpu()
    dims = {"foot_friction": 1, "base_com": 3, "encoder_bias": num_joints}
    names = list(_CANDIDATES)
    for mask in range(1, 1 << len(names)):
        chosen = [(n, dims[n]) for i, n in enumerate(names) if mask >> i & 1]
        if sum(d for _, d in chosen) == mean.numel():
            return layout_from_mean_scale(chosen, mean, scale)
    raise ValueError(f"a sim context {mean.numel()} wide matches no combination of {dims}")


class SimContext(nn.Module):
    def __init__(self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]):
        super().__init__()
        self.specs = list(specs)
        stats = [_mean_scale(lo, hi) for lo, hi in ranges]
        self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
        self.register_buffer("scale", torch.tensor([s for _, s in stats]))

    @classmethod
    def from_env(cls, env: ManagerBasedRlEnv, layout: list[dict] | None = None) -> SimContext:
        """Discover, build, bind, and validate from the env's active DR events.

        Spec order here is the column layout of the normalized context: the replay
        buffer's ``sim_ctx`` field and the Dynamics input.

        ``layout`` (``layout()`` of the run a fine-tune or resume starts from): that run's specs, widths and ranges
        instead, so its world model and critics keep their input. A spec whose event this env lacks reads the
        source's mean (``HeldSpec``); a live one keeps the source's normalisation, so a wider draw here reads
        beyond +-1.
        """
        if layout is not None:
            specs, ranges, held = [], [], []
            for entry in layout:
                name, dim, rng = entry["name"], int(entry["dim"]), [tuple(r) for r in entry["ranges"]]
                if getattr(env.cfg.events, name, None) is not None:
                    spec = _CANDIDATES[name]()
                    spec.bind(env)
                    if spec.dim != dim:
                        raise ValueError(f"sim context {name}: {spec.dim} columns here, {dim} in the source run")
                else:
                    mean = torch.tensor([_mean_scale(lo, hi)[0] for lo, hi in rng])
                    spec = HeldSpec(name, dim, mean)
                    held.append(name)
                specs.append(spec)
                ranges += rng
            if held:
                print(f"[INFO] Sim Context: {', '.join(held)} held at the source run's mean (normalised 0): this "
                      "env has no such event, the source run's world model and critics take the columns", flush=True)
            return cls(specs, ranges).to(env.device)
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

    def layout(self) -> list[dict]:
        """Each spec's name, width and per-column range (from the normalisation), for run.json and ``from_env``."""
        return layout_from_mean_scale([(s.name, s.dim) for s in self.specs], self.mean.cpu(), self.scale.cpu())

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
