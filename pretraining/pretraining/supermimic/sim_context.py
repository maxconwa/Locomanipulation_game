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

from pretraining.constants import JOINT_NAMES

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

    @classmethod
    def available(cls, env: ManagerBasedRlEnv) -> bool:
        """Whether ``env`` draws this quantity (its event, by default ``env.cfg.events.<name>``)."""
        return getattr(env.cfg.events, cls.name, None) is not None

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
    """Per-env static foot friction: the mean over the collision shapes of the feet.

    The tracking env's ``foot_friction`` event writes one value per env to every shape of the foot bodies (mjlab's
    ``shared_random=True``): the first shape's is read. Main's ``physics_material`` event (the fine-tune env's, and
    main's own) draws each shape of every body on its own: the column then holds the mean of the feet's shapes, with
    ``static_friction_range``. Read once from PhysX in ``bind()`` and cached on the env device: this assumes a
    start-up-mode event (``bind()`` raises otherwise).
    """

    name = "foot_friction"
    dim = 1

    def __init__(self) -> None:
        self._value: torch.Tensor | None = None

    @classmethod
    def available(cls, env: ManagerBasedRlEnv) -> bool:
        return any(getattr(env.cfg.events, n, None) is not None for n in (cls.name, "physics_material"))

    def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
        shared = getattr(env.cfg.events, self.name, None) is not None
        if shared:
            params = self._startup_event(env, self.name).params
            ranges, asset_cfg = params["friction_range"], params["asset_cfg"]
            asset, bodies = env.scene[asset_cfg.name], asset_cfg.body_names
            if not bodies:
                raise ValueError(f"FootFrictionSpec: '{self.name}' has empty body_names")
        else:
            params = self._startup_event(env, "physics_material").params
            ranges, asset, bodies = params["static_friction_range"], env.scene[params["asset_cfg"].name], FEET
        self._value = feet_friction(asset, self._body_ids(asset, bodies), first_only=shared).to(env.device)
        return [tuple(ranges)]

    def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("FootFrictionSpec.read() called before bind()")
        return self._value


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
"""The feet (``track_env_cfg.FEET``; main's ``FOOT_LINK_NAMES``)."""


def feet_friction(asset, body_ids: list[int], first_only: bool = False) -> torch.Tensor:
    """``(num_envs, 1)`` cpu: the mean static friction of the collision shapes of ``body_ids`` (``first_only``: the
    first shape's)."""
    from pretraining.tasks.events import _shape_slices  # Isaac Lab: imported with the env, not this module

    shapes = _shape_slices(asset, body_ids)
    if not shapes:
        raise ValueError(f"FootFrictionSpec: no collision shapes on bodies {body_ids}")
    materials = asset.root_physx_view.get_material_properties()  # (E, shapes, 3), cpu
    return materials[:, shapes[:1] if first_only else shapes, 0].mean(dim=1, keepdim=True)


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


ACTION_DELAY_NORM = (5.0, 5.0)
"""(mean, half-width) of the ``action_delay`` column, physics substeps: main's ``ACTION_DELAY_RANGE`` (0, 10)."""


def delay_steps(env) -> torch.Tensor | None:
    """Each env's target delay (physics substeps, ``(num_envs,)``) or None if the env has none: the tracking env's
    ``TargetDelay`` term (``delay_steps``, drawn per reset from ``delay_range``) or main's ``LocoManipMarlEnv``
    (``_delay_steps``, drawn per reset from ``cfg.action_delay_range``)."""
    manager = getattr(env, "action_manager", None)
    if manager is not None:
        for name in manager.active_terms:
            term = manager.get_term(name)
            if hasattr(term, "delay_steps"):
                return term.delay_steps
    if getattr(env.cfg, "action_delay_range", None) is not None:
        return env._delay_steps
    return None


class ActionDelaySpec(SimContextSpec):
    """Per-env target delay in physics substeps, drawn at every reset (read live). Normalised with the fixed
    ``ACTION_DELAY_NORM`` (0-10 substeps -> -1..1), not the env's range."""

    name = "action_delay"
    dim = 1

    @classmethod
    def available(cls, env) -> bool:
        return delay_steps(env) is not None

    def bind(self, env) -> list[tuple[float, float]]:
        m, h = ACTION_DELAY_NORM
        return [(m - h, m + h)]

    def read(self, env) -> torch.Tensor:
        return delay_steps(env).float().unsqueeze(-1)


class _ActuatorScaleSpec(SimContextSpec):
    """Per env and policy joint (``JOINT_NAMES`` order), a factor of main's ``randomize_actuator_scales`` reset event
    (``actuator_scales``), read live from the explicit actuators against the nominal values the event keeps
    (``env._actuator_nominal``): kp and kd are both nominal x gain x strength and the effort limit nominal x strength.
    Before the event's first draw the actuators are nominal (1). Normalised with the event's own ranges."""

    dim = len(JOINT_NAMES)

    @classmethod
    def available(cls, env) -> bool:
        return getattr(env.cfg.events, "actuator_scales", None) is not None

    def bind(self, env) -> list[tuple[float, float]]:
        import re

        from isaaclab.actuators import IdealPDActuator

        params = env.cfg.events.actuator_scales.params
        robot = env.scene[params.get("asset_cfg").name if params.get("asset_cfg") is not None else "robot"]
        where = {}
        for act_name, act in robot.actuators.items():
            if isinstance(act, IdealPDActuator):
                for col, joint in enumerate(act.joint_names):
                    where[joint] = (act_name, col)
        missing = [j for j in JOINT_NAMES if j not in where]
        if missing:
            raise ValueError(f"{self.name}: no explicit PD actuator drives {missing}")
        self._robot, self._where = robot, [where[j] for j in JOINT_NAMES]
        if self.name == "actuator_strength":
            return [tuple(params["strength_range"])] * self.dim
        ranges = []
        for joint in JOINT_NAMES:
            rng = params["gain_range"]
            for pattern, r in (params.get("joint_gain_ranges") or {}).items():
                if re.fullmatch(pattern, joint):
                    rng = r
            ranges.append(tuple(rng))
        return ranges

    def read(self, env) -> torch.Tensor:
        nominal = getattr(env, "_actuator_nominal", None)
        if nominal is None:
            return torch.ones(env.num_envs, self.dim, device=env.device)
        cols = []
        for act_name, col in self._where:
            act = self._robot.actuators[act_name]
            kp, _, limit = nominal[act_name]
            strength = act.effort_limit[:, col] / limit[:, col]
            if self.name == "actuator_strength":
                cols.append(strength)
            else:
                cols.append(act.stiffness[:, col] / (kp[:, col] * strength))
        return torch.stack(cols, dim=-1)


class ActuatorGainSpec(_ActuatorScaleSpec):
    """kp and kd's common gain factor (one draw scales both: their scales are the same number, so one column block
    holds the stiffness and the damping scale), per joint."""

    name = "actuator_gain"


class ActuatorStrengthSpec(_ActuatorScaleSpec):
    """The strength factor (effort limit; kp and kd are scaled by it too), per joint."""

    name = "actuator_strength"


class TorsoMassSpec(SimContextSpec):
    """The mass (kg) main's ``add_torso_mass`` start-up event added to the torso, against the asset's default mass.
    Read once in ``bind()`` and cached. Normalised with the event's range."""

    name = "torso_mass"
    dim = 1

    @classmethod
    def available(cls, env) -> bool:
        return getattr(env.cfg.events, "add_torso_mass", None) is not None

    def bind(self, env) -> list[tuple[float, float]]:
        params = self._startup_event(env, "add_torso_mass").params
        if params.get("operation", "add") != "add":
            raise ValueError(f"{self.name}: add_torso_mass is operation={params['operation']!r}, not 'add'")
        asset_cfg = params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        bodies = [asset_cfg.body_names] if isinstance(asset_cfg.body_names, str) else asset_cfg.body_names
        self._value = torso_mass_offset(asset, self._body_ids(asset, bodies)[0]).to(env.device)
        return [tuple(params["mass_distribution_params"])]

    def read(self, env) -> torch.Tensor:
        return self._value


def torso_mass_offset(asset, body_id: int) -> torch.Tensor:
    """``(num_envs, 1)`` cpu: the body's PhysX mass less the asset's default (pre-randomisation) mass."""
    masses = asset.root_physx_view.get_masses()  # (E, bodies), cpu
    return (masses[:, body_id] - asset.data.default_mass[:, body_id].cpu()).unsqueeze(-1)


class HeldSpec(SimContextSpec):
    """A column block of a source run's layout whose event the current env does not have: it reads the source's
    mean (0 in normalised units), so the world model and critics see the nominal value they were trained around."""

    def __init__(self, name: str, dim: int, mean: torch.Tensor):
        self.name, self.dim, self._mean = name, dim, mean

    def bind(self, env):
        raise RuntimeError("HeldSpec is built bound")

    def read(self, env) -> torch.Tensor:
        return self._mean.to(env.device).expand(env.num_envs, -1)


_CANDIDATES = {"foot_friction": FootFrictionSpec, "base_com": CoMOffsetSpec, "encoder_bias": EncoderBiasSpec,
               "action_delay": ActionDelaySpec, "actuator_gain": ActuatorGainSpec,
               "actuator_strength": ActuatorStrengthSpec, "torso_mass": TorsoMassSpec}
"""Every spec, in column order. ``layout_from_bundle`` infers an old run's layout from the first three only."""
_LEGACY = ("foot_friction", "base_com", "encoder_bias")


def _wider(source: tuple[float, float], here: tuple[float, float]) -> tuple[float, float]:
    """``here`` if it is wider than ``source`` about the same centre, else ``source``."""
    (slo, shi), (hlo, hhi) = source, here
    same_centre = abs((slo + shi) - (hlo + hhi)) < 2e-9
    return (float(hlo), float(hhi)) if same_centre and hhi - hlo > shi - slo + 1e-9 else tuple(source)


def widen_input_weight(weight: torch.Tensor, src_mean: torch.Tensor, src_scale: torch.Tensor,
                       tgt_mean: torch.Tensor, tgt_scale: torch.Tensor) -> torch.Tensor:
    """A first layer's weight whose input ends with a source sim context (``src_mean`` / ``src_scale``), for the
    target context (``tgt_*``: the source's columns first, possibly renormalised about the same means, then appended
    ones). The source columns are rescaled by ``tgt_scale / src_scale`` (a column normalised over a wider range reads
    smaller for the same raw value) and the appended columns get zero weights: the layer's output is the source's
    for the same raw values, whatever the appended columns hold. Nothing else (the bias) changes."""
    n_src, n_tgt = src_mean.numel(), tgt_mean.numel()
    if n_tgt < n_src or not torch.allclose(tgt_mean[:n_src].cpu(), src_mean.cpu(), atol=1e-6):
        raise ValueError(f"the sim context ({n_tgt} columns) does not start with the source's {n_src} at their means")
    head = weight[:, : weight.shape[1] - n_src]
    ctx = weight[:, weight.shape[1] - n_src:] * (tgt_scale[:n_src] / src_scale).to(weight)
    return torch.cat([head, ctx, weight.new_zeros(weight.shape[0], n_tgt - n_src)], dim=1)


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
    names = list(_LEGACY)
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
    def from_env(cls, env: ManagerBasedRlEnv, layout: list[dict] | None = None, fine_tune: bool = False) -> SimContext:
        """Discover, build, bind, and validate from the env's active DR events.

        Spec order here is the column layout of the normalized context: the replay
        buffer's ``sim_ctx`` field and the Dynamics input.

        ``layout`` (``layout()`` of the run a fine-tune or resume starts from): that run's specs, widths and ranges,
        in its order, so its world model and critics keep their input. A spec this env does not draw reads the
        source's mean (``HeldSpec``). A resume (``fine_tune`` False) keeps the layout exactly. A fine-tune
        (``fine_tune``) also widens a live column's normalisation to this env's range where that is wider about the
        same centre (main's encoder offsets), and appends, after the source's columns, every spec this env draws that
        the source lacks: ``widen_input_weight`` adapts the loaded first layers so the source's function is kept.
        A column whose range here is wider about another centre keeps the source's normalisation (it reads beyond
        +-1).
        """
        if layout is not None:
            specs, ranges, held, renormalised = [], [], [], []
            for entry in layout:
                name, dim, rng = entry["name"], int(entry["dim"]), [tuple(r) for r in entry["ranges"]]
                spec_cls = _CANDIDATES.get(name)
                if spec_cls is not None and spec_cls.available(env):
                    spec = spec_cls()
                    live = spec.bind(env)
                    if spec.dim != dim:
                        raise ValueError(f"sim context {name}: {spec.dim} columns here, {dim} in the source run")
                    if fine_tune:
                        widened = [_wider(src, here) for src, here in zip(rng, live)]
                        if widened != rng:
                            renormalised.append(name)
                        rng = widened
                else:
                    mean = torch.tensor([_mean_scale(lo, hi)[0] for lo, hi in rng])
                    spec = HeldSpec(name, dim, mean)
                    held.append(name)
                specs.append(spec)
                ranges += rng
            appended = []
            if fine_tune:
                have = {e["name"] for e in layout}
                for name, spec_cls in _CANDIDATES.items():
                    if name not in have and spec_cls.available(env):
                        spec = spec_cls()
                        ranges += spec.bind(env)
                        specs.append(spec)
                        appended.append(name)
            print(f"[INFO] Sim Context: appended {', '.join(appended) or 'none'} (zero-initialised weights); "
                  f"{', '.join(held) or 'none'} held at the source run's mean (normalised 0); renormalised "
                  f"{', '.join(renormalised) or 'none'} to this env's range", flush=True)
            return cls(specs, ranges).to(env.device)
        specs = [c() for c in _CANDIDATES.values() if c.available(env)]
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
