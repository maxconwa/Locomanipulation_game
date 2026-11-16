"""LocoManip-VlaUpper-v0's own terms: the arms on delta end-effector commands, the Magpie grippers on one open/close
command each, the blocks' sizes and colours, and the arms' proprioception.

Imported by vla_upper_env_cfg.py only (not in the `mdp` namespace): the other tasks never load it.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .actions import _BoundedTargets
from .arm_kinematics import ArmKinematics, quat_from_matrix, rotation_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# the Magpie's grasp point in the wrist_yaw_link (the URDF's *_grasp_frame): x is the approach direction, the jaws
# close along y
TCP_OFFSET = (0.231991, 0.0, 0.000625)
# crank angle (rad) at which the open jaws (0) have closed on each other
GRIPPER_THETA_CLOSED = 1.45


def rotation_from_axis_angle(v: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices of axis-angle vectors (..., 3) (Rodrigues)."""
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / angle.clamp(min=1e-9)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1), torch.stack([-y, x, zero], -1)], -2)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return torch.eye(3, device=v.device, dtype=v.dtype) + s * K + (1.0 - c) * (K @ K)


class TcpKinematics:
    """ArmKinematics with the grasp point (TCP_OFFSET beyond each wrist) as the end effector: its pose, its Jacobian
    and a damped-least-squares solve, all in the pelvis frame. Plain torch, as ArmKinematics."""

    def __init__(self, urdf: str, spheres: str, device: str, tcp_offset: Sequence[float] = TCP_OFFSET):
        self.kin = ArmKinematics(urdf, spheres, device=device)
        self.offset = torch.tensor(tcp_offset, device=device, dtype=self.kin.dtype)

    def links(self, q: torch.Tensor):
        """q (N, 2, 7) -> the arm links' positions and rotations and the grasp points (N, 2, 3)."""
        p, R = self.kin._arm_links(q)
        return p, R, p[:, :, 6] + (R[:, :, 6] @ self.offset)

    def poses(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q (N, 2, 7) -> the grasp points (N, 2, 3) and their rotations (N, 2, 3, 3)."""
        _, R, tcp = self.links(q)
        return tcp, R[:, :, 6]

    def step(self, q: torch.Tensor, goal_p: torch.Tensor, goal_R: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
             posture: torch.Tensor | None = None, damping: float = 0.05, rot_weight: float = 0.3,
             null_gain: float = 0.0, max_step: float = 0.2, passes: int = 3, null_damping: float = 1e-3) -> torch.Tensor:
        """One damped-least-squares step of q (N, 2, 7) toward the grasp-point goals, on the error [position,
        rot_weight x axis-angle], with a pull of null_gain x (posture - q) in the task's null space (projected with
        null_damping, not the step's damping: with that one the pull leaked into the task wherever the Jacobian is
        weak, the grasp point crept toward the ready posture by itself, 2 mm a policy step, and stopped 1-2 cm short
        of goals approached slowly); capped at max_step per joint and kept inside [low, high]. A joint whose step would take it past its bound is locked
        there and the step solved again without it, up to `passes` times (as IKPostureArmAction._step), so the
        other joints take over the motion a joint at its limit can't give."""
        p, R, tcp = self.links(q)
        W = torch.tensor([1, 1, 1, rot_weight, rot_weight, rot_weight], device=q.device, dtype=q.dtype)
        err = (torch.cat([goal_p - tcp, rotation_error(goal_R, R[:, :, 6])], -1) * W).unsqueeze(-1)
        z = (R @ self.kin.axes.unsqueeze(-1)).squeeze(-1)
        J = torch.cat([torch.cross(z, tcp.unsqueeze(2) - p, dim=-1), z], -1).transpose(-1, -2) * W.view(6, 1)
        eye = torch.eye(6, device=q.device, dtype=q.dtype) * damping ** 2
        pull = null_gain * (posture - q) if posture is not None and null_gain > 0.0 else None
        eye_null = torch.eye(6, device=q.device, dtype=q.dtype) * null_damping ** 2
        free = torch.ones_like(q, dtype=torch.bool)
        for _ in range(passes):
            f = free.to(q.dtype)
            Jf = J * f.unsqueeze(-2)
            JJt = Jf @ Jf.transpose(-1, -2) + eye
            dq = (Jf.transpose(-1, -2) @ torch.linalg.solve(JJt, err)).squeeze(-1)
            if pull is not None:
                pf = pull * f
                dq = dq + pf - (Jf.transpose(-1, -2) @ torch.linalg.solve(JJt - eye + eye_null, Jf @ pf.unsqueeze(-1))).squeeze(-1)
            dq = (dq * f).clamp(-max_step, max_step)
            new = q + dq
            hit = ((new < low) & (dq < 0)) | ((new > high) & (dq > 0))
            if not (hit & free).any():
                break
            free = free & ~hit
        return torch.maximum(torch.minimum(new, high), low)

    def solve(self, goal_p, goal_R, q0, low, high, iters: int = 60, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint angles for the grasp-point goals from q0, and the position (m) and rotation (rad) errors left."""
        q = q0.clone()
        for _ in range(iters):
            q = self.step(q, goal_p, goal_R, low, high, **kwargs)
        tcp, R = self.poses(q)
        return q, (goal_p - tcp).norm(dim=-1), rotation_error(goal_R, R).norm(dim=-1)


class DeltaEEArmAction(_BoundedTargets):
    """Arm joint targets from delta end-effector commands: classical IK under joint PD tracking.

    The action is, per arm, a position delta and a rotation delta (axis-angle) of the grasp point in the pelvis frame,
    each in [-1, 1] times pos_scale / rot_scale per policy step. The term keeps its own commanded joint angles; each
    policy step the grasp point they give (forward kinematics) plus the delta is the goal, held within max_lead of the
    measured grasp point (so pushing on something doesn't wind the command up), and ik_iters damped-least-squares
    steps move the commanded angles toward it, at most max_joint_step per joint, inside the target bounds, with a
    null-space pull to the rest posture. A zero action holds the command. The joint targets are the commanded angles
    plus each joint's gravity torque over its nominal kp (ArmKinematics.gravity_torques), reached linearly over the
    policy step's physics substeps and bounded as the other arm terms' (_BoundedTargets); the motors' PD tracks them.
    """

    cfg: DeltaEEArmActionCfg

    def __init__(self, cfg: DeltaEEArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.tcp = TcpKinematics(cfg.kinematics_urdf, cfg.kinematics_spheres, self.device, cfg.tcp_offset)
        robot = self._asset
        self._rest = robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._kp = robot.data.default_joint_stiffness[:, self._joint_ids].clone()
        self._q_cmd = robot.data.joint_pos[:, self._joint_ids].clone()
        self._start = self._q_cmd.clone()   # the target at the policy step's start
        self._end = self._q_cmd.clone()     # and at its last substep
        self._substep = 0
        self._low = (self._lower + cfg.limit_margin).view(-1, 2, 7)
        self._high = (self._upper - cfg.limit_margin).view(-1, 2, 7)
        self._describe(f"delta end-effector IK ({cfg.pos_scale} m, {cfg.rot_scale} rad per step, {cfg.ik_iters} DLS"
                       f" steps, lead <= {cfg.max_lead} m); ")

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def q_cmd(self) -> torch.Tensor:
        """(N, 2, 7): the commanded arm joint angles."""
        return self._q_cmd.view(-1, 2, 7)

    def measured_tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The grasp points (N, 2, 3) and rotations (N, 2, 3, 3) at the measured joint angles, pelvis frame."""
        return self.tcp.poses(self._asset.data.joint_pos[:, self._joint_ids].view(-1, 2, 7))

    def commanded_tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tcp.poses(self.q_cmd)

    def process_actions(self, actions: torch.Tensor):
        cfg = self.cfg
        self._raw_actions[:] = actions
        a = actions.detach().clamp(-1.0, 1.0).view(-1, 2, 6)
        q = self.q_cmd
        cmd_p, cmd_R = self.tcp.poses(q)
        meas_p, _ = self.measured_tcp()
        lead = cmd_p + a[..., :3] * cfg.pos_scale - meas_p
        goal_p = meas_p + lead * (cfg.max_lead / lead.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        goal_R = rotation_from_axis_angle(a[..., 3:] * cfg.rot_scale) @ cmd_R
        new = q
        for _ in range(cfg.ik_iters):
            new = self.tcp.step(new, goal_p, goal_R, self._low, self._high, posture=self._rest.view(-1, 2, 7),
                                damping=cfg.ik_damping, rot_weight=cfg.rot_weight, null_gain=cfg.null_gain)
        new = q + (new - q).clamp(-cfg.max_joint_step, cfg.max_joint_step)
        self._q_cmd = new.reshape(self.num_envs, -1)
        gravity_b = self._asset.data.projected_gravity_b * abs(self._env.sim.cfg.gravity[2])
        tau = self.tcp.kin.gravity_torques(new, gravity_b).reshape(self.num_envs, -1)
        self._start = self._end
        self._end = self._q_cmd - tau / self._kp
        self._substep = 0

    def apply_actions(self):
        self._substep += 1
        frac = min(self._substep / self._env.cfg.decimation, 1.0)
        self._apply_bounds(self._start + (self._end - self._start) * frac)
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._q_cmd[ids] = q
        self._start[ids] = q
        self._end[ids] = q


@configclass
class DeltaEEArmActionCfg(JointPositionActionCfg):
    """joint_names: both arms' joints, left then right, each in kinematic order (scale and offset are unused)."""

    class_type: type = DeltaEEArmAction

    use_default_offset: bool = False
    preserve_order: bool = True
    kinematics_urdf: str = MISSING
    kinematics_spheres: str = MISSING
    tcp_offset: tuple[float, float, float] = TCP_OFFSET
    pos_scale: float = 0.03
    """m of grasp-point motion per policy step per action unit."""
    rot_scale: float = 0.15
    """rad of grasp-point rotation per policy step per action unit."""
    max_lead: float = 0.04
    """m: the goal's largest distance from the measured grasp point."""
    ik_iters: int = 4
    ik_damping: float = 0.05
    rot_weight: float = 0.3
    """The rotation error's weight against position's (1 per m), as IKPostureArmAction's."""
    null_gain: float = 0.1
    """Share of the arm's distance from the ready posture the IK closes per step in the task's null space. Strong
    enough that the joint angles settle to a function of the grasp point's pose within half a second: with 0.02 the
    arm's configuration depended on its path, and paths ended against the wrist limits."""
    max_joint_step: float = 0.1
    """rad per joint per policy step."""
    limit_margin: float = 0.05
    """rad inside the target bounds where the IK stops."""
    target_margin: float = MISSING
    joint_limits: dict[str, tuple[float, float]] | None = None
    torque_headroom: float = MISSING


class MagpieGripperAction(ActionTerm):
    """Both Magpie grippers, one command each: -1 open, +1 closed.

    Each finger is a parallelogram four-bar (crank, finger, rocker) whose closing joint the USD lacks, so its three
    hinges are driven together. The crank's target moves toward command x theta_closed at max_speed, held within `squeeze`
    of the two jaws' mean measured angle (the grip force, and what centres the jaws). The finger's and rocker's stiff drives follow the measured crank angle, minus and plus it,
    at every physics substep: the pad stays parallel to the gripper's axis wherever the crank stops, as the real
    linkage keeps it.
    """

    cfg: MagpieGripperActionCfg
    _asset: Articulation

    def __init__(self, cfg: MagpieGripperActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        # per gripper (left, right) and finger: the crank, finger and rocker joints, and the crank's closing sign
        self._crank, self._finger, self._rocker, signs = [], [], [], []
        for g in cfg.gripper_prefixes:
            for side, sign in (("left", 1.0), ("right", -1.0)):
                self._crank.append(robot.find_joints(f"{g}_{side}_hinge_1")[0][0])
                self._finger.append(robot.find_joints(f"{g}_{side}_hinge_2")[0][0])
                self._rocker.append(robot.find_joints(f"{g}_{side}_hinge_3")[0][0])
                signs.append(sign)
        self._sign = torch.tensor(signs, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._closure = torch.zeros(self.num_envs, len(cfg.gripper_prefixes), device=self.device)  # commanded, 0-1
        self._theta = torch.zeros(self.num_envs, len(signs), device=self.device)  # the cranks' targets (rad, >= 0)

    @property
    def action_dim(self) -> int:
        return len(self.cfg.gripper_prefixes)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._closure

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._raw_actions

    def closure(self) -> torch.Tensor:
        """(N, grippers): each gripper's measured closure, its cranks' mean angle over theta_closed (0 open)."""
        theta = self._asset.data.joint_pos[:, self._crank] * self._sign
        return theta.view(self.num_envs, -1, 2).mean(dim=2) / self.cfg.theta_closed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._closure = (actions.detach().clamp(-1.0, 1.0) + 1.0) * 0.5

    def apply_actions(self):
        cfg, robot = self.cfg, self._asset
        goal = self._closure.repeat_interleave(2, dim=1) * cfg.theta_closed
        step = cfg.max_speed * self._env.physics_dt
        self._theta = self._theta + (goal - self._theta).clamp(-step, step)
        measured = robot.data.joint_pos[:, self._crank]
        # closing, neither crank's target runs more than `squeeze` ahead of the gripper's mean measured angle: the
        # jaws stop on what they hold with that much drive error (the grip), and one ahead of the other is driven
        # less, so they centre. With each crank simply driven to the command, both drives saturated on a block and
        # nothing centred it: it sat up to 3 cm to one side of the grasp point.
        angle = measured * self._sign
        mean = angle.view(self.num_envs, -1, 2).mean(dim=2).repeat_interleave(2, dim=1)
        target = torch.minimum(self._theta, mean + cfg.squeeze)
        robot.set_joint_position_target(target * self._sign, joint_ids=self._crank)
        robot.set_joint_position_target(-measured, joint_ids=self._finger)
        robot.set_joint_position_target(measured, joint_ids=self._rocker)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._closure[ids] = 0.0
        self._theta[ids] = 0.0


@configclass
class MagpieGripperActionCfg(ActionTermCfg):
    class_type: type = MagpieGripperAction

    gripper_prefixes: tuple[str, ...] = ("lg", "rg")
    """The grippers' joint-name prefixes, in the arms' order (left, right)."""
    theta_closed: float = GRIPPER_THETA_CLOSED
    max_speed: float = 3.0
    """rad/s of the cranks' target (open to closed in 0.5 s)."""
    squeeze: float = 0.08
    """rad: the cranks' targets' lead over the jaws' mean measured angle when closing (x the drive stiffness: the
    grip torque)."""


"""
Blocks.
"""

BLOCK_COLORS = {"red": (0.80, 0.08, 0.08), "green": (0.10, 0.60, 0.15), "blue": (0.10, 0.20, 0.80),
                "yellow": (0.90, 0.80, 0.10), "orange": (1.00, 0.35, 0.00), "purple": (0.50, 0.15, 0.70)}
BLOCK_SIZE_NAMES = ("small", "medium", "large")
BLOCK_SIZE_RANGES = ((0.030, 0.042), (0.045, 0.057), (0.060, 0.070))  # m, each name's edge lengths
BLOCK_SLOT_CLASSES = (0, 0, 1, 1, 2, 2)  # each env's block slots: two of each size name


def block_layout(num_envs: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Each env's blocks, fixed for the run: (sizes (N, slots) m, colours (N, slots) indices into BLOCK_COLORS).
    A slot's size is uniform in its name's range; two slots of the same size name never share a colour, slots of
    different size names may, so an instruction needs both words. From the seed alone: the prestartup event that
    scales and colours the prims and the env both call this."""
    g = torch.Generator().manual_seed(seed)
    classes = torch.tensor(BLOCK_SLOT_CLASSES)
    low = torch.tensor([BLOCK_SIZE_RANGES[c][0] for c in BLOCK_SLOT_CLASSES])
    high = torch.tensor([BLOCK_SIZE_RANGES[c][1] for c in BLOCK_SLOT_CLASSES])
    sizes = low + (high - low) * torch.rand(num_envs, len(classes), generator=g)
    colors = torch.zeros(num_envs, len(classes), dtype=torch.long)
    for c in range(len(BLOCK_SIZE_NAMES)):
        slots = (classes == c).nonzero().flatten()
        for e in range(num_envs):
            colors[e, slots] = torch.randperm(len(BLOCK_COLORS), generator=g)[:len(slots)]
    return sizes, colors


def setup_blocks(env, env_ids, block_names: list[str], seed: int = 0):
    """Prestartup event: scale each env's unit-cube block prims to block_layout's sizes and colour their materials.
    Before the simulation starts: PhysX reads a collider's scale once."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    sizes, colors = block_layout(env.scene.num_envs, seed)
    palette = list(BLOCK_COLORS.values())
    for slot, name in enumerate(block_names):
        paths = sim_utils.find_matching_prim_paths(env.scene[name].cfg.prim_path)
        with Sdf.ChangeBlock():
            for e, path in enumerate(paths):
                spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), path)
                scale = spec.GetAttributeAtPath(path + ".xformOp:scale")
                if scale is None:
                    scale = Sdf.AttributeSpec(spec, path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
                    order = spec.GetAttributeAtPath(path + ".xformOpOrder")
                    if order is None:
                        order = Sdf.AttributeSpec(spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
                    order.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
                scale.default = Gf.Vec3f(*[float(sizes[e, slot])] * 3)
        for e, path in enumerate(paths):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if prim.IsA(UsdShade.Shader):
                    UsdShade.Shader(prim).GetInput("diffuseColor").Set(Gf.Vec3f(*palette[int(colors[e, slot])]))


def block_shader_inputs(env, block_names: list[str]) -> list[list[list]]:
    """[slot][env]: the diffuse-colour inputs of each block's shaders, for color_blocks."""
    from pxr import Usd, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    inputs = []
    for name in block_names:
        paths = sim_utils.find_matching_prim_paths(env.scene[name].cfg.prim_path)
        inputs.append([[UsdShade.Shader(prim).GetInput("diffuseColor") for prim in Usd.PrimRange(stage.GetPrimAtPath(path))
                        if prim.IsA(UsdShade.Shader)] for path in paths])
    return inputs


def color_blocks(shader_inputs: list[list[list]], env_ids: list[int], colors: torch.Tensor):
    """Colour the blocks of the envs env_ids while the simulation runs: colors (N, slots) indices into BLOCK_COLORS."""
    from pxr import Gf, Sdf

    palette = [Gf.Vec3f(*rgb) for rgb in BLOCK_COLORS.values()]
    chosen = colors.tolist()
    with Sdf.ChangeBlock():
        for slot, per_env in enumerate(shader_inputs):
            for e in env_ids:
                for shader_input in per_env[e]:
                    shader_input.Set(palette[chosen[e][slot]])


def make_matte(prim_path: str):
    """No specular reflection on the preview surfaces under prim_path: lamps and the dome leave no sheen on it."""
    from pxr import Gf, Sdf, Usd, UsdShade

    from isaaclab.sim.utils import get_current_stage

    for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(prim_path)):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)


@configclass
class VisualRandomizationCfg:
    """VisualRandomizer's ranges."""

    seed: int = 0
    texture_dir: str = ""
    """A folder of <name>.jpg and index.json (name: tags), e.g. Poly Haven's diffuse maps. Empty: plain colours."""
    texture_skip_tags: tuple[str, ...] = ("terrain", "aerial", "collection: moon")
    table_texture_tags: tuple[str, ...] = ("wood", "fabric", "metal", "plaster", "concrete", "tiles", "leather")
    block_texture_tags: tuple[str, ...] = ("wood", "fabric", "plaster", "concrete", "metal")
    num_table_textures: int = 96
    num_block_textures: int = 32
    """How many of the folder's textures are used: each one loaded stays in the GPU's memory."""
    table_texture_share: float = 0.8
    table_color_range: tuple[float, float] = (0.15, 0.85)
    """The plain table's colour, each channel."""
    table_brightness_range: tuple[float, float] = (0.6, 1.3)
    table_roughness_range: tuple[float, float] = (0.15, 0.95)
    table_metallic_range: tuple[float, float] = (0.0, 0.3)
    table_texture_scale_range: tuple[float, float] = (3.0, 12.0)
    """Repeats of the texture over the prim's unit cube."""
    block_color_gain_range: tuple[float, float] = (0.8, 1.15)
    block_color_shift: float = 0.04
    block_texture_share: float = 0.4
    block_texture_level: float = 0.7
    """A block's texture is turned grey and scaled to this mean (linear) before its colour tints it: a dark texture
    left as it is made a blue block black."""
    block_texture_scale_range: tuple[float, float] = (1.0, 3.0)
    block_roughness_range: tuple[float, float] = (0.05, 0.9)
    block_metallic_share: float = 0.3
    block_metallic_range: tuple[float, float] = (0.2, 0.5)
    """Without sim.render.enable_reflections a fully metallic block reads as black, whatever its colour (there is
    nothing for it to mirror); with them it looks right."""
    lamp_share: float = 0.85
    lamp_radius: float = 0.2
    """m. Open: a lamp's highlight on a glossy surface comes out as bright squares in one wrist camera's images (not
    the other's, nor the head's) in some launches, with reflections on or off, at 0.08 m too; more samples per pixel
    and the denoiser did not end it. Without lamps (lamp_share 0) it was never seen."""
    lamp_intensity_range: tuple[float, float] = (60000.0, 250000.0)
    """Beside the dome at dome_intensity, from lamp_z_range's heights: brighter, the table is near white and the
    blocks' colours pale."""
    dome_intensity: float | None = 500.0
    """The shared dome light's (the task's 1500 leaves a lamp little to add); None keeps it."""
    dome_path: str = "/World/light"
    lamp_excludes: tuple[str, ...] = ("/World/GreenFloor",)
    """Prims the lamps do not light. The renderer still drew their sheen on the chroma floor: make_matte ends it."""
    lamp_temperature_range: tuple[float, float] = (2700.0, 7500.0)
    lamp_x_range: tuple[float, float] = (-1.0, 1.6)
    lamp_y_range: tuple[float, float] = (-1.6, 1.6)
    lamp_z_range: tuple[float, float] = (2.2, 3.0)
    """m from the env's origin. Lower (1.5 m), a lifted block came near enough to the lamp to burn out to white."""


class VisualRandomizer:
    """Each episode's look, drawn at its reset: the table's surface, the blocks' finish and a lamp of the env's own.

    The table and every block get an OmniPBR material of their own (the spawned preview surfaces take a colour and
    nothing else), with its texture projected in the prim's own frame (the cubes have no texture coordinates):
    - the table: a texture of `table_textures` or a plain colour, its brightness, roughness and a little metal;
    - a block: its name's colour (a little off it), its roughness from mirror-like to matt, metallic in a share of
      the draws, and in another share a texture of `block_textures` turned grey and tinted that colour, so the name
      still tells it;
    - the lamp: a sphere light over the table at a drawn place, strength and colour temperature, beside the dome
      light that all envs share (the next env, env_spacing away, gets a few percent of it).
    """

    def __init__(self, env, cfg: VisualRandomizationCfg, block_names: list[str], table_name: str = "table"):
        import json
        import os

        import omni.kit.commands
        from pxr import Gf, Sdf, UsdLux, UsdShade

        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import get_current_stage

        self.cfg, self.num_envs = cfg, env.scene.num_envs
        self._gf, self._sdf = Gf, Sdf
        self.generator = torch.Generator().manual_seed(cfg.seed)
        index = json.load(open(os.path.join(cfg.texture_dir, "index.json"))) if cfg.texture_dir else {}

        def pick(wanted: tuple[str, ...], count: int) -> list[str]:
            names = [name for name, tags in sorted(index.items())
                     if set(wanted) & set(tags) and not set(cfg.texture_skip_tags) & set(tags)]
            spread = names[:: max(1, len(names) // max(count, 1))][:count]  # across the folder, not its first names
            return [os.path.join(cfg.texture_dir, name + ".jpg") for name in spread]

        self.table_textures = pick(cfg.table_texture_tags, cfg.num_table_textures)
        self.block_textures = pick(cfg.block_texture_tags, cfg.num_block_textures)
        from PIL import Image

        means = [(torch.tensor(list(Image.open(path).convert("L").resize((64, 64)).getdata())) / 255.0) ** 2.2
                 for path in self.block_textures]
        self.block_texture_gain = {path: cfg.block_texture_level / max(float(mean.mean()), 0.05)
                                   for path, mean in zip(self.block_textures, means)}
        stage = get_current_stage()

        def material(prim_path: str) -> dict:
            path = prim_path + "/pbr"
            omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=path)
            shader = UsdShade.Shader(stage.GetPrimAtPath(path + "/Shader"))
            types = Sdf.ValueTypeNames
            inputs = {name: shader.CreateInput(name, kind) for name, kind in (
                ("diffuse_color_constant", types.Color3f), ("diffuse_texture", types.Asset), ("diffuse_tint", types.Color3f),
                ("albedo_desaturation", types.Float), ("albedo_brightness", types.Float),
                ("reflection_roughness_constant", types.Float), ("metallic_constant", types.Float),
                ("specular_level", types.Float), ("texture_scale", types.Float2), ("texture_rotate", types.Float))}
            shader.CreateInput("project_uvw", types.Bool).Set(True)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(path)), UsdShade.Tokens.strongerThanDescendants)
            return inputs

        self.table = [material(path) for path in sim_utils.find_matching_prim_paths(env.scene[table_name].cfg.prim_path)]
        self.blocks = [[material(path) for path in sim_utils.find_matching_prim_paths(env.scene[name].cfg.prim_path)]
                       for name in block_names]  # [slot][env]
        self.lamps = []
        for path in sim_utils.find_matching_prim_paths(env.scene[table_name].cfg.prim_path):
            lamp = UsdLux.SphereLight.Define(stage, path.rsplit("/", 1)[0] + "/Lamp")
            lamp.CreateRadiusAttr(cfg.lamp_radius)
            lamp.CreateEnableColorTemperatureAttr(True)
            for excluded in cfg.lamp_excludes:
                if stage.GetPrimAtPath(excluded).IsValid():
                    UsdLux.LightAPI(lamp.GetPrim()).GetLightLinkCollectionAPI().CreateExcludesRel().AddTarget(excluded)
            self.lamps.append((lamp, lamp.AddTranslateOp()))
        dome = stage.GetPrimAtPath(cfg.dome_path)
        if cfg.dome_intensity is not None and dome.IsValid():
            UsdLux.LightAPI(dome).CreateIntensityAttr().Set(cfg.dome_intensity)
        self.draws: list[dict] = [{} for _ in range(self.num_envs)]
        """Each env's current draw, for whoever records the episode."""

    def _uniform(self, low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand((), generator=self.generator))

    def _surface(self, inputs: dict, color, texture: str | None, roughness: float, metallic: float, grey: bool,
                 brightness: float, scale: float):
        Gf, Sdf = self._gf, self._sdf
        inputs["diffuse_texture"].Set(Sdf.AssetPath(texture or ""))
        inputs["diffuse_color_constant"].Set(Gf.Vec3f(*color))
        inputs["diffuse_tint"].Set(Gf.Vec3f(*color) if texture and grey else Gf.Vec3f(1.0, 1.0, 1.0))
        inputs["albedo_desaturation"].Set(1.0 if grey else 0.0)
        inputs["albedo_brightness"].Set(brightness)
        inputs["reflection_roughness_constant"].Set(roughness)
        inputs["metallic_constant"].Set(metallic)
        inputs["specular_level"].Set(0.5)
        inputs["texture_scale"].Set(Gf.Vec2f(scale, scale))
        inputs["texture_rotate"].Set(self._uniform(0.0, 360.0))

    def randomize(self, env_ids: list[int], colors: torch.Tensor):
        """New draws for the envs env_ids: colors (N, slots) indices into BLOCK_COLORS."""
        cfg, Gf = self.cfg, self._gf
        palette = list(BLOCK_COLORS.values())
        names = list(BLOCK_COLORS)
        chosen = colors.tolist()
        choice = lambda items: items[int(torch.randint(len(items), (), generator=self.generator))]
        with self._sdf.ChangeBlock():
            for e in env_ids:
                draw = {}
                # the table
                texture = choice(self.table_textures) if self.table_textures and self._uniform(0, 1) < cfg.table_texture_share else None
                color = [self._uniform(*cfg.table_color_range) for _ in range(3)]
                roughness, metallic = self._uniform(*cfg.table_roughness_range), self._uniform(*cfg.table_metallic_range)
                self._surface(self.table[e], color, texture, roughness, metallic, False,
                              self._uniform(*cfg.table_brightness_range), self._uniform(*cfg.table_texture_scale_range))
                draw["table"] = {"texture": texture and texture.rsplit("/", 1)[-1], "color": [round(c, 3) for c in color],
                                 "roughness": round(roughness, 3), "metallic": round(metallic, 3)}
                # the blocks
                draw["blocks"] = []
                for slot, per_env in enumerate(self.blocks):
                    base = palette[chosen[e][slot]]
                    gain = self._uniform(*cfg.block_color_gain_range)
                    color = [min(1.0, max(0.0, c * gain + self._uniform(-cfg.block_color_shift, cfg.block_color_shift))) for c in base]
                    texture = choice(self.block_textures) if self.block_textures and self._uniform(0, 1) < cfg.block_texture_share else None
                    metallic = self._uniform(*cfg.block_metallic_range) if self._uniform(0, 1) < cfg.block_metallic_share else 0.0
                    roughness = self._uniform(*cfg.block_roughness_range)
                    self._surface(per_env[e], color, texture, roughness, metallic, True, self.block_texture_gain.get(texture, 1.0),
                                  self._uniform(*cfg.block_texture_scale_range))
                    draw["blocks"].append({"slot": slot, "color": names[chosen[e][slot]], "rgb": [round(c, 3) for c in color],
                                           "texture": texture and texture.rsplit("/", 1)[-1],
                                           "roughness": round(roughness, 3), "metallic": round(metallic, 3)})
                # the lamp, placed from the table's prim
                lamp, translate = self.lamps[e]
                place = [self._uniform(*r) for r in (cfg.lamp_x_range, cfg.lamp_y_range, cfg.lamp_z_range)]
                on = self._uniform(0, 1) < cfg.lamp_share
                intensity = self._uniform(*cfg.lamp_intensity_range) if on else 0.0
                kelvin = self._uniform(*cfg.lamp_temperature_range)
                translate.Set(Gf.Vec3d(*place))
                lamp.CreateIntensityAttr().Set(intensity)
                lamp.CreateColorTemperatureAttr().Set(kelvin)
                draw["lamp"] = {"pos": [round(c, 3) for c in place], "intensity": round(intensity), "kelvin": round(kelvin)}
                self.draws[e] = draw


"""
Observations.
"""


def vla_proprio(env, asset_cfg: SceneEntityCfg, arm_action: str = "arm_pos", gripper_action: str = "grippers") -> torch.Tensor:
    """(N, 34): the arm joints' angles from their default (14), each gripper's measured closure (2), and each grasp
    point's position (3) and rotation (its first two columns, 6) in the pelvis frame, from the measured joint angles'
    forward kinematics, left then right (18)."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    pos, R = env.action_manager.get_term(arm_action).measured_tcp()
    closure = env.action_manager.get_term(gripper_action).closure()
    rot6 = R[..., :2].transpose(-1, -2).reshape(env.num_envs, 2, 6)
    return torch.cat([q, closure, torch.cat([pos, rot6], dim=-1).reshape(env.num_envs, -1)], dim=1)


def vla_commanded_tcp(env, arm_action: str = "arm_pos") -> torch.Tensor:
    """(N, 18): each grasp point's commanded position (3) and rotation (first two columns, 6) in the pelvis frame,
    left then right: the controller's own state, which the action's deltas add to (DeltaEEArmAction). The policy
    needs it: the teacher's action is the waypoint's offset from the commanded grasp point, and from the measured
    one alone (which lags and sags a few mm from it) a network predicted that action no better than its mean."""
    pos, R = env.action_manager.get_term(arm_action).commanded_tcp()
    return torch.cat([pos, R[..., :2].transpose(-1, -2).reshape(env.num_envs, 2, 6)], dim=-1).reshape(env.num_envs, -1)


def stack_succeeded(env) -> torch.Tensor:
    """Termination: the instructed block has rested on its target, released, for cfg.success_hold_s (VlaUpperEnv)."""
    return env.stack_success


def block_dropped(env) -> torch.Tensor:
    """Termination (a failure): the instructed block or its target is off the table (VlaUpperEnv)."""
    return env.block_off_table
