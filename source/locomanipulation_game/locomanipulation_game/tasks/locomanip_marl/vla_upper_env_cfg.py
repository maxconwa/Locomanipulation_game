"""The upper body stacking blocks on a table from cameras and an instruction, the pelvis pinned: LocoManip-VlaUpper-v0.

LocoManip-ArmsOnly-v0 (arms_only_env_cfg.py: the pelvis pinned at the stance height, the legs holding their default
pose, the arms agent the only one acting) with a manipulation scene and the interface of a vision-language-action
policy in place of the arm goals:
- a table in front of the robot and, per env, six blocks, two of each size name (small, medium, large), their edge
  lengths and colours drawn once per env (mdp/vla.py block_layout); each episode puts two to four of them on the
  table and names one to stack on another: "stack the small red block on the large blue block" (VlaUpperEnv);
- three RGB cameras at CL_Assets' MuJoCo model's poses: the head camera (on the torso, pitched 51 deg
  down) and one in each Magpie gripper looking along its jaws, 58 deg vertical field of view;
- the arms agent's action (14): per arm a grasp-point position delta and rotation delta in the pelvis frame
  (mdp/vla.py DeltaEEArmAction: IK onto joint targets the motors' PD tracks), then one open/close command per gripper
  (MagpieGripperAction: the four-bar's hinges driven together, the cranks' weak drives the grip force);
- the arms agent's observation (52): arm joint angles, gripper closures, the measured grasp-point poses (mdp/vla.py
  vla_proprio) and the commanded ones (vla_commanded_tcp);
  the images and the instruction are read from the env (VlaUpperEnv.camera_images, .instructions);
- one policy step is 50 ms (20 Hz): 10 physics substeps of 5 ms, the joint targets interpolated across them;
- the arm-goal command stays (the env's odometry and curriculum state read it) but nothing follows it, and every
  reward is off: the policy is trained by imitation (scripts/vla/train.py), not by the reward.

The legs agent's slot is unused here as in ArmsOnly; a later two-agent task can train it beside this policy.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES

from .arms_only_env_cfg import ARMS_KINEMATICS_SPHERES, ARMS_KINEMATICS_URDF, ArmsOnlyEnvCfg
from .golem_safety import GOLEM_TARGET_CLIP
from .locomanip_marl_env_cfg import LEGS_ACTION, TORQUE_HEADROOM, LocoManipMarlEnvCfg
from .mdp import vla
from .scene_cfg import SELF_CONTACT_SENSOR_NAMES, TerrainSceneCfg

VLA_SIM_DT = 0.005   # physics 200 Hz: nothing balances here, and a policy step costs its substeps' count in time
VLA_DECIMATION = 10  # 20 Hz policy steps
GRIPPER_ACTION = "grippers"
ARMS_ACTION = "arm_pos"
NUM_BLOCKS = len(vla.BLOCK_SLOT_CLASSES)
BLOCK_NAMES = [f"block_{i}" for i in range(NUM_BLOCKS)]
IMAGE_SIZE = 168       # the wrist cameras, square: 12 x 12 patches of the wrist encoder's 14 px
# the head camera, 4:3: 73 deg across. The blocks are 0.9 m from it, 25 px wide. Square (58 deg across) it missed the
# blocks beyond 0.35 m to the side; its lower edge is 0.27 m ahead of the pelvis on the table at either shape.
HEAD_IMAGE_SIZE = (448, 336)
GREEN_SCREEN = (0.0, 1.0, 0.0)  # the floor's colour where images are recorded for background augmentation
CAMERA_NAMES = ("head_cam", "left_wrist_cam", "right_wrist_cam")
CAMERA_VFOV = 58.0     # deg, vertical: the MJCF's head_cam and hand_cam
CAMERA_APERTURE = 20.955

# The table: its top TABLE_TOP above the ground (the pinned pelvis is at 0.98 m), its near edge 0.17 m in front of
# the pelvis. With the grasp approach leaning 0.5 rad forward of straight down, each arm reaches grasp points 0.2 m
# below the pelvis to its height over x 0.25-0.45 m and |y| 0.15-0.40 m at any yaw of the jaws (ArmKinematics.solve
# on a grid; straight down reaches about half of that, and neither arm reaches the middle strip).
TABLE_TOP = 0.78
TABLE_SIZE = (0.70, 1.30, 0.04)
TABLE_CENTER_X = 0.52
# the arms' ready pose: both grasp points 0.20 m ahead, 0.40 m to the side, 2 cm over the pelvis's height, the
# approach leaning 0.5 rad forward, the jaws closing sideways (ArmKinematics.solve). Pulled back and out so the arms
# don't hide the blocks from the head camera: at 0.32 m ahead and 0.27 m to the side they covered the blocks' zone.
READY_ARM = (0.342, 1.304, -0.562, -0.561, -1.449, 0.312, 0.645)  # left; the right arm mirrors roll, yaw, wrist roll/yaw
_MIRROR = (1, -1, -1, 1, -1, 1, -1)
READY_JOINT_POS = {
    **{name: angle for name, angle in zip(ARM_JOINT_NAMES[:7], READY_ARM)},
    **{name: angle * sign for name, angle, sign in zip(ARM_JOINT_NAMES[7:], READY_ARM, _MIRROR)},
}


def _camera(parent: str, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), size=(IMAGE_SIZE, IMAGE_SIZE)) -> TiledCameraCfg:
    width, height = size
    focal = CAMERA_APERTURE * height / width / (2.0 * math.tan(math.radians(CAMERA_VFOV) / 2.0))
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + parent + "/vla_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=CAMERA_APERTURE,
                                         clipping_range=(0.02, 20.0)),
        width=width, height=height, data_types=["rgb"],
    )


def _block() -> sim_utils.CuboidCfg:
    """A unit cube: the prestartup event scales it to its env's edge length and colours it (mdp/vla.py)."""
    return sim_utils.CuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, solver_position_iteration_count=8),
        mass_props=sim_utils.MassPropertiesCfg(density=400.0),  # wood: 11 g at 3 cm, 137 g at 7 cm
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.6),
    )


@configclass
class VlaSceneCfg(TerrainSceneCfg):
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_TOP - TABLE_SIZE[2] / 2.0)),
    )
    # the head camera at the MJCF's head_cam: on the torso, 0.69 m over the pelvis, pitched 50.8 deg down (the USD's
    # head_camera_link frame looked about 10 deg to the left of that, which cut the right arm's blocks off)
    head_cam = _camera("torso_link", pos=(0.11109, 0.0175, 0.68789), rot=(0.23743, -0.66605, 0.66605, -0.23743), size=HEAD_IMAGE_SIZE)
    # the grippers' cameras: 4.5 cm up the mount's axis, looking along it (MJCF hand_cam)
    left_wrist_cam = _camera("lg_mount", pos=(0.0, 0.0, 0.045), rot=(0.0, 0.0, 0.0, 1.0))
    right_wrist_cam = _camera("rg_mount", pos=(0.0, 0.0, 0.045), rot=(0.0, 0.0, 0.0, 1.0))
    # nothing reads this one. With the three cameras above alone, the right wrist camera's image was of another view
    # in every env (the block in its jaws at the image's left edge, its fingertips out of sight; the left one's
    # centred): a tiled-rendering fault that goes with the number of cameras. With a fourth it is right; whoever adds
    # or removes a camera should look at both wrist images again (the fingertips: top and bottom centre, at rest).
    spare_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/spare_cam", width=16, height=16, data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, clipping_range=(0.05, 30.0)))

    def __post_init__(self):
        super().__post_init__()
        # a flat plane with the envs on a grid, env_spacing apart: on the walking task's generated terrain every env
        # starts on one of 20 tiles, so robots and blocks of different envs stand in the same place (they don't
        # collide) and each env's cameras saw the others' blocks
        self.terrain.terrain_type = "plane"
        self.terrain.terrain_generator = None
        # markers would show in the cameras
        self.height_scanner.debug_vis = False
        self.imu.debug_vis = False
        self.light.spawn.intensity = 1500.0
        for i, name in enumerate(BLOCK_NAMES):
            # parked on the ground behind the robot, within its own env's cell, until an episode puts it on the table
            setattr(self, name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(), spawn=_block(),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, -0.5 + 0.2 * i, 0.05))))


@configclass
class VlaArmsObsCfg(ObsGroup):
    """The arms agent's proprioception (mdp/vla.py vla_proprio), with the joint encoders' noise, and the controller's
    commanded grasp points (vla_commanded_tcp), exact."""

    proprio = ObsTerm(func=vla.vla_proprio, noise=Unoise(n_min=-0.001, n_max=0.001), params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
        "arm_action": ARMS_ACTION, "gripper_action": GRIPPER_ACTION})
    command = ObsTerm(func=vla.vla_commanded_tcp, params={"arm_action": ARMS_ACTION})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class VlaUpperEnvCfg(ArmsOnlyEnvCfg):
    decimation = VLA_DECIMATION
    episode_length_s = 20.0
    episode_length_range_s = None
    scene: VlaSceneCfg = VlaSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=False)
    # every joint target on time: the arm term spreads its target over the policy step's substeps itself
    action_delay_range = None
    termination_penalty: float = 0.0
    golem_estop_penalty: float = 0.0

    # -- the stacking task (VlaUpperEnv)
    block_seed: int = 0
    """block_layout's seed: each env's block sizes (and colours, where they are not drawn per episode)."""
    block_colors_per_episode: bool = True
    """Each reset colours the env's blocks anew (their sizes are fixed when the simulation starts: PhysX reads a
    collider's scale once). With the 64 envs' colours fixed for a run, the policy's block localization was 0.7 cm
    on them and 5.7 cm on another seed's."""
    visuals: vla.VisualRandomizationCfg | None = None
    """Set: each reset draws the table's surface, the blocks' finish (roughness, metal, texture) and a lamp
    (mdp/vla.py VisualRandomizer). None: the plain table and matt blocks under the dome light."""
    num_blocks_range: tuple[int, int] = (2, 4)
    """Blocks on the table per episode, the instructed pair included."""
    zone_x: tuple[float, float] = (0.28, 0.42)
    """m ahead of the pelvis: where the instructed block and its target are put (the head camera sees the table from
    0.27 m on)."""
    zone_y: tuple[float, float] = (0.24, 0.44)
    """m to the side of the pelvis: the instructed pair's zone, on the arm's side. Inside what the arm reaches at any
    yaw of the jaws (TABLE_TOP's comment); with blocks from 0.15 m to the side and 0.43 m ahead, the teacher's arm
    stopped short in a third of the episodes."""
    shared_zone_share: float = 0.25
    """Share of the episodes whose instructed pair is put in the shared zone, in front of the pelvis, where either arm
    may reach (the teacher picks one at random where both do, and the instruction names it)."""
    shared_zone_x: tuple[float, float] = (0.28, 0.42)
    shared_zone_y: tuple[float, float] = (-0.08, 0.08)
    """m: the shared zone. With both blocks within 0.08 m of the middle and no further than 0.42 m ahead, the teacher's
    plan was feasible for both arms in two thirds of the draws; from 0.20 m to the side on, in none."""
    spare_zone_x: tuple[float, float] = (0.28, 0.52)
    spare_zone_y: tuple[float, float] = (0.08, 0.48)
    """m: where the other blocks on the table are put, on either side (nothing has to reach them)."""
    block_min_distance: float = 0.115
    """m between block centres at the reset: the open jaws (11 cm) fit around one beside another."""
    table_height_range: tuple[float, float] = (-0.015, 0.015)
    """m: the table top's height from TABLE_TOP, drawn at each reset."""
    success_hold_s: float = 0.5
    """s the instructed block must rest on its target, released, to count as stacked (it ends the episode)."""
    success_xy_share: float = 0.5
    """The block's centre within this share of the target's edge length of the target's centre, horizontally."""
    success_z_tol: float = 0.012
    """m between the block's height and its height resting on the target."""
    success_max_speed: float = 0.05
    """m/s of the block."""
    success_release_distance: float = 0.06
    """m from the block's centre to each grasp point."""
    sampler_floor: float = 0.2
    """Share of episodes whose configuration is drawn uniformly, the rest by the success-guided weights."""
    sampler_target: float = 0.4
    """The success rate whose configurations are drawn most."""
    sampler_concentration: float = 4.0
    """The Beta weight's a + b (above 2): the larger, the more the draws keep to configurations at the target rate."""
    sampler_decay: float = 0.9
    """Each configuration bin's success rate: an exponential mean over its episodes, this decay per episode."""

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = VLA_SIM_DT
        robot = self.scene.robot
        # the arms start at the ready pose
        arm_patterns = (".*_shoulder_", ".*_elbow_", ".*_wrist_")
        robot.init_state.joint_pos = {
            **{k: v for k, v in robot.init_state.joint_pos.items() if not k.startswith(arm_patterns)},
            **READY_JOINT_POS}
        # the grippers: the cranks' drives soft (the grip is their 0.08 rad lead on a held block, 1.6 Nm on a 4.5 cm
        # crank: about 35 N; MagpieGripperAction), the finger and rocker hinges' stiff, as the locked grippers' were
        robot.actuators = {
            **{name: act for name, act in robot.actuators.items() if name != "grippers"},
            "gripper_cranks": ImplicitActuatorCfg(joint_names_expr=["[lr]g_.*_hinge_1"], effort_limit_sim=4.0,
                                                  velocity_limit_sim=3.14, stiffness=20.0, damping=1.0, armature=0.01),
            "gripper_links": ImplicitActuatorCfg(joint_names_expr=["[lr]g_.*_hinge_[23]"], effort_limit_sim=10.0,
                                                 velocity_limit_sim=3.14, stiffness=100.0, damping=5.0, armature=0.01),
        }
        # the actions: the legs' held term, the arms' delta end-effector IK, the grippers
        envelope = self.actions.arm_pos.joint_limits
        self.actions.arm_pos = vla.DeltaEEArmActionCfg(
            asset_name="robot", joint_names=ARM_JOINT_NAMES, kinematics_urdf=ARMS_KINEMATICS_URDF,
            kinematics_spheres=ARMS_KINEMATICS_SPHERES, target_margin=-GOLEM_TARGET_CLIP, joint_limits=envelope,
            torque_headroom=TORQUE_HEADROOM)
        self.actions.grippers = vla.MagpieGripperActionCfg(asset_name="robot")
        self.agent_action_terms = {"legs": [LEGS_ACTION], "arms": [ARMS_ACTION, GRIPPER_ACTION]}
        self.observations.arms = VlaArmsObsCfg()
        # the arm-goal command: kept for the env, drawing rest goals nothing follows; no markers in the cameras
        arm = self.commands.arm_targets
        arm.seed_solve, arm.collision_check, arm.rest_goal_prob, arm.debug_vis = False, False, 1.0, False
        self.commands.base_velocity.debug_vis = False
        self.rewards.arms.amp_style = None
        # resets: the robot within 2 cm and 3 deg of the env's frame (the table stays), the arms near the ready pose
        pose = self.events.reset_base.params["pose_range"]
        pose["x"], pose["y"], pose["yaw"] = (-0.02, 0.02), (-0.02, 0.02), (-0.05, 0.05)
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_joints.params["position_range"] = (-0.03, 0.03)
        self.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
        # the pads grip: their friction 0.8-1.2, not the walking task's 0.1-1.25
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.2)
        self.events.setup_blocks = EventTerm(func=vla.setup_blocks, mode="prestartup",
                                             params={"block_names": BLOCK_NAMES, "seed": self.block_seed})
        self.terminations.stacked = DoneTerm(func=vla.stack_succeeded)
        self.terminations.block_dropped = DoneTerm(func=vla.block_dropped, time_out=True)
        self.viewer.eye, self.viewer.lookat = (1.6, 1.2, 1.5), (0.4, 0.0, 0.85)
        self.sync_derived()

    def sync_derived(self):
        # the walking task's (the physics-rate settings); ArmsOnly's variants don't apply
        LocoManipMarlEnvCfg.sync_derived(self)
        # one render per policy step, the cameras read then
        self.sim.render_interval = self.decimation
        for name in CAMERA_NAMES:
            camera = getattr(self.scene, name, None)
            if camera is not None:
                camera.update_period = 0.0
        # the contact sensors once per policy step, the self-contact ones only when read (no reward reads them):
        # with a history every sensor recomputes at every physics substep, which was most of a step's time
        self.scene.contact_forces.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.history_length = 1
        for name in SELF_CONTACT_SENSOR_NAMES:
            getattr(self.scene, name).history_length = 0
        if "setup_blocks" in vars(self.events):
            self.events.setup_blocks.params["seed"] = self.block_seed
        # imitation only: every reward term off
        for rewards in (self.rewards.arms, self.rewards.legs):
            for term in vars(rewards).values():
                if isinstance(term, RewTerm):
                    term.weight = 0.0
