"""The scene: one H1-2 per env on ALMI-Open's rough terrain, its contact sensors, height scan and head IMU."""


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import COLLISION_LINK_NAMES, H1_2_MAGPIE_CFG

HEIGHT_SCAN_RAISE = 20.0
SELF_CONTACT_LINK_NAMES = COLLISION_LINK_NAMES[:-1]
SELF_CONTACT_SENSOR_NAMES = [f"self_contact_{link}" for link in SELF_CONTACT_LINK_NAMES]
# placeholder: LocoManipMarlEnvCfg.sync_physics_rate sets it to one slot per physics substep of a policy step
CONTACT_HISTORY = 5


@height_field_to_mesh
def _rough_by_difficulty(difficulty: float, cfg: "RoughByDifficultyCfg"):
    """Isaac Lab's random_uniform_terrain with the bumps scaled by the row's difficulty (it ignores difficulty)."""
    height = cfg.height_at_0 + cfg.height_per_difficulty * difficulty
    step = cfg.step_at_0 + cfg.step_per_difficulty * difficulty
    return random_uniform_terrain.__wrapped__(difficulty, cfg.replace(noise_range=(-height, height), noise_step=step))


@configclass
class RoughByDifficultyCfg(HfRandomUniformTerrainCfg):
    """ALMI-Open's only_rough terrain (legged_gym utils/terrain.py): heights uniform in +-(height_at_0 + 0.05 d) m
    on steps of 0.005 + 0.005 d m, d the row's difficulty in [0, 1], drawn every 0.2 m (ALMI-Open's height_at_0 is
    0.005; here 0, so row 0 is flat)."""

    function = _rough_by_difficulty
    noise_range: tuple[float, float] = (0.0, 0.0)  # set per difficulty
    noise_step: float = 0.005
    downsampled_scale: float = 0.2
    height_at_0: float = 0.0
    height_per_difficulty: float = 0.05
    step_at_0: float = 0.005
    step_per_difficulty: float = 0.005


# Rows by difficulty: flat (row 0) to bumps of about +-5 cm (row 9); the terrain curriculum moves envs between them.
TERRAINS_ROUGH_CFG = TerrainGeneratorCfg(
    num_rows=10,
    num_cols=20,
    size=(8.0, 8.0),
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    seed=0,
    curriculum=True,
    sub_terrains={"rough": RoughByDifficultyCfg(proportion=1.0)},
)


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAINS_ROUGH_CFG,
        # every env starts on row 0; the terrain_levels curriculum moves envs up and down
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = H1_2_MAGPIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # All bodies: the reward terms need feet, knees, pelvis and torso.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=CONTACT_HISTORY, track_air_time=True
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, HEIGHT_SCAN_RAISE)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # The head's Livox MID-360 IMU chip, the frame of GOLEM's /livox/imu: livox_link/livox_imu_link, which CL_Assets
    # ed592de places with its measured lever arm and the chip's 0.68 deg tilt (torso_link -> chip: pitch 9.45 deg).
    # debug_vis also makes the sensor compute at every physics substep (SensorBase.update), which the gyro low-pass
    # samples.
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/livox_link/livox_imu_link",
        debug_vis=True,
    )
    def __post_init__(self):
        # PhysX filters one body against many, never many against many, so each
        # link gets its own sensor. Each filters only against the links after it
        # in COLLISION_LINK_NAMES, so every pair is reported by exactly one sensor.
        for i, link in enumerate(SELF_CONTACT_LINK_NAMES):
            setattr(self, SELF_CONTACT_SENSOR_NAMES[i], ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + link,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + other for other in COLLISION_LINK_NAMES[i + 1:]], #slice avoids double counting contacts
                history_length=CONTACT_HISTORY,
            ))
