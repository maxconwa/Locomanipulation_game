import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns


from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG

from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from .. import mdp


HEIGHT_SCAN_RAISE = 20.0


_TERRAIN_COMMON = dict(
    size=(8.0, 8.0),
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    seed=0,
    curriculum=True,          # required by mdp.terrain_levels_vel
)


TERRAINS_FLAT_CFG = TerrainGeneratorCfg(
    num_rows=10, num_cols=20, **_TERRAIN_COMMON,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


TERRAINS_FULL_CFG = TerrainGeneratorCfg(
    num_rows=10, num_cols=20, **_TERRAIN_COMMON,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.02, 0.14), step_width=0.40,
            platform_width=2.0, border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.02, 0.14), step_width=0.40,
            platform_width=2.0, border_width=0.25,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10, obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0), obstacle_height_range=(0.05, 0.20),
            num_obstacles=12, platform_width=2.0, border_width=0.25,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.0, 0.20), num_waves=4,
            border_width=0.25,
        ),
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.15, stone_height_max=0.05,
            stone_width_range=(0.4, 1.0), stone_distance_range=(0.0, 0.25),
            holes_depth=-0.35, platform_width=2.0, border_width=0.25,
        ),
    },
)




@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAINS_FULL_CFG,
        max_init_terrain_level=0,   # start everyone flat; None means all levels
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
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
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
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_link",
        debug_vis=False,
    )



@configclass
class FlatSceneCfg(TerrainSceneCfg):
    def __post_init__(self):
        self.terrain.terrain_generator = TERRAINS_FLAT_CFG

@configclass
class CurriculumCfg:
    # Per env, at episode end. Promotes past 4 m from the origin (half a patch),
    # demotes below half the commanded distance. Logs Curriculum/terrain_levels.
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
