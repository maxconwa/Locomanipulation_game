"""What the MuJoCo tools and the export share: the assets' path, and a training run's settings read from its
params/env.yaml (the IMU, the target clip, and what lmg_deploy takes over its policy template). Imports only the
standard library (yaml lazily), so the Isaac tools can import it before AppLauncher.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# CL_Assets: CL_ASSETS_DIR, else the checkout's third_party/CL_Assets submodule
DEFAULT_ASSETS = Path(os.environ.get("CL_ASSETS_DIR", str(REPO / "third_party" / "CL_Assets")))
# the sole's height in each foot's ankle_roll_link frame (the collision mesh's lowest point)
SOLE_Z = -0.045


def _yaml_load_any(text: str):
    """yaml.safe_load that turns the python/* tags Isaac Lab's dump_yaml writes into plain values."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def any_tag(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", any_tag)
    Loader.add_multi_constructor("!", any_tag)
    return yaml.load(text, Loader=Loader)


def read_imu(run_dir: Path) -> dict:
    """The IMU a run trained with: {"frame": the robot frame it sits on, "offset": [x, y, z] in it (m)}, plus
    "gyro_lowpass_hz" when the policy read its gyro through the low-pass (imu_filter.GyroLowpass; lmg_deploy applies
    it only when the key is present)."""
    env = _yaml_load_any((Path(run_dir) / "params" / "env.yaml").read_text())
    imu = (env.get("scene") or {}).get("imu") or {}
    frame = str(imu.get("prim_path", "/imu_link")).rsplit("/", 1)[-1]
    pos = ((imu.get("offset") or {}).get("pos")) or [0.0, 0.0, 0.0]
    out = {"frame": frame, "offset": [round(float(v), 6) for v in pos]}
    if env.get("gyro_lowpass_hz"):
        out["gyro_lowpass_hz"] = float(env["gyro_lowpass_hz"])
    return out


def read_target_limits(run_dir: Path) -> dict[str, tuple[float, float]] | None:
    """joint name -> (low, high) a run's action terms clipped every target into (GOLEM's position e-stop range,
    actions.*.joint_limits in params/env.yaml); None for runs without it, whose targets only kept to the URDF clip."""
    path = Path(run_dir) / "params" / "env.yaml"
    if not path.is_file():
        return None
    actions = _yaml_load_any(path.read_text()).get("actions", {})
    limits = {}
    for term in ("joint_pos", "arm_pos"):
        for name, bounds in ((actions.get(term) or {}).get("joint_limits") or {}).items():
            limits[name] = (float(bounds[0]), float(bounds[1]))
    return limits or None


def read_deploy_settings(run_dir: Path) -> dict:
    """What a run trained with that GOLEM's lmg_deploy reads from its policy yaml, under the template's keys
    (config/policy_template.yaml; marl_deploy_node.py lets export.yaml override them), from params/env.yaml:
    control_dt (decimation x sim dt), kp and kd per joint in the policies' joint order (the actuators' stiffness and
    damping), gait_period, zero_cmd_eps, command_ranges and command_filter_tau (the first-order low-pass on the
    velocity command, s; 0: none), and the default pose and each action term's scale per joint."""
    import re

    env = _yaml_load_any((Path(run_dir) / "params" / "env.yaml").read_text())
    names = env["observations"]["legs"]["joint_pos"]["params"]["asset_cfg"]["joint_names"]
    gains = {"kp": [], "kd": []}
    actuators = env["scene"]["robot"]["actuators"]
    for joint in names:
        group = [a for a in actuators.values() if any(re.fullmatch(e, joint) for e in a["joint_names_expr"])]
        if len(group) != 1:
            raise ValueError(f"{joint}: in {len(group)} actuator groups")
        for key, field in (("kp", "stiffness"), ("kd", "damping")):
            value = group[0][field]
            if isinstance(value, dict):
                value = [v for e, v in value.items() if re.fullmatch(e, joint)]
                if len(value) != 1:
                    raise ValueError(f"{joint}: {len(value)} {field} entries")
                value = value[0]
            gains[key].append(float(value))
    velocity = env["commands"]["base_velocity"]

    def matching(table, joint):
        hits = [v for e, v in table.items() if re.fullmatch(e, joint)]
        if len(hits) != 1:
            raise ValueError(f"{joint}: {len(hits)} entries in {list(table)}")
        return float(hits[0])

    # the default pose (the legs' action offset) and each action term's scale per joint, when it is a dict (lmg_deploy
    # reads them over the template's legs.action_scale and arms.residual_scale)
    pose = {"default_joint_pos": [matching(env["scene"]["robot"]["init_state"]["joint_pos"], j) for j in names]}
    actions = env.get("actions", {})
    for term, key in (("joint_pos", "leg_action_scales"), ("arm_pos", "arm_residual_scales")):
        scale = (actions.get(term) or {}).get("scale")
        if isinstance(scale, dict):
            pose[key] = [matching(scale, j) for j in actions[term]["joint_names"]]
    if (actions.get("arm_pos") or {}).get("max_ik_step") is not None:
        pose["max_ik_step"] = float(actions["arm_pos"]["max_ik_step"])
    return {
        "control_dt": round(float(env["decimation"]) * float(env["sim"]["dt"]), 6),
        **gains,
        **pose,
        "gait_period": float(env.get("gait_period", 0.8)),
        "zero_cmd_eps": float(env.get("zero_cmd_eps", 0.1)),
        "command_ranges": {axis: [float(v) for v in velocity["ranges"][axis]]
                           for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "command_filter_tau": float(velocity.get("command_filter_tau", 0.0)),
    }
