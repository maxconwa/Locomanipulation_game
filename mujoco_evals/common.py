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
    velocity command, s; 0: none), the default pose and each action term's scale per joint, and arm_goals (whether the
    run trained arm goals at all, arm_goal_prob > 0), and arm_ik (mdp.IKPostureArmAction's settings, when the run used it)."""
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

    # the default pose (the legs' action offset) and each action term's scale per joint (lmg_deploy reads them over the
    # template's legs.action_scale and arms.residual_scale). A single number too: dropped, the template's would apply
    # (an arms residual trained at scale 0 would deploy at the template's 0.2)
    pose = {"default_joint_pos": [matching(env["scene"]["robot"]["init_state"]["joint_pos"], j) for j in names]}
    actions = env.get("actions", {})
    for term, key in (("joint_pos", "leg_action_scales"), ("arm_pos", "arm_residual_scales")):
        scale = (actions.get(term) or {}).get("scale")
        if isinstance(scale, dict):
            pose[key] = [matching(scale, j) for j in actions[term]["joint_names"]]
        elif isinstance(scale, (int, float)):
            pose[key] = [float(scale)] * len(actions[term]["joint_names"])
    # the arms' residual gain (IKResidualArmActionCfg.residual_gain) folded into their scales
    gain = (actions.get("arm_pos") or {}).get("residual_gain")
    if gain is not None and "arm_residual_scales" in pose:
        pose["arm_residual_scales"] = [float(gain) * s for s in pose["arm_residual_scales"]]
    if (actions.get("arm_pos") or {}).get("max_ik_step") is not None:
        pose["max_ik_step"] = float(actions["arm_pos"]["max_ik_step"])
    # the run's own, where the deploy template has one value for every run
    for key in ("residual_cutoff_hz", "ik_damping"):
        if (actions.get("arm_pos") or {}).get(key) is not None:
            pose[key] = float(actions["arm_pos"][key])
    if env.get("clip_actions") is not None:
        pose["clip_actions"] = float(env["clip_actions"])
    # mdp.IKPostureArmAction's IK (LocoManip-ArmsOnly-v0): its settings, and its action twice as wide
    arm_pos = actions.get("arm_pos") or {}
    if "rot_weight" in arm_pos:
        pose["arm_ik"] = {key: float(arm_pos[key]) for key in ("rot_weight", "null_gain", "posture_scale",
                                                              "posture_gain", "limit_margin", "ik_damping")}
        pose["arm_ik"]["ik_passes"] = int(arm_pos["ik_passes"])
        pose["arm_ik"]["posture_from_command"] = bool(arm_pos.get("posture_from_command", False))
        pose["arm_ik"]["trajectory"] = bool(arm_pos.get("trajectory", False))
        pose["arm_ik"]["seed_rot_weight"] = float(env["commands"]["arm_targets"].get("seed_rot_weight", 0.3))
        pose["arm_ik"]["posture_actions"] = bool(arm_pos.get("posture_actions", True))
        for key in ("trajectory_max_vel", "trajectory_max_acc"):
            pose["arm_ik"][key] = None if arm_pos.get(key) is None else float(arm_pos[key])
        # IKPostureArmActionCfg.gravity_offset: the arm joints' gravity torque at the target over kp, on the target
        pose["arm_ik"]["gravity_offset"] = bool(arm_pos.get("gravity_offset", False))
        # residual_integrate: the residual integrated (leak, gain, clamp), not low-passed
        if arm_pos.get("residual_integrate"):
            pose["arm_ik"]["residual_integrate"] = {"gain": float(arm_pos["residual_integrate_gain"]),
                                                    "leak_s": float(arm_pos["residual_leak_s"]),
                                                    "clamp": float(arm_pos["residual_clamp"])}
        # integral_gain: the measured wrists' pose error integrated onto the target once it has stopped
        if float(arm_pos.get("integral_gain") or 0.0) > 0.0:
            pose["arm_ik"]["integral"] = {"gain": float(arm_pos["integral_gain"]),
                                          "leak_s": float(arm_pos["integral_leak_s"]),
                                          "clamp": float(arm_pos["integral_clamp"])}
        # IKPostureArmActionCfg.freeze_residual_on_goal: the residual latched on the goal
        if arm_pos.get("freeze_residual_on_goal"):
            pose["arm_ik"]["freeze"] = {key: float(arm_pos[f"freeze_{key}"]) for key in ("pos_tol", "rot_tol", "settle_s")}
        # ArmTargetsCommandCfg.seed_solve: the seed solved for the command from the nearest row (mdp.arm_kinematics)
        arm_cmd = env["commands"]["arm_targets"]
        if arm_cmd.get("seed_solve"):
            pose["arm_ik"]["seed_solve"] = {"iters": int(arm_cmd["seed_solve_iters"]),
                                            "damping": float(arm_cmd["seed_solve_damping"]),
                                            "rot_weight": float(arm_cmd["seed_solve_rot_weight"]),
                                            "joint_range_margin": float(arm_cmd.get("joint_range_margin") or 0.0)}
    return {
        "control_dt": round(float(env["decimation"]) * float(env["sim"]["dt"]), 6),
        **gains,
        **pose,
        "gait_period": float(env.get("gait_period", 0.8)),
        "zero_cmd_eps": float(env.get("zero_cmd_eps", 0.1)),
        "command_ranges": {axis: [float(v) for v in velocity["ranges"][axis]]
                           for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "command_filter_tau": float(velocity.get("command_filter_tau", 0.0)),
        "arm_goals": float(env["commands"]["arm_targets"]["arm_goal_prob"]) > 0.0,
        # each actor's observation terms in order (MarlMujoco builds the vectors from them)
        **{f"{agent}_obs_terms": [name for name, term in env["observations"][agent].items()
                                  if isinstance(term, dict) and "func" in term] for agent in ("legs", "arms")},
    }


def load_deploy_cfg(export_dir: str, config: str | None = None, template: str | None = None) -> dict:
    """An export folder's deploy cfg, as lmg_deploy reads it: `config`, else the folder's yaml that isn't export.yaml
    (the copy of the policy template that export_marl.py leaves there), else `template`; with export.yaml's keys over
    it and the policies' file names filled in."""
    import glob

    import yaml

    yamls = sorted(p for p in glob.glob(os.path.join(export_dir, "*.yaml")) if os.path.basename(p) != "export.yaml")
    path = config or (yamls[0] if yamls else template)
    if path is None:
        raise FileNotFoundError(f"{export_dir}: no deploy yaml beside export.yaml")
    cfg = yaml.safe_load(open(path))
    cfg.update(yaml.safe_load(open(os.path.join(export_dir, "export.yaml"))) or {})
    cfg.setdefault("legs", {}).setdefault("policy", "legs.pt")
    cfg.setdefault("arms", {}).setdefault("policy", "arms.pt")
    return cfg
