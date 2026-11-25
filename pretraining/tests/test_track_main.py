"""The tracking env on main's plant (LocoManip-Pretrain-Track-Main-v0 and its fine-tune) and train.py's --plant."""

import dataclasses
import json

import gymnasium as gym
import pytest
import torch

import pretraining.tasks  # noqa: F401  (registers the gym ids)
from pretraining.constants import JOINT_NAMES
from pretraining.plants import MAIN
from standins import write_clips

MAIN_TASK = "LocoManip-Pretrain-Track-Main-v0"
FINETUNE_TASK = "LocoManip-Pretrain-Track-Main-FineTune-v0"
FRAMES = 60
CONFIG = "num_envs: 64\nrunner: {buffer_size: 50}\n"


def _make(task, num_envs, motion_dir):
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.utils import load_cfg_from_registry

    sim_utils.create_new_stage()  # a fresh stage: an earlier sim test may have left its prims
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    motion = cfg.commands.motion
    motion.motion_file = str(motion_dir)
    motion.sampling_mode = "start"
    motion.pose_range, motion.velocity_range, motion.joint_position_range = {}, {}, (0.0, 0.0)
    return gym.make(task, cfg=cfg).unwrapped


def test_registered_and_listed():
    for task_id, cfg_name in ((MAIN_TASK, "TrackMainEnvCfg"), (FINETUNE_TASK, "TrackMainFineTuneEnvCfg")):
        assert "LocoManip-" in task_id  # scripts/list_envs.py lists only these
        assert gym.registry[task_id].kwargs["env_cfg_entry_point"] == f"pretraining.tasks.track_env_cfg:{cfg_name}"


@pytest.mark.sim
def test_track_main_step_dt_and_targets(sim_app, tmp_path):
    """100 Hz on main's plant: the 50 Hz clip is resampled to 2n-1 frames and a zero action holds main's default."""
    write_clips(tmp_path, n=1, frames=FRAMES)
    env = _make(MAIN_TASK, 2, tmp_path)
    try:
        assert env.step_dt == pytest.approx(0.01) and env.physics_dt == pytest.approx(0.002)
        assert env.cfg.plant == "main"
        assert env.command_manager.get_term("motion").motion.time_step_total == 2 * FRAMES - 1
        env.reset()
        env.step(torch.zeros(2, len(JOINT_NAMES), device=env.device))
        target = env.action_manager.get_term("joint_pos").processed_actions
        default = torch.tensor([MAIN.default_joint_pos[j] for j in JOINT_NAMES], device=env.device)
        torch.testing.assert_close(target, default.expand(2, -1))
    finally:
        env.close()


@pytest.mark.sim
def test_finetune_delay_draws_within_range(sim_app, tmp_path):
    """Each env's target delay is an integer in ACTION_DELAY_RANGE substeps, drawn again at each reset."""
    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import ACTION_DELAY_RANGE

    write_clips(tmp_path, n=1, frames=FRAMES)
    n = 64
    env = _make(FINETUNE_TASK, n, tmp_path)
    try:
        term = env.action_manager.get_term("joint_pos")
        env.reset()
        first = term.delay_steps.clone()
        assert first.dtype == torch.long
        assert first.min() >= ACTION_DELAY_RANGE[0] and first.max() <= ACTION_DELAY_RANGE[1]
        assert len(first.unique()) > 3
        env.reset()
        assert not torch.equal(term.delay_steps, first)
        assert term.delay_steps.min() >= ACTION_DELAY_RANGE[0] and term.delay_steps.max() <= ACTION_DELAY_RANGE[1]
        # at the step's last substep (4) an env delayed by <= 4 substeps applies the step's target, one delayed by
        # 5-10 the previous step's (here the pose it was reset to)
        env.step(torch.ones(n, len(JOINT_NAMES), device=env.device))
        ids = env.scene["robot"].find_joints(list(JOINT_NAMES), preserve_order=True)[0]
        applied = env.scene["robot"].data.joint_pos_target[:, ids]
        new = term.processed_actions - env.encoder_bias
        on_time = (applied - new).abs().max(dim=1).values < 1e-5
        assert torch.equal(on_time, term.delay_steps <= env.cfg.decimation - 1)
    finally:
        env.close()


@pytest.mark.sim
def test_finetune_sim_context_reads_the_envs_draws(sim_app, tmp_path):
    """A new fine-tune-env context: the delay is the action term's, the gains and strength reproduce the actuators'
    kp, kd and effort limit from their nominal values, the torso mass is the PhysX mass less the default, friction the
    feet's material, all redrawn (delay, gains, strength) at a reset."""
    from pretraining.supermimic.sim_context import SimContext

    write_clips(tmp_path, n=1, frames=FRAMES)
    n = 64
    env = _make(FINETUNE_TASK, n, tmp_path)
    try:
        env.reset()
        ctx = SimContext.from_env(env)
        assert [s.name for s in ctx.specs] == ["foot_friction", "encoder_bias", "action_delay", "actuator_gain",
                                               "actuator_strength", "torso_mass"]
        raw = ctx.read(env) * ctx.scale + ctx.mean
        cols = {s.name: ctx.find(s.name) for s in ctx.specs}
        term = env.action_manager.get_term("joint_pos")
        torch.testing.assert_close(raw[:, cols["action_delay"]].squeeze(-1), term.delay_steps.float())
        gain, strength = raw[:, cols["actuator_gain"]], raw[:, cols["actuator_strength"]]
        assert strength.min() >= 0.85 - 1e-5 and strength.max() <= 1.15 + 1e-5 and strength.std() > 0.01
        assert gain.min() >= 0.5 - 1e-5 and gain.max() <= 1.2 + 1e-5
        robot = env.scene["robot"]
        for name, act in robot.actuators.items():
            if name not in env._actuator_nominal:
                continue
            kp, kd, limit = env._actuator_nominal[name]
            idx = [JOINT_NAMES.index(j) for j in act.joint_names]
            torch.testing.assert_close(act.stiffness, kp * gain[:, idx] * strength[:, idx])
            torch.testing.assert_close(act.damping, kd * gain[:, idx] * strength[:, idx])
            torch.testing.assert_close(act.effort_limit, limit * strength[:, idx])
        mass = raw[:, cols["torso_mass"]]
        assert mass.min() >= -2.0 - 1e-4 and mass.max() <= 4.0 + 1e-4 and mass.std() > 0.1
        friction = raw[:, cols["foot_friction"]]
        assert friction.min() >= 0.1 - 1e-5 and friction.max() <= 1.25 + 1e-5
        torch.testing.assert_close(raw[:, cols["encoder_bias"]], env.encoder_bias)
        before = ctx.read(env).clone()
        env.reset()
        after = ctx.read(env)
        assert not torch.equal(after[:, cols["actuator_gain"]], before[:, cols["actuator_gain"]])
        torch.testing.assert_close(after[:, cols["torso_mass"]], before[:, cols["torso_mass"]])  # start-up draw
    finally:
        env.close()


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PRETRAINING_RUNS_DIR", str(tmp_path / "runs"))
    write_clips(tmp_path / "motions", n=2, frames=FRAMES)
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    return ["--config", str(config), "--motion-dir", str(tmp_path / "motions")]


def test_plan_records_plant_and_task(tmp_path, monkeypatch):
    from pretraining.scripts import train

    argv = _setup(tmp_path, monkeypatch)

    def plan(*extra):
        return train.plan(train.build_parser().parse_args([*argv, "--run-name", "r", *extra]))

    p = plan()
    assert (p["plant"], p["task"]) == ("pretrain", "LocoManip-Pretrain-Track-v0")
    p = plan("--plant", "main")
    assert (p["plant"], p["task"]) == ("main", MAIN_TASK)
    p = plan("--plant", "main", "--fine-tune")
    assert (p["plant"], p["task"]) == ("main", FINETUNE_TASK)
    assert train.run_json_entries(p) == {"plant": "main", "task": FINETUNE_TASK}
    with pytest.raises(train.UsageError, match="--fine-tune"):
        plan("--fine-tune")


def _old_run(tmp_path, **extra):
    from pretraining.marl.config import ImitationRewardCfg
    from pretraining.marl.runner import TwoAgentRunnerCfg

    run = tmp_path / "runs" / "old"
    run.mkdir(parents=True)
    files = write_clips(tmp_path / "motions", n=2, frames=FRAMES)
    meta = {"runner": dataclasses.asdict(TwoAgentRunnerCfg()), "rewards": dataclasses.asdict(ImitationRewardCfg()),
            "num_envs": 8, "motion_files": [str(f) for f in files], **extra}
    (run / "run.json").write_text(json.dumps(meta, default=str))
    (run / "checkpoint.json").write_text(json.dumps({"iteration": 3, "wall_s": 1.0}))
    (run / "checkpoint.pt").write_bytes(b"")
    return run


def test_resume_old_run_json_means_pretrain(tmp_path):
    from pretraining.scripts import train

    p = train.plan(train.build_parser().parse_args(["--resume", str(_old_run(tmp_path))]))
    assert (p["plant"], p["task"]) == ("pretrain", "LocoManip-Pretrain-Track-v0")
    with pytest.raises(train.UsageError, match="--plant"):
        train.plan(train.build_parser().parse_args(["--resume", str(tmp_path / "runs" / "old"), "--plant", "main"]))


def test_resume_reads_plant_and_task(tmp_path):
    from pretraining.scripts import train

    run = _old_run(tmp_path, plant="main", task=FINETUNE_TASK)
    p = train.plan(train.build_parser().parse_args(["--resume", str(run)]))
    assert (p["plant"], p["task"]) == ("main", FINETUNE_TASK)


def test_fine_tune_from_refuses_another_plant(tmp_path, monkeypatch):
    from pretraining.scripts import train

    argv = _setup(tmp_path, monkeypatch)
    source = _old_run(tmp_path / "src")  # no plant: a pretrain-plant run
    # the source's clips are tmp_path/src/motions; use them
    argv[3] = str(tmp_path / "src" / "motions")
    with pytest.raises(train.UsageError, match="plant"):
        train.plan(train.build_parser().parse_args([*argv, "--run-name", "ft", "--fine-tune-from", str(source),
                                                    "--plant", "main"]))
    p = train.plan(train.build_parser().parse_args([*argv, "--run-name", "ft", "--fine-tune-from", str(source)]))
    assert p["plant"] == "pretrain"


@pytest.mark.sim
def test_main_fine_tune_from_a_main_base_run_keeps_its_world_model(tmp_path):
    """A main-plant base run (31 sim-context columns: friction, torso COM, encoder bias), then --fine-tune from it in
    the fine-tune env: the layout is the source's columns (friction live from main's material draw, COM held at its
    mean, the encoder offsets renormalised to main's half-widths) then the env's new ones (delay, gains, strength,
    torso mass); the loaded world model and critics are the source's widened (zero weights on the new columns, the
    offsets' columns rescaled), the actors the source's, before the first update; the fine-tune trains, and its
    resume keeps its layout."""
    import csv
    import math

    from pretraining.supermimic.sim_context import widen_input_weight
    from test_train_smoke import TRAIN_ENVS, _train, _train_setup

    new = [("action_delay", 1), ("actuator_gain", 27), ("actuator_strength", 27), ("torso_mass", 1)]
    _, config = _train_setup(tmp_path)
    common = ["--config", config, "--motion-dir", tmp_path / "motions", "--num-envs", TRAIN_ENVS, "--plant", "main"]
    _train(tmp_path, *common, "--run-name", "base", "--max-iterations", 6)
    base = tmp_path / "runs" / "base"
    layout = json.loads((base / "run.json").read_text())["sim_context"]
    assert [(e["name"], e["dim"]) for e in layout] == [("foot_friction", 1), ("base_com", 3), ("encoder_bias", 27)]

    # resume_warmup (2) iterations of collection only: the networks it saves are the ones it loaded
    proc = _train(tmp_path, *common, "--fine-tune", "--run-name", "ft0", "--fine-tune-from", base,
                  "--max-iterations", 2)
    assert ("appended action_delay, actuator_gain, actuator_strength, torso_mass" in proc.stdout_text
            and "base_com held at the source run's mean" in proc.stdout_text
            and "renormalised encoder_bias" in proc.stdout_text), proc.stdout_text[-4000:]
    ft0 = tmp_path / "runs" / "ft0"
    meta = json.loads((ft0 / "run.json").read_text())
    ft_layout = meta["sim_context"]
    assert (meta["plant"], meta["task"]) == ("main", FINETUNE_TASK)
    assert [(e["name"], e["dim"]) for e in ft_layout] == [(e["name"], e["dim"]) for e in layout] + new
    assert ft_layout[:2] == layout[:2]  # friction (another centre) and COM (held) keep the source's normalisation
    half = [(hi - lo) / 2 for lo, hi in ft_layout[2]["ranges"]]
    assert max(half) == pytest.approx(0.04, abs=1e-6) and min(half) == pytest.approx(0.01, abs=1e-6)
    ours_wm, base_wm = (torch.load(d / "world_model.pt", map_location="cpu", weights_only=False) for d in (ft0, base))
    ctx_b, ctx_f = base_wm["sim_context"], ours_wm["sim_context"]
    torch.testing.assert_close(ctx_f["mean"][:31], ctx_b["mean"])
    new_mean = ctx_f["mean"][31:]  # fixed constants: delay 5, gains 1 (the four ankle joints 0.85), strength 1, mass 1
    torch.testing.assert_close(new_mean[[0, -1]], torch.tensor([5.0, 1.0]))
    torch.testing.assert_close(new_mean[28:55], torch.ones(27))
    gain_mean = new_mean[1:28]
    assert int(torch.isclose(gain_mean, torch.tensor(0.85)).sum()) == 4
    assert int(torch.isclose(gain_mean, torch.tensor(1.0)).sum()) == 23
    torch.testing.assert_close(ctx_f["scale"][31:32], torch.tensor([5.0]))
    torch.testing.assert_close(ctx_f["scale"][-1:], torch.tensor([3.0]))
    flat_ours, flat_theirs = _flatten(ours_wm), _flatten(base_wm)
    assert flat_ours.keys() == flat_theirs.keys()
    for k in flat_ours:
        if "normalizer" in k or k.startswith(".sim_context"):  # running statistics: collection updates them
            continue
        theirs = flat_theirs[k]
        if k == ".dynamics.mlp.0.weight":
            theirs = widen_input_weight(theirs, ctx_b["mean"], ctx_b["scale"], ctx_f["mean"], ctx_f["scale"])
            assert flat_ours[k][:, -56:].abs().max() == 0
        torch.testing.assert_close(flat_ours[k], theirs, msg=f"world_model.pt: {k}")
    ours_c, base_c = (torch.load(d / "critics.pt", map_location="cpu", weights_only=False) for d in (ft0, base))
    for a in base_c:
        for k, v in base_c[a].items():
            if k == "0.weight":
                v = widen_input_weight(v, ctx_b["mean"], ctx_b["scale"], ctx_f["mean"], ctx_f["scale"])
            torch.testing.assert_close(ours_c[a][k], v, msg=f"critics.pt: {a} {k}")
    ours_a, base_a = (_flatten(torch.load(d / "actors.pt", map_location="cpu", weights_only=False))
                      for d in (ft0, base))
    assert ours_a.keys() == base_a.keys()
    for k in ours_a:
        if "normalizer" not in k:
            torch.testing.assert_close(ours_a[k], base_a[k], msg=f"actors.pt: {k}")

    _train(tmp_path, *common, "--fine-tune", "--run-name", "ft1", "--fine-tune-from", base, "--max-iterations", 6)
    ft1 = tmp_path / "runs" / "ft1"
    with open(ft1 / "progress.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["iteration"]) >= 6]
    losses = [float(r["Loss/state"]) for r in rows if r.get("Loss/state")]
    assert losses and all(math.isfinite(v) for v in losses)
    ft1_layout = json.loads((ft1 / "run.json").read_text())["sim_context"]
    assert ft1_layout == ft_layout
    stopped = json.loads((ft1 / "checkpoint.json").read_text())["iteration"]
    proc = _train(tmp_path, "--resume", ft1, "--max-iterations", stopped + 4)
    assert "appended none" in proc.stdout_text and "renormalised none" in proc.stdout_text
    assert json.loads((ft1 / "run.json").read_text())["sim_context"] == ft_layout


def _flatten(tree, prefix=""):
    """{dotted key: tensor} of a nested dict of tensors."""
    if isinstance(tree, torch.Tensor):
        return {prefix: tree}
    if isinstance(tree, dict):
        return {k: v for key, sub in tree.items() for k, v in _flatten(sub, f"{prefix}.{key}").items()}
    return {}
