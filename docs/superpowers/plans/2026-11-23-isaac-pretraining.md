# Isaac Lab Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the H1-2's legs and upper-body tracking policies on the retargeted AMASS clips in Isaac Lab with SuperMimic's world-model method, from a self-contained `pretraining/` package.

**Architecture:** An Isaac Lab manager-based tracking env (`LocoManip-Pretrain-Track-v0`) that mirrors, term for term, the mjlab task `amass_s1` trained in, on `main`'s H1-2 USD with `h12_train`'s gains and scales. On top of it, SuperMimic's learner stack and `h12_train`'s two-agent runner, ported as mjlab-free torch code; the env is the only simulator-specific layer.

**Tech Stack:** Python 3.11, Isaac Lab (isaaclab 0.54.4), torch 2.7, tensordict, tyro-free argparse CLI, pytest. Conda env `env_isaaclab`.

**Spec:** `docs/superpowers/specs/2026-11-23-isaac-pretraining-design.md`

## Global Constraints

- Never import `supermimic` or `mjlab`; never require `mujoco`.
- Never commit motion data, runs, policies or anything derived from AMASS or SMPL-X. `/data/` is git-ignored; `git add` by path only (the tree holds untracked `paper/` and old submodule folders).
- Policy contract unchanged from `h12_train` (spec, "The policy contract"): 50 Hz, Unitree motor order, legs 12 actions, upper body 15 (torso first), 87-wide proprio frame, history length 1, 32-wide tanh latent, hidden sizes 2048/1024/512/256.
- The env is the one `amass_s1` trained in: anchor `pelvis`, physics step 0.005 s, decimation 4, events foot friction + torso COM + encoder bias only, terminations anchor position, anchor orientation, motion end, time-out. Where this plan and `data/ref/h12_train/src/h12_train/tasks.py` disagree, the code wins.
- New gym ids contain `LocoManip-`.
- Every ported file starts with a header naming the upstream file and commit: SuperMimic `7e94b07`, mjlab_research `6ea3e57`, h12_train `b175f1c` (branch `supermimic`). `pretraining/NOTICE` lists them. All Apache-2.0 except h12_train (this repository's).
- Reference sources on this machine: `third_party/supermimic/src/supermimic`, `third_party/mjlab_research/src/mjlab`, `data/ref/h12_train/src/h12_train` (read-only).
- Wall-time budget for the long run: 8 hours total (base 7.5 h, GOLEM-aware 0.5 h). State the plan to Max before starting it.
- Nothing is pushed unless Max asks. Never edit `/home/max/ALMI-Open`.
- Tests that need the simulator are marked `@pytest.mark.sim` and launch one `AppLauncher(headless=True)` per session (`tests/conftest.py`); the rest run without Isaac.
- Run everything as `conda run -n env_isaaclab ...` from the repository root.

## Review Focus

1. A motion directory that is empty, missing, or holds a clip at another rate or joint order: the loader refuses with the file name and the reason, before the simulator starts. (Task 2)
2. A clip shorter than the goal window (10 frames at stride 5): goal indices clamp to the clip's last frame and never read the next clip. (Task 2)
3. Fewer envs than the config's batch sizes assume (a small smoke run, or the memory fallback): training runs, and the run's `run.json` records the env count actually used. (Task 7)
4. Resuming a run after the wall-time stop: `--resume` continues from `checkpoint.pt` with the settings in `run.json` and the iteration count carries on. (Task 7)
5. Loading an `actors.pt` whose shapes do not match (another action space or goal layout): refused with both shapes named, never partially loaded. (Task 6)

---

## File Structure

```
pretraining/
  README.md  NOTICE  pyproject.toml
  configs/base.yaml  configs/golem_aware.yaml
  pretraining/
    __init__.py            registers the gym ids on import
    constants.py           joint order, gains, limits, action scales, default pose, tracked bodies (from h12_constants.py, tasks.py)
    robot.py               ArticulationCfg: main's USD + constants
    kinematics.py          UrdfTree, ForwardKinematics (torch)
    motions.py             MotionLoader, check_motion_dir
    math.py                the quaternion helpers the ported code needs, from isaaclab.utils.math where they exist
    tasks/
      __init__.py
      commands.py          MotionCommand, MotionCommandCfg
      observations.py  terminations.py  events.py  metrics.py
      track_env_cfg.py     TrackEnvCfg, TrackEnvCfg_PLAY
    supermimic/            ported: models.py replay_buffer.py representations.py sim_context.py
                           learners/{learner_utils,policy_learner,wm_learner}.py
                           envs/{world_model_env,motion_tracking_env}.py
                           runners/{base_runner,reset_sampling}.py
    marl/                  ported: actors.py config.py learner.py representation.py rewards.py runner.py runs.py wm_env.py
    scripts/train.py  scripts/crosscheck.py  scripts/play.py
  tests/
    conftest.py  standins.py  test_constants.py  test_kinematics.py  test_motions.py
    test_track_env.py  test_supermimic_port.py  test_two_agent.py  test_train_smoke.py
```

---

### Task 1: Package skeleton, constants and robot config

**Files:**
- Create: `pretraining/pyproject.toml`, `pretraining/NOTICE`, `pretraining/pretraining/__init__.py`, `pretraining/pretraining/constants.py`, `pretraining/pretraining/robot.py`, `pretraining/tests/conftest.py`, `pretraining/tests/test_constants.py`

**Interfaces:**
- Produces, in `constants.py`: `JOINT_NAMES: tuple[str, ...]` (27, Unitree motor order, equal to `locomanipulation_game.assets.h1_2.ALL_JOINTS_NAMES`), `LEG_JOINTS = slice(0, 12)`, `UPPER_JOINTS = slice(12, 27)`, `DEFAULT_JOINT_POS: dict[str, float]`, `KP`, `KD`, `EFFORT`, `ARMATURE`, `ACTION_SCALE: dict[str, float]`, `TARGET_LIMITS: dict[str, tuple[float, float]]`, `ESTOP_LIMITS`, `TRACKED_BODIES: tuple[str, ...]` (14), `ANCHOR_BODY = "pelvis"`, `IMU_BODY = "torso_link"`, `SIM_DT = 0.005`, `DECIMATION = 4`. Values copied from `data/ref/h12_train/src/h12_train/robot/h12_constants.py` and `tasks.py`.
- Produces, in `robot.py`: `H1_2_PRETRAIN_CFG: ArticulationCfg` (main's `H1_2_MAGPIE_USD` spawn, self-collision on, `ImplicitActuatorCfg` per joint group with `constants` kp/kd/effort/armature, no joint damping or friction, gripper hinges held closed by a stiff actuator), and `MASSLESS_FRAME_FIX` (main's start-up event term that zeroes the four 1 kg frames, reused from `locomanip_marl`).

- [ ] **Step 1: Write the failing tests** in `tests/test_constants.py`

```python
def test_joint_order_is_mains():
    from locomanipulation_game.assets.h1_2 import ALL_JOINTS_NAMES
    assert list(constants.JOINT_NAMES) == ALL_JOINTS_NAMES

def test_action_scale_is_quarter_effort_over_kp_capped():
    for j in constants.JOINT_NAMES:
        assert constants.ACTION_SCALE[j] == pytest.approx(min(0.25, 0.25 * constants.EFFORT[j] / constants.KP[j]), abs=1e-6)

def test_values_match_the_h12_train_export():
    # data/runs/amass_s2_golem/export/export.yaml, when present (skipped otherwise)
    cfg = yaml.safe_load(EXPORT.read_text())
    assert cfg["joint_names"] == list(constants.JOINT_NAMES)
    np.testing.assert_allclose(cfg["kp"], [constants.KP[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["kd"], [constants.KD[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["default_joint_pos"], [constants.DEFAULT_JOINT_POS[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["action_scale"], [constants.ACTION_SCALE[j] for j in constants.JOINT_NAMES], atol=1e-6)
    for j, (lo, hi) in cfg["target_limits"].items():
        assert constants.TARGET_LIMITS[j] == pytest.approx((lo, hi), abs=1e-6)
```

- [ ] **Step 2:** Run `conda run -n env_isaaclab python -m pytest pretraining/tests/test_constants.py -q`; expected: import error.
- [ ] **Step 3:** Write `pyproject.toml` (package `pretraining`, no dependencies beyond the env's), `constants.py`, `robot.py`, `NOTICE`, and `conftest.py` with a session fixture `sim_app` that starts `AppLauncher(headless=True)` once and a `sim` marker.
- [ ] **Step 4:** `pip install -e pretraining` in `env_isaaclab`; run the tests; expected: pass (the export test passes on this machine).
- [ ] **Step 5:** Commit `pretraining/pyproject.toml pretraining/NOTICE pretraining/pretraining pretraining/tests`.

### Task 2: Kinematics and the motion loader

**Files:**
- Create: `pretraining/pretraining/math.py`, `kinematics.py`, `motions.py`, `tests/standins.py`, `tests/test_kinematics.py`, `tests/test_motions.py`

**Interfaces:**
- Produces: `kinematics.UrdfTree.from_file(path, joint_names) -> UrdfTree` (bodies in depth-first order with parent index, joint origin position and quaternion, joint axis, joint index or -1 for fixed joints); `kinematics.ForwardKinematics(tree, body_names, device)` with `__call__(root_pos, root_quat, joint_pos, root_lin_vel=None, root_ang_vel=None, joint_vel=None) -> (pos, quat, lin_vel, ang_vel)` in the world frame, shapes `(..., B, 3|4)`, the call signature of `mjlab.utils.forward_kinematics.ForwardKinematics.__call__`; `body_names` property.
- Produces: `motions.MotionLoader(motion_dir | list[Path], fk, device)` with the attributes mjlab's `MotionLoader` has (`joint_pos`, `joint_vel`, `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `motion_starts`, `motion_ends`, `time_step_total`, `fps`, `fk`, `files`); `motions.check_motion_dir(path) -> list[Path]`.
- `URDF` path: `CL_ASSETS_DIR / "ros_assets" / "h1_2_magpie.urdf"`.
- `standins.write_clips(dir, n=3, frames=120, seed=0)`: procedurally generated clips in the spec's `.npz` format (smooth joint sinusoids inside the limits, a pelvis that sways and yaws), for every test.

- [ ] **Step 1: Write the failing tests**

```python
# test_kinematics.py
def test_default_pose_wrists_mirror_in_y():           # at DEFAULT_JOINT_POS the left and right wrist positions differ only in the sign of y (1e-4 m)
def test_velocities_match_finite_differences():      # 1e-3 rad step, body lin/ang vel within 1e-3
def test_batch_shapes():                              # (4, 7, 27) joints -> (4, 7, B, 3)
@pytest.mark.sim
def test_matches_isaac_body_poses(sim_app):           # random joint pose written to the articulation, body_pos_w/quat_w within 1e-4 m / 1e-4 rad for TRACKED_BODIES

# test_motions.py
def test_loads_clips_and_concatenates():              # 3 stand-in clips: time_step_total == sum, motion_ends cumulative
def test_refuses_empty_or_missing_dir():              # ValueError naming the path
def test_refuses_wrong_fps_or_joint_order():          # ValueError naming the file and the reason
def test_goal_window_clamps_at_clip_end():            # representations.goal_window_indices on a 20-frame clip never crosses into the next clip (added in Task 5's file; test lives here once Task 5 lands)
```

- [ ] **Step 2:** Run both test files; expected: import errors.
- [ ] **Step 3:** Implement `math.py` (re-export `isaaclab.utils.math` functions under the names `mjlab.utils.lab_api.math` uses in the ported files; add any that are missing), `kinematics.py` (parse the URDF with `xml.etree`; fixed joints merge into their parent's transform; the FK loop is mjlab's, per body in tree order), `motions.py`.
- [ ] **Step 4:** Run the tests; expected: pass (`sim` test included).
- [ ] **Step 5:** Commit.

### Task 3: The Isaac Lab tracking env

**Files:**
- Create: `pretraining/pretraining/tasks/{__init__,commands,observations,terminations,events,metrics,track_env_cfg}.py`, `tests/test_track_env.py`
- Modify: `pretraining/pretraining/__init__.py` (gym registration)

**Interfaces:**
- Consumes: `constants`, `robot.H1_2_PRETRAIN_CFG`, `motions.MotionLoader`, `kinematics.ForwardKinematics`.
- Produces: gym ids `LocoManip-Pretrain-Track-v0`, `LocoManip-Pretrain-Track-Play-v0` (entry point `isaaclab.envs:ManagerBasedRLEnv`, `env_cfg_entry_point` `TrackEnvCfg` / `TrackEnvCfg_PLAY`).
- `MotionCommand` (an `isaaclab.managers.CommandTerm`) ported from `mjlab/tasks/tracking/mdp/commands.py`: same cfg fields (`motion_file`, `anchor_body_name`, `body_names`, `pose_range`, `velocity_range`, `joint_position_range`, `sampling_mode` in `adaptive | uniform | start | external`, the `adaptive_*` fields), same public attributes and properties (`motion`, `time_steps`, `body_indexes`, `anchor_pos_w`, `robot_anchor_pos_w`, `body_pos_relative_w`, ...), same reset (state written to the articulation with `write_root_state_to_sim` / `write_joint_state_to_sim`). The viz code is dropped.
- Observation groups, exactly as `apply_gencoord` builds them: `actor` (projected gravity and angular velocity of `IMU_BODY`, `joint_pos_rel` with the encoder bias, `joint_vel_rel`, `last_action`; corruption off: the learner adds the noise), `critic` (mjlab's tracking critic terms), `full_state` (`root_pos` minus env origin, `root_ori`, `joint_pos`, `root_lin_vel`, `root_ang_vel`, `joint_vel`), `goal` (`supermimic/tasks/env_cfgs.py::_goal_obs_group`).
- Action: one `JointPositionActionCfg` over `JOINT_NAMES`, `scale=ACTION_SCALE`, `use_default_offset=True`, `clip=TARGET_LIMITS`, `preserve_order=True`.
- Events: `foot_friction`, `base_com` (on `torso_link`), `encoder_bias`, with mjlab's ranges; the massless-frame fix at start-up. Terminations: `time_out`, `motion_end`, `anchor_pos`, `anchor_ori` with mjlab's thresholds. Rewards: mjlab's tracking rewards (the learner does not use them; they stay so the env's reward manager and `max_step_reward` exist) with `joint_limit` weight -2.0 and no `self_collisions`. Metrics: the ten `Metrics/motion/error_*` terms written to `extras["log"]`.
- `TrackEnvCfg.sim.dt = 0.005`, `decimation = 4`, `episode_length_s = 10.0`, flat ground plane, `scene.num_envs` from the CLI.

- [ ] **Step 1: Write the failing tests** (`@pytest.mark.sim`, 16 envs, stand-in clips)

```python
def test_registered_and_listed():                     # both ids in gym registry, contain "LocoManip-"
def test_observation_shapes():                        # actor (N, 87); full_state (N, 3+4+27+3+3+27); goal dims equal representations.wm_goal_dim
def test_reset_puts_robot_on_the_clip_frame():        # sampling_mode "start", zero pose/velocity range: joint_pos, root pose equal the clip's frame 0 within 1e-4
def test_kinematic_replay_reads_zero_error():         # each step write the clip's state to the sim; every Metrics/motion/error_* < 1e-3
def test_actor_frame_matches_h12_train_layout():      # gravity (0,0,-1) upright; q - default zero at default pose; last action equals the action just applied
def test_targets_are_clipped_to_golem_limits():       # action +100 on every joint: processed target == TARGET_LIMITS upper
def test_motion_end_terminates_and_resets():
```

- [ ] **Step 2:** Run; expected: fail on the missing ids.
- [ ] **Step 3:** Port and write the env files.
- [ ] **Step 4:** Run the tests; expected: pass. Also run `conda run -n env_isaaclab python scripts/list_envs.py` after `import pretraining` is added to it only if the listing needs it (otherwise leave `scripts/` alone and note the import in the README).
- [ ] **Step 5:** Commit.

### Task 4: Port SuperMimic's models, replay buffer, learners and world-model env

**Files:**
- Create: `pretraining/pretraining/supermimic/` (`models.py`, `replay_buffer.py`, `representations.py` (GenCoord only), `sim_context.py`, `learners/`, `envs/world_model_env.py`, `envs/motion_tracking_env.py`, `runners/reset_sampling.py`, `runners/base_runner.py` = `BaseRunnerCfg`, `WorldModel`, `build_world_model`, `Ratio`, `BaseRunner` from `supermimic_runner.py`), `tests/test_supermimic_port.py`

**Interfaces:**
- Consumes: the env of Task 3 through these names only: `env.num_envs`, `env.device`, `env.step_dt`, `env.cfg.decimation`, `env.max_episode_length`, `env.scene["robot"].data.{soft_joint_pos_limits,default_joint_pos}`, `env.scene.env_origins`, `env.command_manager.get_term("motion")`, `env.termination_manager`, `env.action_manager`, `env.cfg.{events,terminations,rewards}`, `env.observation_manager.compute()` (groups as a dict), `env.step`, `env.reset`.
- Changes from upstream, and nothing else: imports (`pretraining.math`, `pretraining.kinematics`, `pretraining.motions`); `sim_context` reads foot friction, torso COM offset and encoder bias from the Isaac articulation (`root_physx_view` material properties and COMs, the bias the `encoder_bias` event stored) instead of `env.sim.model`; the env wrapper calls go through a small `EnvAdapter` (in `base_runner.py`) exposing `get_observations() -> TensorDict`, `step(action) -> (TensorDict, reward, dones, extras)`, `num_actions`, `unwrapped`; CUDA-graph capture (`CapturedUpdate`) kept behind a flag that defaults to off if capture fails on this torch build; wandb optional (`WANDB_MODE`, off when the package is missing).
- Produces: the same class and function names as upstream, importable from `pretraining.supermimic...`.

- [ ] **Step 1: Write the failing tests** (no simulator)

```python
def test_dynamics_forward_shapes():
def test_replay_buffer_causal_sampling_never_crosses_a_done():
def test_gencoord_roundtrip():                        # full_state -> representation state -> FK bodies equals the stand-in clip's bodies
def test_goal_window_indices_clamp_to_the_clip():     # 20-frame clip, 10 goals at stride 5
def test_world_model_learner_reduces_loss_on_a_linear_system():   # 200 updates on synthetic transitions, loss drops by 10x
def test_bptt_policy_learner_improves_return_in_a_known_world_model():
```

- [ ] **Step 2:** Run; expected: import errors.
- [ ] **Step 3:** Port file by file with the upstream header; delete MaxCoord, the PPO runner, the exporter and the viewers.
- [ ] **Step 4:** Run; expected: pass.
- [ ] **Step 5:** Commit.

### Task 5: Port the two-agent stage-1 code from h12_train

**Files:**
- Create: `pretraining/pretraining/marl/{__init__,actors,config,learner,representation,rewards,runner,runs,wm_env}.py`, `tests/test_two_agent.py`

**Interfaces:**
- Consumes: Task 4's modules, Task 3's env, `constants`.
- Stage-1 parts only: `build_imitation_actor`, `ImitationRewardCfg`, `EstopPenaltyCfg` (velocity, torque, position, `position_margin`), `TorsoImuGenCoordRepresentation`, `ClampedDynamics`, `build_world_model`, `TwoAgentLearner`, `ImitationWorldModelEnv`, `TwoAgentRunnerCfg`, `TwoAgentRunner`, `ImitationRunner`, `imitation_goal_dims`, `resume_info`, `runs.*`. Stage-2 classes (`CommandActor`, `CommandRunner`, `CommandWorldModelEnv`, `goals.py`, `LatentPrior`) are left out.
- Produces: `ImitationRunner(env, cfg: TwoAgentRunnerCfg, rewards: ImitationRewardCfg, run_dir: Path, motion_files: list[str])` with `.run()`, `.save()`, and `load_actors(run_dir, device) -> dict[str, nn.Module]` that refuses a state dict whose shapes differ, naming both shapes.
- Run directory layout as `h12_train`: `run.json`, `progress.csv`, `world_model.pt`, `actors.pt`, `critics.pt`, `checkpoint.pt`, `checkpoint.json`, `checkpoints/`.
- Runs root: `$PRETRAINING_RUNS_DIR`, default `<repo>/data/pretraining/runs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_actor_shapes_and_latent_is_tanh_32():
def test_goal_dims_split_between_agents_as_h12_train():
def test_estop_penalty_values():                      # hand-computed cases from data/ref/h12_train/tests/test_two_agent.py
def test_load_actors_refuses_other_shapes():          # ValueError naming both shapes
def test_actors_pt_from_h12_train_loads():            # data/runs/amass_s1/actors.pt -> forward on zeros gives finite (12,) and (15,); skipped when absent
```

- [ ] **Step 2:** Run; expected: import errors.
- [ ] **Step 3:** Port; carry over the applicable cases of `data/ref/h12_train/tests/test_two_agent.py` and `test_runs.py`.
- [ ] **Step 4:** Run; expected: pass, including the `amass_s1` load on this machine.
- [ ] **Step 5:** Commit.

### Task 6: Cross-check script

**Files:**
- Create: `pretraining/pretraining/scripts/__init__.py`, `scripts/crosscheck.py`, `scripts/play.py`

**Interfaces:**
- `python -m pretraining.scripts.crosscheck --run <h12_train or pretraining run dir> --motion-dir <dir> [--clips N=40] [--seed 0] [--num-envs N] [--out <dir>]`: loads the run's actors, rolls each chosen clip from its first frame for its length (cap 10 s) with no perturbation and no noise, and writes `crosscheck.csv` (per clip: fell, time upright, mean body position error, joint position error, anchor position error, the `Safety/*` shares) and a summary line.
- `play.py`: the same rollout on named clips, with `--video` using Isaac Lab's `RecordVideo` wrapper.
- A fall: the pelvis below 0.5 m or tilted more than 60 degrees (as `render_stage1.py`).

- [ ] **Step 1:** Write `tests/test_train_smoke.py::test_crosscheck_runs_on_standins` (`sim`): an untrained actor on 3 stand-in clips produces a csv with 3 rows and finite numbers.
- [ ] **Step 2:** Run; expected: fail.
- [ ] **Step 3:** Implement both scripts.
- [ ] **Step 4:** Run the test; expected: pass.
- [ ] **Step 5:** Run the real cross-check: `python -m pretraining.scripts.crosscheck --run data/runs/amass_s1 --motion-dir data/motions/h12_amass --clips 40 --out data/pretraining/crosscheck/amass_s1` and the same for `amass_s1_golem_aware`. Compare with the MuJoCo numbers in `data/runs/amass_s1/renders/metrics.csv`. Apply the spec's cross-check rule (more than a fifth of clips fall: up to half a day on a port bug or model mismatch, then record the gap).
- [ ] **Step 6:** Commit the scripts and the test (never the outputs).

### Task 7: Training script and configs

**Files:**
- Create: `pretraining/pretraining/scripts/train.py`, `pretraining/configs/base.yaml`, `pretraining/configs/golem_aware.yaml`
- Modify: `tests/test_train_smoke.py`

**Interfaces:**
- `python -m pretraining.scripts.train --config pretraining/configs/base.yaml --motion-dir <dir> --run-name <name> [--num-envs N] [--max-iterations N] [--max-seconds S] [--fine-tune-from <run dir>]` and `--resume <run dir> [--max-seconds S] [--max-iterations N]`. An existing run directory is refused unless `--resume`.
- `base.yaml`: every runner and reward value of `data/runs/amass_s1/run.json` (spec, "The two runs"), `num_envs: 5120`, `max_iterations: 320000`. `golem_aware.yaml`: the same with `rewards.estop.position: -20.0`, `rewards.estop.position_margin: 0.05`, `max_iterations: 20000`, to be used with `--fine-tune-from`.
- `run.json` records the env count actually used, the config file's content, the motion file list and the seed.

- [ ] **Step 1: Write the failing tests** (`sim`)

```python
def test_smoke_training_writes_a_run(tmp_path):       # 64 envs, 30 iterations, stand-in clips: run.json, progress.csv, actors.pt exist; Loss/state finite
def test_resume_continues_the_iteration_count(tmp_path):   # 20 iterations, stop, --resume to 30: progress.csv's last iteration is 29
def test_existing_run_dir_is_refused(tmp_path):
def test_run_json_records_the_env_count(tmp_path):
```

- [ ] **Step 2:** Run; expected: fail.
- [ ] **Step 3:** Implement `train.py` and the configs.
- [ ] **Step 4:** Run; expected: pass.
- [ ] **Step 5:** Commit.

### Task 8: Measure, then the two full runs

- [ ] **Step 1:** Short real run: `--num-envs 5120 --max-iterations 3000` on the AMASS clips. Read `peak_gpu_gb`, `s_per_iteration`, `env_steps_per_s`, `Loss/state` and `Metrics/motion/error_body_pos` from `progress.csv`. If it runs out of memory, halve the envs until it fits (4096, 3072, 2048).
- [ ] **Step 2:** Tell Max: the env count, the speed, the iterations 7.5 h will reach, and that the base run is starting.
- [ ] **Step 3:** Base run in the background: `--config base.yaml --run-name isaac_s1 --max-seconds 27000`. Check it after 10 minutes (losses falling), then leave it.
- [ ] **Step 4:** GOLEM-aware run: `--config golem_aware.yaml --fine-tune-from data/pretraining/runs/isaac_s1 --run-name isaac_s1_golem_aware --max-seconds 1800`.
- [ ] **Step 5:** Run `crosscheck` on both Isaac runs with the same 40 clips and seed as Task 6.

### Task 9: README and results

**Files:**
- Create: `pretraining/README.md`
- Modify: `README.md` (one paragraph and a link), `CLAUDE.md` (the `pretraining/` entry under "This repo")

- [ ] **Step 1:** Write the README: what this is and how it differs from `h12_train`; setup; the data format table and default paths (spec, "Data format"), including where the two MuJoCo-trained policies are expected; the commands of Tasks 6 to 8; the known differences from `h12_train` with the measured cross-check numbers; a results table (the four policies: iterations, wall time, env count, body position error, joint position error, falls on the 40 clips, `Safety/*` shares); what was not done (stage 2, fine-tuning).
- [ ] **Step 2:** Run the whole test suite: `conda run -n env_isaaclab python -m pytest pretraining/tests -q`; expected: pass.
- [ ] **Step 3:** Commit. Do not push.
