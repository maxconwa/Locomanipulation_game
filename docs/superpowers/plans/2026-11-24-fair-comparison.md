# Fair Comparison (Pure RL / PULSE + MAPPO / PULSE + World-Model Gradients) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put pure RL (A), PULSE + MAPPO (B) and PULSE + world-model gradients (C) on one POMDP, main's
`LocoManip-Marl-Direct-v0` with the arms agent driving the torso, and score all three at 40k steps with one eval code.

**Architecture:** Five pieces. (1) A torso action for the arms agent behind a cfg field, on in a new id
`LocoManip-Marl-Torso-v0`. (2) A `Plant` abstraction in `pretraining/` so stage 1 can run on main's plant (100 Hz,
explicit PD, main's gains and default pose) with the 50 Hz clips resampled to 100 Hz. (3) `LocoManip-Marl-Pulse-v0`
rebuilt on the Torso task's cfg with the plant overrides removed. (4) h12_train's `CommandRunner` ported into
`pretraining/marl/`, its command encoder an LSTM fed main's command and a torch model of main's actor observation,
its data collected in the piece-3 env. (5) The Isaac and MuJoCo scorecards and `get_video` take a latent policy.
Then the runs, which the controller runs itself.

**Tech Stack:** Python 3.11, Isaac Sim 5.1, Isaac Lab 0.54.4, torch 2.7, skrl MAPPO, tensordict, pytest. Conda env
`env_isaaclab`; MuJoCo evals in the mujoco python named in `evals/README.md`.

**Spec:** `docs/superpowers/specs/2026-11-24-fair-comparison-design.md`

## Global Constraints

- Nothing AMASS/SMPL-X-derived is committed: no clips, runs, policies, reports that list clip names, and no clip names
  in code, tests, docs or commit messages. `git add` by path only (`paper/`, `third_party/*` are untracked).
- Never edit anything under `/home/max/ALMI-Open`, `data/ref`, `third_party/`, `paper/`. Never touch
  `data/runs/amass_s1` (write-protected). Nothing is deleted from `data/`, runs or logs.
- `LocoManip-LegsOnly-v0`, `LocoManip-ArmsOnly-v0`, `LocoManip-Marl-Direct-v0` stay bit-identical in their resolved
  configs: wherever a shared file is touched, the task ends with the Task 1 cfg-dump compare against the baseline dump.
  `LocoManip-Pretrain-Track-v0` likewise (its dump joins the baseline).
- New gym ids contain `LocoManip-` (`scripts/list_envs.py:54`).
- Python: `/home/max/miniconda3/envs/env_isaaclab/bin/python` (below: `$PY`). One GPU process at a time (16 GB card);
  every simulator command under `timeout -s KILL <s>`; simulator tests are `@pytest.mark.sim`, one `AppLauncher` per
  session (`pretraining/tests/conftest.py`).
- Scripts that close the Isaac app after writing results flush and leave with `os._exit(0)` (`app.close()` has hung).
- Commits only, no push. Commit trailer:
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_015EG7pCb2mxJmaeyLUceRoa`
- Lines up to 120 columns; comments say why, as the surrounding code does.
- Runs (Tasks 14-23) are the controller's, never an implementer subagent's. Every run is stated to Max before it starts.

## Review Focus

1. **Explicit PD on main's gains at stage 1.** The first PULSE run's coupled damping mode (eigenvalue of dt M⁻¹ Kd)
   was 1.28 with pretraining's arm kd. Main's arm kd are lower (2 / 1 on yaws, elbows, wrists) but main's armature is
   0.01 everywhere: Task 3's eigenvalue check must report < 1.0 at nominal and under the gain draws (x1.2 kd) before
   the track env goes explicit. Test: `test_main_plant_damping_eigenvalues_below_one` (Task 3).
2. **A stage-1 run of one plant decoded on the other** (the zero-policy fall rate of 23% came from a plant
   mismatch). `PulseLatentAction` and `CommandRunner` refuse a stage-1 run whose `run.json` plant is not the env's,
   naming both. Tests: `test_pulse_refuses_a_run_of_another_plant` (Task 7), `test_command_runner_refuses_plant_mismatch`
   (Task 10).
3. **Wrist frames.** Stage 1 tracks `*_wrist_yaw_link` + `GRASP_OFFSET` (a translation; `pretraining/marl/rewards.py:52`),
   main's goals are `*_wrist_yaw_link` poses (`locomanip_marl_env_cfg.py:67-68`). The 120-140° errors are not
   explained by that offset. Task 15's clip-replay check measures FK agreement and the orientation coverage of main's
   target table by the clips; Task 9's reward test pins C's wrist terms to main's frame, without the grasp offset.
4. **Clip resampling edges:** a clip of 1 or 2 frames, and the quaternion double cover (q and -q between frames):
   the resampler keeps unit norm and takes the short arc. Test: `test_resample_takes_the_short_arc_and_keeps_edges`
   (Task 4).
5. **Policy std at its bound.** B's legs std sat at 0.5 under entropy 0.01 in the first run. The Pulse cfg keeps
   the bounds; the TensorBoard std is part of each run task's report, and a run whose std sits at the bound for the
   last 20% is reported as such, not re-tuned silently.

---

## File Structure

```
scripts/dump_task_cfgs.py                          resolved env cfg per id to json; --compare
source/.../locomanip_marl/
  locomanip_marl_env_cfg.py   + arms_drive_torso field; TORSO_ACTION_SCALE
  marl_torso_env_cfg.py       MarlTorsoEnvCfg: MarlDirect with the torso on (new id LocoManip-Marl-Torso-v0)
  marl_pulse_env_cfg.py       rebuilt on MarlTorsoEnvCfg, main's plant
  mdp/pulse_actions.py        plant from run.json; refuse mismatch
  agents/__init__.py          skrl_mappo_pulse_cfg at main's discount
  __init__.py                 registers LocoManip-Marl-Torso-v0
pretraining/pretraining/
  plants.py                   Plant, PRETRAIN, MAIN, get_plant
  robot.py                    robot_cfg(plant)
  motions.py                  resample_clip, MotionLoader(fps=)
  tasks/track_env_cfg.py      TrackMainEnvCfg, TrackMainFineTuneEnvCfg (+ new ids)
  tasks/delay.py              per-env 0-10 substep target delay for the track env
  marl/pulse.py               PulseDecoders(plant)
  marl/command_law.py         main's command law, segment process (torch, no sim)
  marl/main_rewards.py        main's velocity/wrist kernels + GOLEM e-stop on WM state
  marl/main_obs.py            MainActorObs: main's actor obs from WM state
  marl/command_actor.py       RecurrentCommandActor
  marl/marl_source.py         MarlEnvSource: the piece-3 env as the runner's data source
  marl/command_runner.py      CommandRunner (from h12_train marl/runner.py:639-869)
  scripts/train_commands.py   stage 2 (C) CLI
  scripts/pd_stability.py     eigenvalues of dt M^-1 Kd per plant
  scripts/wrist_frames.py     clip-replay wrist check
  configs/main_plant.yaml  configs/main_plant_finetune.yaml  configs/commands.yaml
evals/isaac/latent.py           LatentPolicy: B (skrl in Pulse env) and C (command run) behind one act()
evals/mujoco/latent.py          decoder-in-the-loop MuJoCo policy
pretraining/scripts/export_decoders.py   TorchScript decoders + prior + plant json for MuJoCo
```

---

### Task 1: Resolved-cfg dump guard

**Files:**
- Create: `scripts/dump_task_cfgs.py`, `pretraining/tests/test_cfg_dump.py`

**Interfaces:**
- Produces: `dump(task_ids: list[str]) -> dict[str, dict]` (each id's env cfg through `isaaclab.utils.class_to_dict`,
  values made JSON-safe with `repr` for callables); CLI `--out <json> [--tasks ...]` (default the three main ids and
  `LocoManip-Pretrain-Track-v0`), `--compare <a.json> <b.json>` exits 1 and prints every differing key path.
  Baseline written once to `logs/cfg_dumps/baseline.json` (logs/ is git-ignored).

- [ ] **Step 1:** `@pytest.mark.sim test_dump_is_deterministic_and_compare_flags_a_change`: dump twice, compare equal;
  mutate `MarlDirectEnvCfg.lambda_legs` in the dict copy, compare reports exactly `LocoManip-Marl-Direct-v0.lambda_legs`.
- [ ] **Step 2:** run, expect FAIL (no module).
  `timeout -s KILL 600 $PY -m pytest pretraining/tests/test_cfg_dump.py -m sim -v`
- [ ] **Step 3:** implement; the script launches `AppLauncher(headless=True)`, loads cfgs with
  `isaaclab_tasks.utils.load_cfg_from_registry(id, "env_cfg_entry_point")`, writes, `os._exit(0)`.
- [ ] **Step 4:** test passes; write the baseline:
  `timeout -s KILL 600 $PY scripts/dump_task_cfgs.py --out logs/cfg_dumps/baseline.json`
- [ ] **Step 5:** commit `scripts/dump_task_cfgs.py pretraining/tests/test_cfg_dump.py`.

### Task 2: Torso action for the arms agent (piece 1, arm A)

**Files:**
- Modify: `locomanip_marl/locomanip_marl_env_cfg.py` (`MarlActionsCfg` :198, `LocoManipMarlEnvCfg` :574),
  `locomanip_marl/__init__.py`
- Create: `locomanip_marl/marl_torso_env_cfg.py`, `pretraining/tests/test_torso_task.py`

**Interfaces:**
- Produces: `TORSO_ACTION_SCALE = 0.25` (GOLEM `lmg_deploy/policies/h12/export.yaml:266`, `controller.py:26-28`:
  target = default + scale x action, the torso's action first of the arms'); `TORSO_ACTION = "torso_pos"`;
  `LocoManipMarlEnvCfg.arms_drive_torso: bool = False`; `MarlTorsoEnvCfg(MarlDirectEnvCfg)` with it True;
  gym id `LocoManip-Marl-Torso-v0` (MAPPO yaml as Direct's). With the field on: `actions.torso_pos =
  BoundedJointPositionActionCfg(joint_names=["torso_joint"], scale=0.25, use_default_offset=True,
  target_margin=-GOLEM_TARGET_CLIP, torque_headroom=TORQUE_HEADROOM)`; `agent_action_terms["arms"] =
  ["torso_pos", "arm_pos"]` (arms action 15 wide, torso first as GOLEM); arms and critic obs gain
  `torso_action = applied_action(torso_pos)`; arms rewards gain `torso_action_rate` (action_rate_l2 on `torso_pos`)
  at the arms' `action_rate` weight. Applied in `sync_derived` only when the field is True, so the default path adds
  nothing to the resolved cfg.

- [ ] **Step 1:** sim tests: `test_torso_task_arms_action_is_15_with_torso_first` (env with 2 envs: `action_spaces ==
  {"legs": 12, "arms": 15}`; arms action `[1, 0...]` moves the torso joint target to `default + 0.25` within the
  GOLEM clip); `test_torso_off_by_default` (`MarlDirectEnvCfg().arms_drive_torso is False`, no `torso_pos` attribute).
- [ ] **Step 2:** run, expect FAIL.
- [ ] **Step 3:** implement the field, the cfg class and the registration.
- [ ] **Step 4:** tests pass; then the guard:
  `timeout -s KILL 600 $PY scripts/dump_task_cfgs.py --out logs/cfg_dumps/t2.json && $PY scripts/dump_task_cfgs.py
  --compare logs/cfg_dumps/baseline.json logs/cfg_dumps/t2.json` → exit 0.
- [ ] **Step 5:** smoke: `timeout -s KILL 900 $PY scripts/skrl/train.py --headless --task LocoManip-Marl-Torso-v0
  --num_envs 64 --max_iterations 2` exits 0 and writes `params/task.txt`.
- [ ] **Step 6:** commit.

### Task 3: Plant abstraction and the PD stability check (piece 2)

**Files:**
- Create: `pretraining/pretraining/plants.py`, `pretraining/pretraining/scripts/pd_stability.py`,
  `pretraining/tests/test_plants.py`
- Modify: `pretraining/pretraining/robot.py` (keep `H1_2_PRETRAIN_CFG = robot_cfg(PRETRAIN)`)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Plant: name: str; sim_dt: float; decimation: int; explicit: bool;
  kp/kd/effort/armature/velocity_limit/action_scale/default_joint_pos: dict[str, float]; default_root_height: float;
  target_limits: dict[str, tuple[float, float]]`, property `step_dt`. `PRETRAIN` (constants.py values, implicit,
  0.005 x 4), `MAIN` (0.002 x 5, explicit `IdealPDActuatorCfg`; gains, efforts, velocity limits, armature 0.01 from
  `assets/h1_2.py:134-150`; default pose `h1_2.py:104-121`; root height `STANCE_PELVIS_HEIGHT` 0.9798;
  `action_scale = min(0.25, 0.25 * effort / kp)` as `constants.py:68-69`; target limits pretraining's
  `TARGET_LIMITS`). `get_plant(name: str) -> Plant`, refuses unknown names listing the known.
  `robot_cfg(plant: Plant) -> ArticulationCfg` (explicit → `IdealPDActuatorCfg` per `ACTUATOR_GROUPS`, else implicit;
  main's grippers group either way).
  `pd_stability.py --plant main [--kd-scale 1.2]`: one env, writes the default pose, reads
  `root_physx_view.get_generalized_mass_matrices()`, prints the largest eigenvalues of `dt * M^-1 * Kd` over the 27
  policy joints with the joints of their eigenvectors, exit 0 if max < 1.0, else 1.

- [ ] **Step 1:** tests: `test_pretrain_plant_is_constants` (every dict equals constants'); `test_main_plant_matches_
  main_articulation` (kp, kd, effort per joint equal to `H1_2_MAGPIE_CFG.actuators` resolved per name;
  `MAIN.step_dt == 0.01`); `test_main_action_scales` (torso 0.1, knees 0.1875, hip 0.2, ankle pitch 0.1875, wrists
  0.2375, shoulder pitch 0.04, elbows 0.05625); `@sim test_main_plant_damping_eigenvalues_below_one` (runs the
  check at kd x1.0 and x1.2, both < 1.0).
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement. **Step 4:** pass. If the eigenvalue test fails, stop and
  report the joints to the controller (making them implicit changes main's plant: Max decides).
- [ ] **Step 5:** cfg-dump compare (robot.py is shared by the existing track id) → exit 0. Commit.

### Task 4: Clips resampled to the env rate

**Files:**
- Modify: `pretraining/pretraining/motions.py` (`FPS` :21, `check_motion_dir` :25, `_check_clip` :39,
  `MotionLoader` :49)
- Test: `pretraining/tests/test_motions.py` (stand-in clips from `tests/standins.py`, never real ones)

**Interfaces:**
- Produces: `SOURCE_FPS = 50.0`; `resample_clip(clip: dict[str, np.ndarray], factor: int) -> dict[str, np.ndarray]`
  (positions and joint angles linear, quaternions slerp on the short arc, velocities linear, `fps` x factor,
  `(n-1)*factor + 1` frames); `MotionLoader(..., fps: float = SOURCE_FPS)` resamples when `fps / SOURCE_FPS` is an
  integer > 1, refuses a non-integer ratio naming both rates; `check_motion_dir` still requires 50 Hz files.

- [ ] **Step 1:** `test_resample_doubles_frames_and_keeps_originals` (even frames equal the source exactly);
  `test_resample_takes_the_short_arc_and_keeps_edges` (frame 1 = -q of frame 0's neighbour: midpoint unit-norm and
  within 1e-6 of the short-arc slerp; a 1-frame clip stays 1 frame; a 2-frame clip gives 3);
  `test_loader_refuses_non_integer_ratio` (fps 75).
- [ ] **Step 2-4:** fail, implement, pass. Existing `test_motions.py` unchanged and passing.
- [ ] **Step 5:** commit.

### Task 5: Track env on main's plant, base and fine-tune (piece 2)

**Files:**
- Modify: `pretraining/pretraining/tasks/track_env_cfg.py` (`TrackEnvCfg` :236), `tasks/__init__.py`,
  `pretraining/pretraining/scripts/train.py` (`TASK` :47, `plan` :83, `train` :154)
- Create: `tasks/delay.py`, `configs/main_plant.yaml`, `configs/main_plant_finetune.yaml`,
  `pretraining/tests/test_track_main.py`

**Interfaces:**
- Consumes: `Plant`, `robot_cfg`, `MotionLoader(fps=)`.
- Produces: `TrackMainEnvCfg(TrackEnvCfg)`: `robot_cfg(MAIN)`, `sim.dt 0.002`, decimation 5, action scale
  `MAIN.action_scale` with `use_default_offset` (target = main's default + scale x output), motion fps 100; proprio
  unchanged (87 wide: it reads the articulation's default pose and `step_dt`, `observations.py:59`,
  `representation.py:120-129`). `TrackMainFineTuneEnvCfg(TrackMainEnvCfg)`: main's events reused from
  `MarlEventCfg` (friction 0.1-1.25, torso mass -2/+4, `actuator_scales`, encoder offsets), joint noise as
  `joint_noise.py`, and `TargetDelay` (`tasks/delay.py`: per-env integer delay drawn at reset from
  `ACTION_DELAY_RANGE` (0, 10) substeps, applied to the joint targets each substep). Ids
  `LocoManip-Pretrain-Track-Main-v0`, `LocoManip-Pretrain-Track-Main-FineTune-v0`.
  `train.py --plant {pretrain,main}` (new runs; default pretrain), `--fine-tune` picks the FineTune id when the
  plant is main; `run.json` gains `"plant"` and `"task"`; `--resume` reads both from it; an old `run.json` without
  them means `pretrain`.
  `main_plant.yaml` = base.yaml with: `wm.horizon 16`, `policy.horizon 64`, `return_sampler.horizon 40`,
  `policy.gamma 0.995`, `return_sampler.gamma 0.995`, `policy.lam 0.975`, `max_iterations 640000`, `num_envs 2048`.
  `main_plant_finetune.yaml` = golem_aware.yaml with the same rate changes and `max_iterations 40000`.

- [ ] **Step 1:** `@sim test_track_main_step_dt_and_targets` (2 envs, stand-in clip at 50 Hz: `step_dt == 0.01`,
  motion has 2n-1 frames, a zero action holds main's default pose as the target); `@sim
  test_finetune_delay_draws_within_range`; `test_plan_records_plant_and_task` (sim-free, `plan()` on a tmp run dir);
  `test_resume_old_run_json_means_pretrain`.
- [ ] **Step 2-4:** fail, implement, pass. Run the whole `pretraining/tests` (sim and not) once.
- [ ] **Step 5:** cfg-dump compare → exit 0 (`LocoManip-Pretrain-Track-v0` unchanged). Commit.

### Task 6: Clip-replay wrist-frame check (tool)

**Files:**
- Create: `pretraining/pretraining/scripts/wrist_frames.py`, `pretraining/tests/test_wrist_frames.py`

**Interfaces:**
- Produces: `python -m pretraining.scripts.wrist_frames --motion-dir <dir> [--clips 40] --out <dir>`: kinematic
  replay (gravity off, root and joints written each step) on `robot_cfg(MAIN)`; writes `wrist_frames.json` with
  (a) `fk_pos_err_m`, `fk_rot_err_deg` (p50/p95/max): Isaac `*_wrist_yaw_link` pose in the pelvis frame against
  `pretraining.kinematics` FK on the same frames; (b) `goal_coverage`: share of main's arm-target table poses
  (`ArmTargetsCommandCfg.table_file`, `locomanip_marl_env_cfg.py:178-194`) with a clip wrist pose within 0.1 m and
  30°; (c) the rotation of main's goal frame against the stage-1 tool frame (identity expected: `GRASP_OFFSET` is a
  translation). Aggregates only: no clip names in the output.
- [ ] **Step 1:** `@sim test_wrist_frames_on_standin_clip` (FK errors < 1 mm / 0.5°; frame rotation identity;
  output json has no key or value containing the clip file stem). **Step 2-4:** fail, implement, pass. **Step 5:**
  commit.

### Task 7: PULSE task on main's plant (piece 3, arm B and C's eval env)

**Files:**
- Modify: `locomanip_marl/marl_pulse_env_cfg.py` (whole file: drop `H1_2_PULSE_CFG`, `_pd`, `_implicit`,
  `PULSE_IMPLICIT_GROUPS`, `PULSE_DECIMATION`, the `joint_ranges` event, `standing_height`, the IMU vibration
  rescale; base class `MarlTorsoEnvCfg`), `mdp/pulse_actions.py` (`_PulseShared.__init__` :50),
  `pretraining/pretraining/marl/pulse.py` (:24, :66-67), `agents/__init__.py` (:13), `pretraining/tests/test_pulse.py`,
  `pretraining/pretraining/scripts/pulse_baseline.py` (reads the plant too)

**Interfaces:**
- Consumes: `get_plant`, `run.json["plant"]`.
- Produces: `PulseDecoders(actors, plant: Plant, limits)` (default pose, scale and target limits from the plant);
  `PULSE_STAGE1_RUN = "data/pretraining/runs/main_s1_finetune"`; `_PulseShared` refuses a stage-1 run whose plant's
  `step_dt` or default pose differ from the env's (`ValueError` naming run, plant and env decimation);
  `skrl_mappo_pulse_cfg()` returns the MAPPO yaml with only the PULSE std/init settings left to the env cfg (discount
  and lambda main's: 100 Hz again). Latent, prior, `PULSE_ZEROED_TERMS`, `PULSE_REMOVED_TERMS`, init scale and std
  bounds unchanged. The arms' latent drives the torso (decoder's upper body), so the Torso task's `torso_pos` term,
  its obs and its rate reward are removed with the other joint-space terms.
- [ ] **Step 1:** update `test_pulse_agent_cfg_is_the_mappo_yaml_at_its_50_hz_values` →
  `test_pulse_agent_cfg_is_the_mappo_yaml` (discount, lambda equal to the yaml's); new
  `test_pulse_refuses_a_run_of_another_plant` (stand-in run dir with `"plant": "pretrain"`); `@sim
  test_pulse_env_is_main_plant` (decimation 5, actuators equal `H1_2_MAGPIE_CFG`'s, `init_state` main's, randomisation
  event names equal `MarlTorsoEnvCfg`'s). Existing `test_existing_tasks_cfgs_keep_...` stays.
- [ ] **Step 2-4:** fail, implement, pass (sim tests with a stand-in stage-1 run: `tests/standins.py` actors at
  plant main).
- [ ] **Step 5:** cfg-dump compare → exit 0. Commit.

### Task 8: Main's command law and the segment process (piece 4, torch only)

**Files:**
- Create: `pretraining/pretraining/marl/command_law.py`, `pretraining/tests/test_command_law.py`

**Interfaces:**
- Produces: `VelocityLaw.from_cfg(cfg: ModalVelocityCommandCfg)` with `sample(n, gen) -> Tensor[n, 3]` (main's
  `MarlCommandsCfg.base_velocity`, `locomanip_marl_env_cfg.py:159-176`: ranges x ±0.55, y ±0.54, wz ±0.97; 25%
  standing; of the moving, 20% pure turn, 20% pure lateral); `ArmGoalLaw.from_table(path, standing_height)` with
  `sample(n, gen) -> Tensor[n, 2, 7]` (pelvis frame, w-first, qw >= 0); `CommandProcess(dt, mean_hold_s=6.0,
  clip_prob=0.25)` with `start(recorded: dict) -> state` (an imagined rollout starts from the command the env held at
  that buffered state) and `step(state, gen) -> state` (each segment ends with probability dt / 6 s: main's
  `nav_time_range` 4-8 s mean; a new segment is a clip segment with probability 0.25, else a law draw).
- [ ] **Step 1:** `test_velocity_law_shares` (1e5 draws: standing 0.25 ± 0.01, turn 0.15 ± 0.01, lateral 0.15 ±
  0.01, all within ranges); `test_arm_law_draws_table_rows`; `test_segment_length_mean` (mean 6.0 s ± 0.2 at dt 0.01);
  `test_start_holds_recorded_command`.
- [ ] **Step 2-4:** fail, implement, pass. **Step 5:** commit.

### Task 9: Main's rewards and actor observation on world-model state (piece 4)

**Files:**
- Create: `pretraining/pretraining/marl/main_rewards.py`, `pretraining/pretraining/marl/main_obs.py`,
  `pretraining/tests/test_main_rewards_obs.py`

**Interfaces:**
- Consumes: the world model's gencoord state (`marl/representation.py`), `TRACKED_BODIES` (`constants.py:149`).
- Produces: `MainRewardCfg.from_env_cfg(env_cfg) -> MainRewardCfg` (weights and kernel stds read from the Pulse env's
  resolved `rewards.legs.track_lin_vel_xy`, `track_ang_vel_z`, `rewards.arms.*_ee_*` and `goal_reached`, at the
  kernel stage passed in; GOLEM e-stop penalties from `pretraining.tasks` estop margins as stage 1's);
  `main_rewards(state, next_state, commands, cfg) -> dict[str, Tensor]` per agent, differentiable, wrist terms on
  `*_wrist_yaw_link` without `GRASP_OFFSET`. `MainActorObs(env_cfg, device)`: `reset(n, gen)` (encoder offsets,
  IMU bias drawn as main's), `__call__(state, commands, last_latent, t) -> dict[agent, Tensor]` building the legs and
  arms actor vectors term for term in `MarlObservationsCfg` order (:239-275): gyro and gravity of `torso_link` from
  the predicted body state through main's gyro low-pass (`imu_filter.py`, state carried), lag and vibration left out;
  commands; joints with offsets and main's noise; gait sin/cos from a clock advanced by dt; the latent as the
  applied action; wrist poses and errors in the pelvis frame.
- [ ] **Step 1:** `@sim test_main_obs_matches_env_obs_without_noise` (Pulse env, 4 envs, corruption, IMU bias, lag and
  vibration off, offsets zero: after 20 steps `MainActorObs` from the env's true state equals the env's actor obs per
  term within 1e-4; widths equal `group_obs_dim`); `@sim test_main_rewards_match_env_terms` (same state: each kernel
  equals the env's reward term value / (weight·dt) within 1e-5); `test_main_rewards_are_differentiable`
  (grad of the sum w.r.t. state is finite and non-zero).
- [ ] **Step 2-4:** fail, implement, pass. **Step 5:** commit.

### Task 10: Recurrent command encoder and `CommandRunner` (piece 4)

**Files:**
- Create: `pretraining/pretraining/marl/command_actor.py`, `marl/marl_source.py`, `marl/command_runner.py`,
  `pretraining/tests/test_command_runner.py`
- Reference (read-only): `data/ref/h12_train/src/h12_train/marl/runner.py:639-869`, `marl/actors.py:93`,
  `marl/goals.py:267`, `marl/config.py:122-205`. Ported files carry the h12_train `b175f1c` header; NOTICE updated.

**Interfaces:**
- Produces: `RecurrentCommandActor(obs_dim, command_dim, latent_dim=32, lstm=64, hidden=(512, 256))`,
  `forward(obs, command, hc) -> (z = tanh(...), hc)` (an LSTM of 64 as main's actor, `agents/skrl_mappo_cfg.yaml:19`);
  `MarlEnvSource(env)`: the Pulse env (Task 7, `pulse_prior=""`, the policy's output `u = atanh(clamp(z, ±tanh 5))`)
  exposing `reset()`, `step(latents) -> (state, actor_obs, commands, dones)` with the gencoord state read from
  `scene["robot"].data`; `CommandRunner(source, stage1_run, cfg)`: loads and freezes stage-1 decoders, encoders,
  normalisers and world model; refuses a plant mismatch; collects sim data with the encoders acting; trains the
  world model on it (as stage 1) and the encoders by back-propagating `main_rewards` plus e-stop penalties through
  `horizon`-step imagined rollouts started from buffered states; clip segments (Task 8) add the latent-BC term
  `0.1 * |z - z_stage1|²` against the stage-1 encoder on the clip's goal; `steps_per_env` counts sim steps.
- [ ] **Step 1:** `test_recurrent_actor_shapes_and_tanh`; `test_command_runner_refuses_plant_mismatch`;
  `@sim test_command_runner_two_iterations` (8 envs, stand-in stage-1 run at plant main: 2 iterations, the encoder's
  parameters change, decoders' do not, `steps_per_env == 2 * rollout`).
- [ ] **Step 2-4:** fail, implement, pass. **Step 5:** commit.

### Task 11: `train_commands` CLI (arm C)

**Files:**
- Create: `pretraining/pretraining/scripts/train_commands.py`, `pretraining/configs/commands.yaml`; modify
  `pretraining/README.md` (a "Stage 2 through the world model" section)
- Test: `pretraining/tests/test_command_runner.py`

**Interfaces:**
- Produces: `python -m pretraining.scripts.train_commands --stage1 <run> --run-name <name> [--num-envs N]
  [--max-steps 40032] [--max-seconds S] [--seed N]` and `--resume <run dir>`; runs in `data/pretraining/runs/`;
  `run.json` (stage1, plant, task, cfg, seed, env count actually used); `commands.pt` (both encoders and the
  frozen stage-1 paths) at every checkpoint, which Task 12 loads. `commands.yaml`: `wm.horizon 16`, `policy.horizon
  64`, `num_envs 4096`, `clip_command_prob 0.25`, `prior_weight 0.1`.
- [ ] **Step 1:** `test_train_commands_plan_and_resume` (sim-free `plan()`: existing dir refused, resume reads
  run.json). **Step 2-4:** fail, implement, pass. **Step 5:** commit.

### Task 12: Latent policies in the Isaac scorecards and video (piece 5)

**Files:**
- Create: `evals/isaac/latent.py`, `pretraining/tests/test_latent_evals.py`
- Modify: `evals/isaac/legs/get_scorecard.py` (`act` :107, `main` :164), `evals/isaac/arms/get_scorecard.py`,
  `evals/isaac/get_video.py`, `evals/isaac/env.py`, `evals/README.md`

**Interfaces:**
- Produces: `LatentPolicy.from_run(run_dir) -> LatentPolicy` — a skrl run whose `params/task.txt` is
  `LocoManip-Marl-Pulse-v0` (B: the MarlRunner as now) or a `train_commands` run (C: `RecurrentCommandActor`s, env
  `LocoManip-Marl-Pulse-v0` with `pulse_prior=""` and the run's stage 1); `act(obs: dict) -> dict[agent, Tensor]`,
  `reset(env_ids)`. Scorecards and `get_video` take `--latent_run <dir>` in place of `--checkpoint`; output under
  `logs/evals/<run>/` as now. A, B and C all run on the same conditions (the Torso/Pulse cfgs share every event).
- [ ] **Step 1:** `@sim test_scorecard_runs_a_c_policy_from_a_standin_run` (2 envs, 1 segment: json written with the
  same keys as a skrl run's). **Step 2-4:** fail, implement, pass; an existing skrl scorecard on a Direct
  checkpoint gives the same json as before (bit-identical with the same seed). **Step 5:** commit.

### Task 13: Latent policies in the MuJoCo scorecards (piece 5)

**Files:**
- Create: `pretraining/pretraining/scripts/export_decoders.py`, `evals/mujoco/latent.py`
- Modify: `evals/mujoco/policy.py` (:54, :58), `evals/mujoco/{legs,arms}/get_scorecard.py`, `evals/mujoco/get_video.py`

**Interfaces:**
- Produces: `export_decoders.py <stage1 run> --out <dir>`: `legs_decoder.pt`, `upper_decoder.pt` (TorchScript),
  `prior.pt` scripted if present, `plant.json` (default pose, scales, limits, joint order). `evals/mujoco/latent.py`
  `LatentMujocoPolicy(export_dir, encoder)` builds the 87-wide proprio frame from the MuJoCo state (as
  `pulse_actions.py:85` does in Isaac) and returns joint targets for the existing runner. `--latent_run` as Task 12.
- [ ] **Step 1:** test in the mujoco python: decoder in TorchScript equals the eager decoder on 100 random inputs
  (1e-5); the proprio frame from a MuJoCo state at the default pose equals the Isaac frame recorded by a sim test into
  a fixture (`pretraining/tests/fixtures/`, numbers only). **Step 2-4:** fail, implement, pass. **Step 5:** commit.

---

## Runs (controller only; one GPU process at a time; each stated to Max before it starts)

Run names: stage 1 `main_s1`, `main_s1_finetune`; arms `FairA_s42`, `FairB_s42`, `FairC_s42`.

### Task 14: Stage-1 timing and memory smoke on main's plant

- [ ] `timeout -s KILL 1800 $PY -m pretraining.scripts.train --config pretraining/configs/main_plant.yaml --plant main
  --run-name main_s1_smoke --num-envs 2048 --max-seconds 900` (2,048 envs: stage 1's limit at 50 Hz; 100 Hz with
  horizon 16/64 may need less). Record it/s, peak `nvidia-smi` memory, WM loss trend. If OOM: 1,536, then 1,024.
- [ ] Decide `max-seconds` and env count for Task 16 from it; report the projected iterations in 3 h.

### Task 15: Clip-replay wrist check

- [ ] `timeout -s KILL 1800 $PY -m pretraining.scripts.wrist_frames --motion-dir data/motions/h12_amass --clips 40
  --out logs/evals/wrist_frames`. Pass: FK p95 < 5 mm / 2°, frame rotation identity. Report `goal_coverage`; below
  50% means stage 1 rarely saw main's goal orientations — tell Max before Task 16 (it bounds what B and C can reach).

### Task 16: Stage-1 base (3 h cap)

- [ ] `timeout -s KILL 11400 $PY -m pretraining.scripts.train --config pretraining/configs/main_plant.yaml --plant main
  --run-name main_s1 --num-envs <Task 14> --max-seconds 10800`.

### Task 17: Fine-tune (1.5 h cap)

- [ ] `timeout -s KILL 6000 $PY -m pretraining.scripts.train --config pretraining/configs/main_plant_finetune.yaml
  --run-name main_s1_finetune --fine-tune-from data/pretraining/runs/main_s1 --max-seconds 5400`.

### Task 18: Cross-check gate

- [ ] `timeout -s KILL 2400 $PY -m pretraining.scripts.crosscheck --run data/pretraining/runs/main_s1_finetune
  --motion-dir data/motions/h12_amass --clips 40`. Gate: at most 3 of 40 fall ("a few"). On fail: resume
  `main_s1` for 3 h (`--resume ... --max-seconds 10800`), redo 17-18, and report.
- [ ] Zero-policy check: `pulse_baseline.py` on the Task 7 env for 10 s; report the fall rate against the first run's 23%.

### Task 19: Prior fit (0.5 h)

- [ ] `timeout -s KILL 2400 $PY -m pretraining.scripts.fit_prior data/pretraining/runs/main_s1_finetune`.

### Task 20: Arm A, 40k steps

- [ ] `timeout -s KILL 14400 $PY scripts/skrl/train.py --headless --task LocoManip-Marl-Torso-v0 --max_iterations 834
  --name FairA_s42 --seed 42`.

### Task 21: Arm B, 40k steps

- [ ] `timeout -s KILL 14400 $PY scripts/skrl/train.py --headless --task LocoManip-Marl-Pulse-v0 --max_iterations 834
  --name FairB_s42 --seed 42`.

### Task 22: Arm C, 40k steps

- [ ] Speed first: `--max-seconds 600` on a throwaway name; if 40,032 steps project past 6 h, tell Max before the run.
- [ ] `timeout -s KILL 21600 $PY -m pretraining.scripts.train_commands --stage1 data/pretraining/runs/main_s1_finetune
  --run-name FairC_s42 --max-steps 40032 --seed 42`.

### Task 23: Scorecard table

- [ ] Isaac legs and arms scorecards and MuJoCo legs and arms scorecards for A (`--checkpoint` last), B and C
  (`--latent_run`), same seed and defaults; each under `timeout -s KILL 3600`.
- [ ] One table (velocity error, wrist position and orientation error, falls, e-stop trips; Isaac and MuJoCo rows)
  to Max, with each run's std-at-bound note (Review Focus 5). The table is committed only if it lists no clip names.

---

## Calls made in this plan

- Torso action `default + 0.25 x a`, torso first in the arms' 15: GOLEM's `torso_action_scale` and order; cost if
  wrong: A's torso authority differs from the latent arms' (the decoders' torso scale is 0.1).
- Torso action rate charged at the arms' `action_rate` weight: no other cue; too low lets the torso jitter in A.
- Main plant's action scale by stage 1's formula `min(0.25, 0.25 effort/kp)` (shoulder pitch 0.04 rad): the spec's
  "`ACTION_SCALE` x output" keeps the convention; if too small, stage 1 cannot reach clip poses (seen in Task 14).
- Main plant target limits = pretraining's `TARGET_LIMITS` (∩ GOLEM in the Pulse env): unchanged from the first run.
- Stage-1 rate changes beyond the spec's: policy horizon 32→64, return horizon 20→40, gamma 0.99→0.995, lam
  0.95→0.975 (same seconds); cost: a different effective horizon if the spec meant only the WM's.
- Stage 1 at 2,048 envs (not 4,096): the card's limit at 50 Hz; Task 14 may lower it.
- Fine-tune randomisation = main's events + 0-10 substep target delay + joint noise; the IMU model is not added
  (stage 1's proprio reads the IMU directly); cost: B and C see a sensing gap the decoders never trained on.
- B's discount back to main's yaml (100 Hz); the PULSE std bounds kept (Review Focus 5).
- Pulse env built on the Torso cfg, so A/B/C share every event and sensor term by construction.
- C's imagined commands start from the env's recorded command and are held/resampled per segment; the arm-goal
  trajectory within a 0.64 s rollout is not modelled; cost: C's arms see slightly staler goals in imagination.
- C's observation model leaves out the IMU lag and vibration (kept: low-pass, bias, noise): exact replication would
  port `lowstate_imu.py`'s whole model; cost: an imagination-vs-sim obs gap, measured by Task 9's test with them on.
- Clip segments only in imagination (the Pulse env has no clips); cost: no sim data under clip commands.
- C evaluated in the Pulse env with the prior off, emitting `atanh(z)`; "40k steps" for C = 40,032 sim steps per env.
- Stage-1 default run name `main_s1_finetune` hard-coded as `PULSE_STAGE1_RUN`.
