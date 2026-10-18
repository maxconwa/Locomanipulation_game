# Locomanipulation game

An Isaac Lab external project for the Unitree H1-2 with Magpie grippers.

**This branch (`marl-direct`):** one robot and two MAPPO agents. The legs track a velocity command. The arms each
track a wrist pose, and they crouch for low goals by squatting. The task targets deployment through GOLEM's safety
layer into RoboCasa.

- `source/locomanipulation_game/locomanipulation_game/tasks/locomanip_marl/`: the task, `LocoManip-Marl-Direct-v0`, and
  its one-agent whole-body baseline, `LocoManip-WholeBody-Direct-v0`.
  - `locomanip_marl_env_cfg.py`: everything configurable.
  - `base_cfg.py`: the scene, the legs' ALMI reward set, events and terminations the cfg extends.
  - `locomanip_marl_env.py`: a DirectMARLEnv that runs Isaac Lab's managers.
  - `mdp/commands.py`: navigation or arm goals, and the reach curriculum.
  - `mdp/legs_rewards.py`: the legs' gait clock and ALMI reward terms.
  - `odometry.py`: the pelvis-motion estimator.
  - `golem_safety.py`: GOLEM's e-stops.
- `scripts/skrl/`: training, playing, the Isaac evaluation and the GOLEM export.
- `agent_scripts/`: standalone diagnostics of a checkpoint's behaviour. `knee_bow_test.py` checks whether the knees
  collapse inward when the robot crouches.
- `mujoco_evals/`: the MuJoCo transfer. The env's policy loop ported to MuJoCo, the reach evaluation's MuJoCo trials,
  the Isaac-MuJoCo parity check, the Isaac side of GOLEM's MuJoCo walking test, and keyboard and mouse control of a
  policy (`human_control.py`).
- `paper/reach/`: the reach evaluation (Isaac and MuJoCo on one fixed goal grid, then the paper's tables, figures,
  numbers and result page); see its README.
- `logs/`: everything the code writes. Training runs go to `logs/skrl/locomanip_marl/`, the reach evaluation to
  `logs/reach/`. Tracked: the published runs, and the reach evaluation's goal grid, figures, LaTeX macros and page;
  its trials and summaries stay local.

## Setup

```bash
conda activate env_isaaclab
python -m pip install -e source/locomanipulation_game
```

## The MARL task

```bash
# train (default 8334 updates = 200k steps, ~5 h; the paper runs used --max_iterations 3800 = 91.2k steps);
# --checkpoint resumes, with the estimator and curriculum saved beside it
python scripts/skrl/train.py --headless [--max_iterations N] [--name <run>] [--checkpoint <run>/checkpoints/agent_<N>.pt]
python scripts/skrl/play.py --checkpoint <run>/checkpoints/agent_<N>.pt
# Isaac evaluation: walking and turning speeds, reach rate, end-effector error, squat depth -> <run>/eval_<N>/eval.md
python scripts/skrl/eval_final.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless

# deployment: networks + export.yaml, arm goals for the game commander, and parity with GOLEM's controller
python scripts/skrl/export_marl.py --checkpoint <agent.pt> --out <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_golem
python scripts/skrl/export_arm_goals.py --checkpoint <agent.pt> --levels 4:10,7:10,10:10 --out <same dir> --headless
python scripts/skrl/deploy_parity.py --checkpoint <agent.pt> --export <same dir> --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy --headless

```

**Not yet ported to the paper task.** `eval_final.py`, `export_arm_goals.py`, `deploy_parity.py` and
`mujoco_evals/eval_sim2sim_walk.py` still use names the task no longer has:
- the spread/drop curriculum (`spread_level`, `drop_level`, `goal_low`, `height_drop`, `settling`), now one depth
  `level`;
- the RoboCasa physics (`events.passive_damping`, `action_delay_substeps`).

They fail with an `AttributeError`. For an Isaac evaluation of a current checkpoint, use the reach evaluation
(`paper/reach/tools/eval_isaac.py`, below). `train.py`, `play.py` and `export_marl.py` do not use the removed names.

**Changes since the paper runs** (`paper00`, `paper05`, `paper1`, `paperwb`, `paperwb256`). None of them changes an
observation or an action, so those checkpoints still load (a saved depth level above 8 is clamped to 8):
- **`knee_valgus`** in the legs' reward charges the knees for falling inward of the hip-ankle line.
- **Tracking-kernel curriculum.** The fine wrist-position and the wrist-rotation kernels narrow from 0.10 m / 0.5 rad
  down to 0.012 m / 0.08 rad. The stage moves after every 20k ended arm goals: it narrows when half of them came
  within the current kernels, and widens when fewer than a quarter did. TensorBoard logs it under
  `Curriculum/arm_target_levels/kernel_stage`, `fine_pos_std` and `rot_std`.
- **Estimator trained on arm goals only** (`estimator.train_on_arm_goals_only`). The estimator fits the pelvis motion
  only during arm goals, the only time it moves the command, and never on walking. The policies train on the estimate,
  as the paper runs did (the drift gate opens it).
  - `kernels_gtodom_200k` instead trained on the true motion (`estimator.true_odometry_while_training`).
- **Targets clipped 0.02 rad inside GOLEM's edges** (`golem_safety.position_envelope`, `cfg.golem_estop`).
  - GOLEM's edges are its command clip, the URDF range shrunk by 0.001 rad. Every joint target, legs and arms, is
    clipped `GolemEstopCfg.position_margin` = 0.02 rad inside them: the URDF range shrunk by 0.021 rad (knee -0.099
    to 2.169, hip yaw ±0.409). h12_adaptive_policy deploys with a clip like this.
  - The clip is recorded in `params/env.yaml` (`actions.*.joint_limits`). The MuJoCo tools apply each run's own clip,
    and `export_marl.py` writes it to `export.yaml` as `target_limits`. GOLEM's deploy node must clip the same way.
- **E-stops are logged, not terminal.** The monitor checks every physics substep against GOLEM's real position
  thresholds, and its velocity and torque thresholds times 0.9. Trips are logged under `Safety/`.
  `golem_estop_ends_episode=True` makes them end the episode once the arm goals start; that run stalled.
  - The clip bounds targets, not measured joints. A robot's weight can still push a joint past its target into the
    e-stop, and `Safety/estop_position` shows how often. `estodom_clip_200k` rested hip yaw and ankle pitch on their
    stops in deep squats: about half of its arm-goal steps would have e-stopped.
- **GOLEM-limit penalties** (`mdp.golem_limits`). Each agent pays for its own measured joints within 0.03 rad of
  GOLEM's position e-stops (legs −40, arms −20 per rad), and for velocity or motor torque beyond 0.8 of GOLEM's
  thresholds (−10 per share of the threshold).
- **Arm speed and effort** are squared shares of GOLEM's velocity and torque e-stops (`speed` −2, `effort` −1). They
  replace the arms' `dof_vel` and `torques`, which cost almost nothing.
- **Goal heights uniform in [z_min, z_max]** (`ArmTargetsCommandCfg`).
  - z_max is fixed at 1.7 m above the ground. z_min starts at 1.0 m and falls 0.1 m per depth level, down to 0.2 m
    at level 8. Goals keep coming from the whole range.
  - Each wrist is a table pose within 0.15 m of the goal height. Below what the standing arms reach, both poses come
    from the table's lowest band and are lowered together.
  - The depth level is judged only on the low goals, those within 0.3 m of z_min. TensorBoard logs `z_min` in place
    of `z_max`.
  - The paper runs lowered table poses by a uniform offset instead, with 10 levels.
- **Odometry: height and tilt are absolute, and the estimator is fit against drift** (`estimator.*`).
  - `vertical_from_feet`: each step's vertical pelvis motion is the change in the pelvis's height above the lowest
    ankle along gravity, from forward kinematics, in place of the estimator's.
  - `tilt_from_gravity`: the estimated rotation is turned so that its roll and pitch follow gravity.
  - Only the horizontal motion and yaw are integrated from the estimate. On `golemlim120k` over a 16 s hold, the
    targets' vertical error stays under 0.1 cm in MuJoCo, against a 29 cm rise without these. In Isaac it stays at
    1.7 cm (p90) against 6.9 cm and growing. Both apply to any checkpoint: the Isaac env and `mujoco_sim.py` turn them
    on by default.
  - The fit adds, at weight 10 (`window_loss_weight`), the squared mean error over every 1 s window
    (`window_steps` 50) in the world frame. A steady bias, which is what drifts the arm goals, counts in full there,
    while the noise averages out. The buffer keeps the last 100 steps in order and fits every 50.
  - The estimator sees 25 frames (0.5 s, `ODOMETRY_HISTORY`), up from 4. Its file records its window, and an older
    file's 4 frames are restored when it is loaded. `export_marl.py` writes `odometry` (the window, and the feet and
    gravity rules) to `export.yaml`; GOLEM's deploy node must apply them.
  - TensorBoard: `Estimator/window_lin_vel_error`, `window_vertical_vel_error` and `window_ang_vel_error` (drift rates);
    `feet_vertical_error`, `learned_vertical_error` and `learned_vertical_bias`.
- **Knees charged from 0.08 rad** of their e-stops (`GOLEM_KNEE_POSITION_MARGIN`; every other joint 0.03), through
  `mdp.golem_limits`' `joint_margins`. `golemlim_zrange_200k` pressed the right knee past 2.19 rad in 1.2% of its
  arm-goal substeps, the most of any joint.
- **The legs learn more gently** (`skrl_mappo_cfg.yaml`): their KL-adaptive learning rate aims at a KL of 0.01 (the
  arms keep 0.02), between 1e-4 and 1e-3, and an update stops early once its KL passes 0.03. At about 187k steps
  `golemlim_zrange_200k`'s legs took one update that spiked their value loss and raised falls from 1% to 6-10%.
- **Falls cost 20, not 5** (`termination_penalty`). Both agents get it once, on the step the robot falls, after
  their rewards are floored at 0. Every run up to `golemlim_zrange_200k` used -5.
- **Actors and critics are [128, 64, 32]** (`skrl_mappo_cfg.yaml`), down from [256, 128, 64]. `almisize_200k` tried
  ALMI's 64-32 actors (critics unchanged): its arms tripped GOLEM's position e-stops 2.5x less, but its legs never
  learned to squat (pelvis 2.6 cm below standing at 200k, against 19 cm). `play.py`, `deploy_parity.py` and
  `eval_final.py` build their models from the yaml, so for an older checkpoint pass
  `'agent.models.policy.network.0.layers=[256,128,64]' 'agent.models.value.network.0.layers=[256,128,64]'`; the
  MuJoCo tools, the reach evaluation and `export_marl.py` read the sizes from the checkpoint.
- **Trained as ALMI-Open trains its H1-2 policies**, for policies that stay smooth on the robot (the earlier ones
  jittered on hardware):
  - **Explicit PD.** The legs', feet's, torso's and arms' actuators are `IdealPDActuatorCfg`
    (`explicit_pd`): the torque is computed every physics step and clipped at the effort limit, in place of PhysX's
    implicit drive. The locked grippers stay implicit. The torque bounds and GOLEM's torque checks read the
    configured gains (`default_joint_stiffness`), since explicit actuators zero the sim's. Physics runs at 0.005 s,
    4 substeps per policy step (`SIM_DT`, `DECIMATION`). almitrain_200k to lstm_run ran ALMI-Open's 500 Hz
    (0.002 s, 10 substeps) at about half the training speed; holding the default pose, the explicit PD jitters no
    more at 200 Hz than at 500.
  - **Action delay.** Each env's joint targets reach the robot 0-2 policy steps late, drawn at every reset
    (`action_delay_range`); the policies observe what they asked for.
  - **Fixed observation scales** (`observation_scales`): joint velocity x0.05, angular velocity x0.25, velocity
    commands x(2, 2, 0.25), the critic's linear velocity x2, linear acceleration x0.1, everything else x1. `train.py`
    freezes skrl's observation and state scalers at mean 0 and std 1/scale, so a checkpoint's scaler has the usual
    format and the export, the MuJoCo tools and GOLEM's deploy node need no change.
  - **Wider randomization** (`base_cfg.EventCfg`): friction 0.1-1.25 (was 0.6-1.2), pelvis, leg and torso link masses
    x0.9-1.1, torso mass -3..+5 kg (was -2..+4), the pelvis's centre of mass shifted up to 10/10/20 cm, pushes of up
    to 1 m/s every 10 s (was 0.4 m/s every 8-12 s).
  - **Rough terrain** (`TERRAINS_ROUGH_CFG`): uniform bumps of +-0.5 cm on row 0 up to +-5.5 cm on row 9, with
    envs starting on rows 0-5. An env goes down a row after a fall and up one after a 20 s episode whose mean
    navigation velocity error was under 0.4 m/s (`mdp.terrain_levels_tracking`). almitrain_lstm_200k used ALMI-Open's
    `terrain_levels_vel`, which promotes by distance from the start: our commands turn and stop for arm goals, so it
    sank every env to row 0 as the tracking improved. All earlier runs trained on flat planes. The reach
    evaluation and the Isaac tools build the env from this config, so they now run on it too.
  - **LSTM actors** (`locomanip_marl/recurrent.py`, `models.policy.rnn` in the yaml): as ALMI-Open's
    RNNActorMLPCritic, each actor runs its observations through one LSTM layer of 64; unlike it, a [256, 128, 64]
    MLP then sees the LSTM's output concatenated with the raw observations (`rnn.skip_observations`). The critics
    stay MLPs. skrl has no recurrent MAPPO, so `RecurrentMAPPO` adds skrl PPO_RNN's state handling to
    MAPPO (each env's 24-step rollout is one training sequence from its stored first state, zeroed at episode ends),
    and `MarlRunner` builds it; train, play, eval_final, deploy_parity and eval_sim2sim_walk use it. Earlier
    MLP checkpoints load with `'~agent.models.policy.rnn'`, almitrain_lstm_200k's (no skip, [128, 64, 32]) with
    `agent.models.policy.rnn.skip_observations=False 'agent.models.policy.network.0.layers=[128,64,32]'`.
    `export_marl.py` exports an LSTM actor as `forward(obs, h, c) -> (action, h, c)`, which GOLEM's `lmg_deploy`
    runs (`RecurrentTorchScriptNetwork`); `mujoco_evals/policy.py` and per-env state resets in the evaluation scripts
    are not done yet.
  - **Pushes on a curriculum** (`mdp.push_levels`, `mdp.push_by_level`; not ALMI-Open's, whose pushes are fixed):
    each env has a push level 0-10 that starts at 0, goes down one after a fall and up one after a 20 s episode that
    tracked its velocity commands (the terrain rows' rule above), and is pushed every 10 s at up to level/10 x 1 m/s.
    Survival alone raised it at first, and lstm_run's robots climbed to 0.86 m/s by standing still. The Isaac
    evaluations turn pushes off as before.
  - **Kneeling is a fall** (`mdp.fell_or_knelt`): an episode also ends as a fall once the pelvis is under 0.55 m
    above the ground (`MIN_PELVIS_HEIGHT`; the deepest crouches lower it about 25 cm). Under the harder pushes and
    terrain, with only the tilt check, `almitrain_gtodom_200k` learned to kneel upright, its pelvis 60 cm down.
  - **Foot contact force once per step** (`mdp.feet_contact_forces`): the force over 1400 N is read at the policy
    step's last physics substep, as ALMI-Open reads it, not the peak over the substeps. At 500 Hz the peak caught
    every landing's spike: the penalty was about 70x feetodom_knee_200k's in lstm_run, whose legs stopped stepping.

### Explicit500 (10-15): what brought the crouch back

lstm_run2, trained with everything above, walked tall and left low goals to the arms. Short ablations (40k steps,
`logs/skrl/ablation_*`) rebuilt feetodom_knee_200k on today's code and added each change back one at a time. Its
crouch needs the foot-contact penalty as the **peak over the substeps**, and dies with **fixed observation scales**
(the wrist targets, gravity and joint angles reach the network 4-30x quieter than feetodom's running statistics),
**explicit PD at 200 Hz** (5 ms torque holds), or a **0-40 ms action delay**; **ALMI-Open's randomization** and
**rough terrain from the start** weaken it; the **LSTM** and the **push curriculum** keep it (the LSTM deepens it).
Explicit PD at ALMI-Open's 500 Hz keeps it, and a 0-20 ms delay in 2 ms substeps keeps it with the best turning.

`Explicit500` (`logs/skrl/locomanip_marl/2026-10-15_12-17-16_Explicit500`, agent_196800, exported to GOLEM's
`lmg_deploy/policies/explicit500`) is these defaults with:

```bash
python scripts/skrl/train.py --headless --name Explicit500 \
  env.commands.arm_targets.table_file=$PWD/logs/skrl/locomanip_marl/golemlim120k/arm_target_tables_v3.pt \
  env.observation_scales=null env.explicit_pd=True env.sim.dt=0.002 env.decimation=10 \
  env.action_delay_range=[0,10] env.action_delay_substeps=True \
  env.rewards.legs.feet_contact_forces.params.peak_over_substeps=True env.rewards.legs.feet_contact_forces.params.window=3 \
  env.terminations.fell.params.min_pelvis_height=-1.0 \
  env.events.physics_material.params.static_friction_range=[0.6,1.2] env.events.physics_material.params.dynamic_friction_range=[0.4,0.9] \
  env.events.physics_material.params.make_consistent=False env.events.scale_link_masses=null env.events.shift_pelvis_com=null \
  env.events.add_torso_mass.params.mass_distribution_params=[-2.0,4.0] \
  env.scene.terrain.max_init_terrain_level=0 env.scene.terrain.terrain_generator.sub_terrains.rough.height_at_0=0.0 \
  env.curriculum.terrain_levels.params.gated=True
```

- **500 Hz physics** follows from `env.sim.dt` / `env.decimation` alone: `cfg.sync_physics_rate()` sets the sensor
  periods and the contact history (train.py and the env call it after the overrides).
- **Contact peak over 6 ms means** (`window=3`): at 500 Hz a landing's impulse reads as a higher force; the peak of
  3-substep means matches the 200 Hz penalty the crouch was learned with (15.2 vs 15.0 N per step, same policy).
- **Delay in physics substeps** (`action_delay_substeps`): 0-10 substeps, 0-20 ms in 2 ms steps.
- **Rough terrain behind a gate** (`mdp.gated_terrain_levels`, `gated=True`): every robot starts on the flat row 0; rows
  open one per 8k steps once falls are under 2%, over 70% of episodes track their commands, the arm depth level is
  at least 4 and the pelvis drops at least 0.75x as far as the targets (it opened at 46k and reached row 9, +-5 cm,
  at 112k).
- **Tilt-only fall check** and feetodom's randomization, as feetodom_knee_200k.
- Navigation logs `Metrics/base_velocity/nav_cos_vel_xy`, the cosine between the commanded and the true xy velocity.

At 200k: pelvis 17 cm down for goals 23 cm low, arm depth 7.4, best error 4.5 cm, falls 1.1%, velocity error
0.25 m/s and 0.83 rad/s (a right-turn bias of about 0.07 rad/s), and it steps at the gait clock when turning in place.
In MuJoCo it reaches the low goals within 2 cm on true odometry, 5-12 cm on the learned estimator, whose drift
gate stayed closed in training.

### The IMU is the head's Livox (10-16, after Explicit500)

The policies' and the estimator's IMU (`scene.imu`) is the MID-360's IMU chip on the head: `livox_link`, the upright
lidar frame (pitched about 14 deg), offset `base_cfg.LIVOX_IMU_OFFSET` (0.011, -0.02329, 0.04412) m, as GOLEM's
`livox_imu_upright` publishes `/livox/imu`. At the default pose it reads gravity as (0.20, -0.04, -0.98) and 9.81 m/s^2.
Every run up to Explicit500 read the torso's `imu_link`; play, evaluate or export one with
`'env.scene.imu.prim_path={ENV_REGEX_NS}/Robot/imu_link' 'env.scene.imu.offset.pos=[0.0,0.0,0.0]'`. `export_marl.py`
writes the run's IMU into export.yaml (`imu.frame`, `imu.offset`), and GOLEM's `lmg_deploy` then reads `/livox/imu`
with FAST-LIO's attitude (`livox_attitude.py`) instead of `LowState`. The MuJoCo tools (`mujoco_evals/mujoco_sim.py`)
still read the torso site: they evaluate torso-IMU runs only for now.

## Knee-bow test (`agent_scripts/`)

```bash
# each env stands, then crouches for an arm goal; front-view geometry of hips, knees and ankles every step
python agent_scripts/knee_bow_test.py --checkpoint <agent.pt> --headless [--goals rest|grid] [--video]
```

- `--goals rest` (default) lowers both wrists' rest pose by 0 to 1 m across 32 envs: a pure squat.
- `--goals grid --num_envs 88` uses the reach evaluation's goals.
- `--video` adds a front view of one env, standing and then crouched.

Outputs go to `logs/knee_bow/<run>_<N>_<goals>/`:
- `summary.txt`: per env and by pelvis drop.
- `knee_bow.png`, `knee_bow.npz`.
- With `--video`: `knee_bow_front.{mp4,png}`.

How to read the results:
- `valgus > 0` means a knee sits inside its hip-ankle line, and `knee_gap < 0` means the knees are narrower than the
  ankles. At the default pose both are 0.
- `vCost` is the `knee_valgus` reward term per second.

## MuJoCo transfer (`mujoco_evals/`)

```bash
# the reach goal grid in MuJoCo, RoboCasa's joint dynamics or (--physics isaac) the training plant's; any python
# with mujoco, torch and pyyaml
python mujoco_evals/eval_mujoco.py --checkpoint <agent.pt> --label <label> [--physics isaac] [--limit N]
# the same state in both simulators, observations term by term and the arm IK targets (Isaac side first)
python mujoco_evals/parity_check.py --side isaac && python mujoco_evals/parity_check.py --side mujoco
# GOLEM's walking sim-to-sim test, Isaac side (scored by GOLEM's tests/locomanipulation_game/sim2sim_walk.py --score)
python mujoco_evals/eval_sim2sim_walk.py --checkpoint <agent.pt> --out <dir> --headless
```

```bash
# drive a two-agent checkpoint by hand in the MuJoCo viewer (default: kernels_gtodom_200k's agent_196800); needs a display
python mujoco_evals/human_control.py [--checkpoint <agent.pt>] [--physics robocasa|isaac] [--odometry learned|true]
```

`human_control.py`:
- **Walking:** the arrow keys and Page Up/Down set the velocity command, and End stops.
- **Reaching:** Enter switches to reaching. The same keys then move the wrist targets 2 cm at a time, or you can
  drag them with the mouse (double-click a target, then Ctrl + right-drag).
- **Overlay:** the command, the gait-cycle mean pelvis velocity, the wrist errors and the odometry drift.
- **Keys:** the full list is in the script's docstring.
- **Quitting:** close the window. Ctrl+C in the terminal ends with an X error while the viewer shuts down.

`eval_mujoco.py` writes to `logs/reach/results/<label>/mujoco*/`, beside the Isaac trials of `paper/reach`.
`eval_sim2sim_walk.py` and the parity check's Isaac side need the Isaac Lab python.

Runs are written to `logs/skrl/locomanip_marl/<time>_<name>/`. Each run directory holds:
- the skrl checkpoints;
- `estimator/` (the pelvis estimator and curriculum state);
- `arm_target_tables_v3.pt`: built on a run's first start, which takes a few minutes, then reused by every script
  pointed at that run.

## Reach evaluation

```bash
make -C paper/reach all     # every run in paper/reach/conditions.yaml without current results, then the paper outputs
```

The outputs (trials, summaries, figures, LaTeX macros and the result page) go to `logs/reach/`.
