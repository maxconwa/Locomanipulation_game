# pretraining: stage-1 motion tracking for the H1-2 in Isaac Lab

This package trains two policies together to make the Unitree H1-2 follow retargeted AMASS motion clips in Isaac
Lab: a legs agent (12 joints) and an upper-body agent (torso and 14 arm joints). It is the Isaac Lab version of the
stage-1 pretraining in `h12_train` (branch `supermimic`), and the base that later RL fine-tuning in Isaac Lab starts
from.

The method is SuperMimic's. A world model (a network that predicts the next simulator state from the current state
and the actions) is learned from simulator rollouts, and the policies are trained by gradients through that world
model over imagined rollouts, not by PPO. Compared with `h12_train`:

- the method, the settings and the policy contract (below) are the same, so `actors.pt` from either simulator loads
  in either codebase;
- the simulator is Isaac Lab / PhysX, not mjlab / MuJoCo-Warp;
- SuperMimic's learner, world model and replay buffer are self-contained copies in `pretraining/supermimic/`
  (`h12_train` imports them from the `supermimic` submodule, which needs Python 3.13; Isaac Lab here is Python
  3.11). Nothing imports `supermimic` or `mjlab`. `NOTICE` lists every ported file with its upstream file and commit;
- only stage 1 (clip tracking) is here.

## Setup

```bash
conda activate env_isaaclab
git submodule update --init --recursive   # third_party/CL_Assets: the robot's USD and URDF
pip install -e source/locomanipulation_game   # main's robot asset, which this package loads
pip install -e pretraining
```

The gym ids `LocoManip-Pretrain-Track-v0` and `LocoManip-Pretrain-Track-Play-v0` are registered when
`pretraining.tasks` is imported (the scripts do it). Run everything from the repository root.

Tests use procedurally generated stand-in clips, no mocap. Tests marked `sim` start one headless Isaac Sim app per
session; the others never start it. Use the env's python directly (`conda run` loses the output when the app shuts
down), and put every simulator command under `timeout -s KILL`: Isaac ignores SIGTERM, and a process that hit a CUDA
device-side assert spins at shutdown instead of exiting.

```bash
PY=/home/max/miniconda3/envs/env_isaaclab/bin/python
$PY -m pytest pretraining/tests -m "not sim" -q                      # no simulator, seconds
timeout -s KILL 1500 $PY -m pytest pretraining/tests -q              # everything, about 7 minutes
```

## Data

Nothing derived from AMASS or SMPL-X is in the repository: not the clips, not the runs, not the policies. AMASS and
SMPL-X are non-commercial and not redistributable, and `tools/amass/README.md` (branch `supermimic`) extends those
terms to the retargeted clips and to policies trained on them. All of it lives under `data/`, which git ignores.

**Clips.** A directory of `.npz` files, one per clip, at 50 Hz, as written on branch `supermimic` by `tools/amass`
and `h12_train.scripts.clip_to_motion`. Default location `data/motions/h12_amass/` (13,578 clips, 1.3 GB on the
development machine).

| key | shape | meaning |
|---|---|---|
| `fps` | (1,) | 50.0 |
| `joint_names` | (27,) str | Unitree motor order (checked at load) |
| `joint_pos`, `joint_vel` | (T, 27) float32 | rad, rad/s |
| `root_pos_w` | (T, 3) | pelvis position, m, world |
| `root_quat_w` | (T, 4) | pelvis orientation, quaternion wxyz, world |
| `root_lin_vel_w`, `root_ang_vel_w` | (T, 3) | pelvis velocities, m/s and rad/s, world |

Body poses of the reference are computed from these by forward kinematics (see "Known differences").

**Runs** go to `$PRETRAINING_RUNS_DIR/<run name>/`, default `data/pretraining/runs/`. The two MuJoCo-trained
policies from `h12_train` are expected at `data/runs/amass_s1` and `data/runs/amass_s1_golem_aware` for the
cross-check.

## The policy contract

Unchanged from `h12_train`:

| item | value |
|---|---|
| control | 50 Hz: physics step 0.005 s, decimation 4 |
| joint order | Unitree motor order, 27 joints: 12 legs, torso, 14 arms |
| agents | legs: joints 0-11 (12 actions); upper body: joints 12-26 (15 actions) |
| action | target = default pose + action scale x action, clipped to GOLEM's target limits; PD at `h12_train`'s kp, kd and effort limits (`constants.py`: not `main`'s sport-mode gains, stance or scales) |
| actor observation | one 87-wide frame (history length 1): projected gravity (3) and gyro (3) of the torso IMU, joint position minus default (27), joint velocity (27), last action of both agents (27); noise half-widths 0.05, 0.2 rad/s, 0.01 rad, 0.5 rad/s |
| actor goal | the reference window of SuperMimic's gencoord representation, split between the agents as `h12_train` does |
| actor network | SuperMimic's `MultiEncoderAE`: encoder(observation, goal) -> 32-wide tanh latent -> decoder(observation, latent); hidden sizes 2048, 1024, 512, 256 |
| critics, world model | critic hidden sizes 256, 128; world-model dynamics hidden sizes 2048, 2048, 2048 |
| tracked bodies | 14 (`constants.TRACKED_BODIES`); anchor body `pelvis` |

A run directory, the same layout as `h12_train`'s:

```
<run>/
  run.json          every setting, the config file's content, the env count, the motion files, the seed;
                    fine_tuned_from and resumed when they apply
  progress.csv      one row per log interval: losses, Metrics/motion/error_*, Safety/*, wall time, peak GPU memory
  world_model.pt    the world model
  actors.pt         the two actors (legs, upper body)
  critics.pt        the two critics
  checkpoint.pt     the newest full state (optimisers, normalisers, reset sampler): what --resume reads
  checkpoints/      it_<iteration>.pt, the last keep_checkpoints periodic ones (40 in the configs)
```

## Commands

```bash
PY=/home/max/miniconda3/envs/env_isaaclab/bin/python

# base run (configs/base.yaml: amass_s1's settings), with the env count and wall time of the long run
timeout -s KILL 25200 $PY -m pretraining.scripts.train --config pretraining/configs/base.yaml --run-name isaac_s1 \
    --num-envs 2048 --max-seconds 23000 [--motion-dir data/motions/h12_amass] [--seed N]

# GOLEM-aware fine-tune of it (configs/golem_aware.yaml), into a new run directory
timeout -s KILL 7200 $PY -m pretraining.scripts.train --config pretraining/configs/golem_aware.yaml \
    --run-name isaac_s1_golem_aware --fine-tune-from data/pretraining/runs/isaac_s1 --num-envs 2048 --max-seconds 5200

# continue a run that stopped (only --max-iterations, --max-seconds and --device are accepted beside it);
# without --max-seconds it runs on to max_iterations (320,000 in base.yaml, 13 h or more), whatever its budget was
timeout -s KILL 25200 $PY -m pretraining.scripts.train --resume data/pretraining/runs/isaac_s1 [--max-iterations N] [--max-seconds S]

# cross-check: a run's actors (either simulator's) on 40 clips -> crosscheck.csv, clips.txt, summary.txt
timeout -s KILL 1800 $PY -m pretraining.scripts.crosscheck --run data/runs/amass_s1 --motion-dir data/motions/h12_amass \
    --clips 40 --out data/pretraining/crosscheck/amass_s1
#   the same clips for another run: --clips-file data/pretraining/crosscheck/amass_s1/clips.txt

# play named clips (file stems) and record env 0 -> <out>/play.csv, <out>/videos/*.mp4
timeout -s KILL 1800 $PY -m pretraining.scripts.play --run data/runs/amass_s1 --motion-dir data/motions/h12_amass \
    --clip <name> [<name> ...] --video [--out <dir>]
```

- A config file has `num_envs`, `runner:` and `rewards:`; an unknown key is an error, and so is a run name whose
  directory exists. `--num-envs` overrides the config's count; `run.json` records the count used. The scripts run
  headless.
- `--max-iterations`: on a new run and on `--resume`, the iteration to stop at; on `--fine-tune-from`, the number
  of iterations to add. Default: the config's (or `run.json`'s) `max_iterations`.
- `--max-seconds`: the wall time of this invocation's training loop. When it is reached the run saves a checkpoint and
  exits 0. The clock is checked after each iteration, so a run can overrun by one iteration including an update
  window. The compile and CUDA-graph capture before the first update (about 8 minutes) counts against it, and is
  paid again on every resume.
- A fine-tune works as `h12_train` made `amass_s1_golem_aware`: the new directory starts from the source's
  `checkpoint.pt` (world model, actors, critics, optimisers, normalisers, reset sampler), the source's iteration
  count, wall time and `progress.csv` carry on, and `run.json` records `fine_tuned_from` and the end iteration
  (start + added). The source directory is not touched.
- A fine-tune must use the source run's clips, the same files in the same order: the reset sampler's saved per-clip
  statistics belong to them. Other clips are refused before the app starts.
- The sim context (the world model's and critics' privileged per-env input, `supermimic/sim_context.py`; `run.json`'s
  `sim_context` records the layout). A new run takes every quantity its env draws: the base envs foot friction,
  torso COM and encoder bias (31 columns); `--plant main --fine-tune`'s env (`TrackMainFineTuneEnvCfg`) its foot
  friction (the feet shapes' mean static friction from main's per-shape material draw), encoder offsets, target
  delay `action_delay` (0-10 physics substeps, normalised as 5 +- 5), `actuator_gain` and `actuator_strength` (main's
  `randomize_actuator_scales`: per env and joint at every reset, one gain scaling kp and kd together, so it is both
  the stiffness and the damping scale; strength scales kp, kd and the effort limit; normalised over the event's own
  ranges, the ankles' gains 0.5-1.2, the rest 0.8-1.2, strength 0.85-1.15) and `torso_mass` (kg added at start-up,
  -2 to 4, normalised as 1 +- 3). A `--fine-tune-from` keeps the source's columns in its order (one its env does not
  draw, as main's torso COM, held at the source's mean), widens a live column's normalisation where this env draws
  wider about the same centre (main's encoder offsets, +-0.02 and +-0.04 at the ankle pitch, against the base's
  +-0.01), and appends the env's quantities the source lacks; the loaded world model's and critics' first layers are
  widened to match (zero weights on the appended columns, the renormalised columns' weights rescaled), so they
  compute the source's function until trained. Friction keeps the source's normalisation (0.3-1.2; main draws
  0.1-1.25 about another centre). One line at the start names what was appended, held and renormalised. A
  `--resume` keeps the run's layout unchanged.
- In a delayed env (the fine-tune's, main's in stage 2) the world model's action input is the step's effective
  target: the mean over its substeps of the targets applied, the step's own, the previous step's and the one before,
  by the env's substep rule and each env's delay from the sim context (`Dynamics.effective_target`). The replay
  buffer stores each step's two previous targets as the env holds them (`prev_target`; after a reset the pose it
  starts in); world-model training feeds them per step, imagination starts from the sampled step's and rolls its own
  targets in. A delay of 0, and an env without one, give the step's own target as before; the weights are unchanged.
- `PRETRAINING_CUDA_GRAPHS=0` turns off CUDA-graph capture of the updates (crosscheck and play turn it off
  themselves). A capture that fails stops the run with an error saying to set it: on torch 2.7 a failed capture
  leaves the CUDA random-number generator unusable for the rest of the process.
- The cross-check rolls each clip from its first frame for its length, at most `--seconds` (default 10 s), with
  nominal physics (foot friction 0.6, no COM offset, no encoder bias, no noise) and deterministic actions, and scores
  against the reference at the frame after each step: body position error (mean over the 14 tracked bodies, world,
  m), pelvis position error (m), joint error (mean absolute over the 27 joints, rad), and per clip the share of
  steps with each `Safety/*` flag.

## PULSE on the cooperative task

`LocoManip-Marl-Pulse-v0` (main's `locomanip_marl/marl_pulse_env_cfg.py`, `mdp/pulse_actions.py`; design in
`docs/superpowers/specs/2026-11-24-fair-comparison-design.md`, piece 3; the first version, at 50 Hz on this package's
plant, in `2026-11-24-pulse-cooperative-design.md`) is `LocoManip-Marl-Torso-v0` (Marl-Direct with the arms agent
driving the torso) with each agent acting in the 32-wide latent of its frozen stage-1 actor: the env decodes
`z = tanh(u + prior(proprio))` with the stage-1 decoder (`marl/pulse.py`) into joint targets, main's default pose +
the plant's action scale x output, clipped to the plant's target limits within GOLEM's position envelope. The plant is
main's throughout (100 Hz, explicit PD at main's gains, armature, default pose and joint ranges, main's full
randomisation), so the stage-1 run must have been trained on it (`run.json`'s `"plant": "main"`; another is refused).
The gait, stance and torso posture rewards are at weight 0. The latent prior (`marl/actors.py::LatentPrior`, ported
from h12_train) is fitted first on the stage-1 actors tracking the clips (`scripts/fit_prior.py`: their run's clips
from its `run.json`, frames drawn uniformly) and written beside them as `prior.pt` and `prior.json` (per agent the
mean squared error, the latent variance and the share of it explained, over the last 500 steps). The env cfg's
`pulse_stage1_run` (default `data/pretraining/runs/main_s1_finetune`, relative to the repository root) names the
stage-1 run, `pulse_prior` (default `prior.pt`, relative to that run's directory; `null` or empty: no prior,
`z = tanh(u)`) the prior; both are hydra overrides (`env.pulse_stage1_run=...`, `env.pulse_prior=null`). The skrl
agent is main's yaml as it is (`agents.skrl_mappo_pulse_cfg`).

```bash
PY=/home/max/miniconda3/envs/env_isaaclab/bin/python

# fit the prior -> <stage-1 run>/prior.{pt,json}
timeout -s KILL 5400 $PY -m pretraining.scripts.fit_prior data/pretraining/runs/main_s1_finetune \
    [--out <file>] [--num-envs 2048] [--iterations 6000] [--hidden-dims 512 256] [--hide-last-action]

# 40k steps at 4,096 envs -> logs/skrl/locomanip_marl/<time>_<name>; the pure-RL arm is the same command with
# --task LocoManip-Marl-Torso-v0
$PY scripts/skrl/train.py --headless --task LocoManip-Marl-Pulse-v0 --num_envs 4096 --max_iterations <N> --name <name>
```

Scoring a latent policy (this env's skrl runs and `train_commands` runs): `evals/README.md`, "Latent policies". Not
done for one: export and the deploy gate.

## Stage 2 through the world model

Arm C of the comparison (`docs/superpowers/specs/2026-11-24-fair-comparison-design.md`, piece 4): h12_train's
`train_commands` on main's task. A command encoder per agent (`marl/command_actor.py`: an LSTM of 64 with main's cell
cap, then an MLP, as main's actor) reads main's actor observation and its command (the legs the velocity command, the
arms both wrist goals) and outputs the 32-wide latent of the frozen stage-1 decoder. `marl/command_runner.py`
collects data in `LocoManip-Marl-Pulse-v0` built on the stage-1 run without a prior (`marl/marl_source.py`; the env's
`u = atanh(z)`), keeps training the stage-1 world model on it, and trains the encoders by back-propagating, through
64-step imagined rollouts from buffered states, main's velocity and wrist tracking terms on the predicted state plus
stage 1's e-stop penalties (`marl/main_rewards.py`), with main's actor observation synthesised from it
(`marl/main_obs.py`) and main's command law (`marl/command_law.py`). A quarter of the imagined command segments
follow a clip, and on them the latent is pulled toward the stage-1 encoder's (weight 0.1). The stage-1 run must be on
main's plant; another is refused. The world model's sim context is read from the Pulse env each step, column by
column of the stage-1 run's layout (`MarlEnvSource.sim_ctx`): main's feet friction, encoder offsets, own target delay
(`_delay_steps`), actuator gains and strength and torso mass; a quantity main does not draw takes its true nominal
value (the torso COM offset 0), and a column it can neither read nor knows the nominal value of is an error. What the training reward leaves out of main's is printed at the start.

```bash
PY=/home/max/miniconda3/envs/env_isaaclab/bin/python

# smoke: 20 iterations (480 steps per env), to read peak_gpu_gb and the seconds per iteration off the log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout -s KILL 2400 $PY -m pretraining.scripts.train_commands \
    --stage1 data/pretraining/runs/<main-plant stage-1 run> --run-name commands_smoke --max-steps 480

# the run: 40,032 steps per env (1,668 iterations of 24) or a wall-time cap, whichever comes first
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout -s KILL 14400 $PY -m pretraining.scripts.train_commands \
    --stage1 data/pretraining/runs/<main-plant stage-1 run> --run-name <name> --max-seconds 12600

# continue it (only the caps are taken from the command line)
$PY -m pretraining.scripts.train_commands --resume data/pretraining/runs/<name> [--max-steps N] [--max-seconds S]
```

`configs/commands.yaml` holds the settings and the memory estimate behind them (4,096 envs; world model 10 x 1,024 x
16; policy 128 x 64); `--num-envs`, `--rollout`, `--buffer-size`, `--replay-ratio`, `--wm-batch-size`,
`--wm-horizon`, `--policy-batch-size`, `--policy-horizon` and `--max-clips` override it. The run directory holds
`run.json` (stage-1 run, plant, task, settings, seed, env count), `progress.csv` (one row per iteration, with
`peak_gpu_gb`, the world-model loss, each agent's imagined return and latent-BC term, the sim's velocity and wrist
errors), `commands.pt` (both encoders and the stage-1 run they decode through), `world_model.pt` and `checkpoint.pt`.

## Known differences from `h12_train`

Each is deliberate; what was measured or decided is stated.

- **Robot model.** The robot is `main`'s H1-2 USD (CL_Assets), with `h12_train`'s gains, armature, default pose,
  action scales, soft limits (0.9) and no joint damping or friction loss. The Magpie grippers are articulated fingers
  in this USD (rigid masses in `h12_train`'s MJCF); their hinges are not policy joints and are held at joint angle 0
  (`main`'s rest pose, jaws open) by stiff drives, the pose at which `h12_train` composed their rigid inertia.
  `main`'s start-up fix for the four massless frames is applied. Collision shapes differ (USD meshes and convex
  hulls against capsules and four sole capsules per foot).
- **Joint ranges.** `main`'s USD has shoulder roll at ±0.19 rad; `h12_train` and GOLEM have ±0.38 rad. A start-up
  event sets `h12_train`'s joint ranges, because the clips were retargeted to them. This env's shoulder-roll range
  therefore differs from `main`'s fine-tuning env.
- **Physics.** PhysX contact against MuJoCo-Warp, implicit PD, self-collision on. This is the main source of the
  cross-check gap below.
- **Foot friction.** Randomised per env (0.3-1.2) as a PhysX material on every shape of both ankle-roll links, with
  the ground at friction 1 and combine mode "multiply", in place of MuJoCo's geom friction.
- **Torso COM context.** The world model's torso-COM context is the COM offset from its USD-authored value, not
  MuJoCo's absolute body COM. The world model is trained fresh in Isaac, so an `h12_train` `world_model.pt` is not
  meaningful here; actors are interchangeable.
- **`Safety/*_force_saturated`.** Computed from Isaac's applied (clipped) joint torque, the closest equivalent of
  MuJoCo's actuator force.
- **Forward kinematics.** In torch from the CL_Assets URDF (`kinematics.py`), not from the MJCF; checked against
  Isaac's body poses in a sim test.
- **Reference body poses.** Computed on demand, by forward kinematics on the frames indexed each step, not
  precomputed for the whole dataset at load. Precomputing took 9.56 GB of GPU memory at load (about 3.5 GB resident)
  for the 13,578 clips; on demand the whole dataset takes 1.30 GB and loads in 9.3 s. The values are the same.
- **Muon optimizer.** SuperMimic takes `torch.optim.Muon`, which `env_isaaclab`'s torch 2.7 lacks (added in 2.9).
  `supermimic/learners/muon.py` is a copy of torch's, used only when torch has none.
- **Cross-check fall rule.** A fall is the pelvis below 0.6 m or tilted more than 1.0 rad, the rule of `h12_train`'s
  `render_stage1`, so falls compare like for like.

## Memory and speed

Measured on the development machine: one RTX 5070 Ti, 16 GB (15.46 GB usable), shared by Isaac's simulation and
the learner. Isaac itself holds about 5 GB.

- **5,120 envs (`amass_s1`'s count) do not fit.** The first attempt ran out of memory building the motion loader for
  the full dataset: 9.56 GB of torch allocations for precomputed body tensors on top of Isaac's. With reference body
  poses computed on demand, the second attempt got through set-up (torch 5.33 GB at iteration 0: motion data, replay
  buffer, models) and ran out of memory in the first policy update: torch 9.68 GB plus Isaac's 5.07 GB on 15.46 GB.
  The estimate is about 0.64 MB of replay buffer per env, about 6 GB of transient memory per update and about 5 GB
  for Isaac, so 5,120 envs need about 16.3 GB.
- **2,048 envs fit.** A 600-iteration measurement run: torch peak 9.33 GB, GPU total 14.69 of 15.46 GB,
  0.29 s per iteration.
- **Fewer updates per iteration.** The learner's batch sizes and replay ratio stay as configured, so updates per
  iteration scale with the env count: 2 per 10 iterations at 2,048 envs, against `h12_train`'s 5 per 10 at 5,120
  envs (40%).
- **Compile.** About 8 minutes (470 s measured at 1,024 envs) of compilation and CUDA-graph capture before the first
  update, on every start and resume.

## Results

**Cross-check: the MuJoCo-trained policies in Isaac.** Both `h12_train` policies, loaded into this env, on the same
40 clips (seed 0, including the 4 clips `h12_train`'s render scored in MuJoCo), nominal physics:

| policy | falls (of 40) | body position error | pelvis position error | joint error |
|---|---|---|---|---|
| `amass_s1` | 0 | 0.236 m | 0.234 m | 0.064 rad |
| `amass_s1_golem_aware` | 0 | 0.306 m | 0.303 m | 0.068 rad |
| `isaac_s1` | 0 | 0.178 m | 0.176 m | 0.079 rad |
| `isaac_s1_golem_aware` | 0 | 0.170 m | 0.163 m | 0.087 rad |

The two Isaac-trained rows were added after their runs finished, on the same clip list (`--clips-file`); they have no
MuJoCo number, so the 4-clip table below covers the MuJoCo-trained policies only.

On the 4 clips scored in both simulators (means of the 4):

| policy | MuJoCo body / joint error | Isaac body / joint error |
|---|---|---|
| `amass_s1` | 0.193 m / 0.062 rad | 0.258 m / 0.068 rad |
| `amass_s1_golem_aware` | 0.190 m / 0.066 rad | 0.442 m / 0.073 rad |

Joint errors are within about 0.006 rad of MuJoCo's. The extra body position error is almost all pelvis error: the
whole robot drifts in the world frame while the joints track. One clip accounts for most of the GOLEM-aware gap
(0.882 m in Isaac against 0.280 m in MuJoCo). On one of the 40 clips an arm joint's applied torque is at its effort
limit on 57% (`amass_s1`) and 64% (GOLEM-aware) of steps; on every other clip 9% or less, and the legs never
saturate. What this shows: the port runs the MuJoCo policies without falls and with near-equal joint tracking, and a
kinematic replay reads near-zero error (sim test). What it does not show: only 4 clips have a MuJoCo number to
compare with, so the drift gap between the simulators is measured on those 4 clips only.

**Early training against `amass_s1`.** At an equal number of updates (117: the Isaac measurement run's iteration 590
at 2,048 envs, its last row, against `amass_s1`'s iteration 240), the Isaac measurement run's curve matches
`amass_s1`'s: `Loss/state` 0.039 against 0.041, body position error 0.334 m against
0.370 m, mean episode length 60 against 50 (as logged in `progress.csv`).

**The four policies.** Errors and falls from the cross-check in Isaac on the same 40 clips; `Safety/*` shares are
the mean over the 40 clips of the share of steps with the flag.

| policy | trained in | iterations | wall time | envs | body position error | joint error | falls (of 40) | legs / arms force saturated | legs q over e-stop | arms torque demand over 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `amass_s1` | MuJoCo | 320,000 | 26.4 h | 5,120 | 0.236 m | 0.064 rad | 0 | 0.02% / 2.2% | 2.7% | 2.4% |
| `amass_s1_golem_aware` | MuJoCo | 340,000 (320,000 + 20,000) | 28.1 h (26.4 + 1.7) | 5,120 | 0.306 m | 0.068 rad | 0 | 0.13% / 2.3% | 0.19% | 2.5% |
| `isaac_s1` | Isaac | 79,165 | 6.4 h | 2,048 | 0.178 m | 0.079 rad | 0 | 0.03% / 0.50% | 1.9% | 0.70% |
| `isaac_s1_golem_aware` | Isaac | 96,131 (79,165 + 16,966) | 7.8 h (6.4 + 1.4) | 2,048 | 0.170 m | 0.087 rad | 0 | 0.09% / 1.5% | 0.01% | 2.0% |

Iterations and wall time are the last row of each run's `progress.csv` (`wall_s`: 23,000 s and 28,200 s for the
Isaac runs; the fine-tune's file carries the base run's rows). Updates, summed from `updates`: `amass_s1` 159,992,
`amass_s1_golem_aware` +10,221, `isaac_s1` 15,832, `isaac_s1_golem_aware` +3,384 (19,216 in all).

**Isaac against MuJoCo.** Training-rollout metrics from `progress.csv`, each the mean over the 50 logged rows (500
iterations) ending at the point. Body error is `Metrics/motion/error_body_pos` (pelvis-relative), pelvis error is
`error_anchor_pos` (global). `Episode_Termination/*` is logged on a different scale by the two runs (the Isaac values
sum to about 1, the MuJoCo ones to about 20), so the table gives each reason's share of their sum. `legs q over
e-stop` is not logged by `amass_s1`.

| point | run | iteration | updates | `Loss/state` | body error | pelvis error | episode length | time-out / motion end / pelvis-position share | legs / arms force saturated | legs q over e-stop | arms torque demand over 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| equal updates | `amass_s1` | 31,670 | 15,832 | 0.0031 | 0.071 m | 0.266 m | 499.3 | 12.9% / 86.9% / 0.12% | 0.11% / 1.29% | n/a | 1.37% |
| equal wall time (23,000 s) | `amass_s1` | 77,450 | 38,722 | 0.0028 | 0.054 m | 0.279 m | 497.9 | 19.0% / 80.7% / 0.39% | 0.17% / 0.96% | n/a | 1.15% |
| end of base run | `isaac_s1` | 79,165 | 15,832 | 0.0047 | 0.058 m | 0.394 m | 497.4 | 26.6% / 72.8% / 0.57% | 0.31% / 1.36% | 5.2% | 1.29% |
| end of base run | `amass_s1` | 319,990 | 159,992 | 0.0027 | 0.057 m | 0.246 m | 498.5 | 19.8% / 79.9% / 0.31% | 0.14% / 1.01% | n/a | 1.09% |
| end of fine-tune | `isaac_s1_golem_aware` | 96,131 | 19,216 | 0.0050 | 0.063 m | 0.461 m | 487.5 | 27.6% / 71.7% / 0.69% | 0.41% / 1.71% | 0.08% | 1.71% |
| end of fine-tune | `amass_s1_golem_aware` | 339,990 | 170,213 | 0.0028 | 0.058 m | 0.335 m | 493.2 | 20.1% / 79.4% / 0.32% | 0.20% / 1.05% | 0.03% | 1.09% |

Pelvis-orientation terminations are 0.11% or less of the sum everywhere. In training rollouts the Isaac runs end
with a higher `Loss/state` (0.0047 against 0.0027-0.0031) and a larger global pelvis error (0.394 m against 0.246 to
0.279 m) than `amass_s1` at any of the three points, a body error between `amass_s1`'s at equal updates and at its
end, and the same episode length to within 2%. The GOLEM-aware fine-tune cut the legs' e-stop share in both
simulators (Isaac 5.2% to 0.08%) and raised the global pelvis error in both. What this does not show: the training
numbers come from two different simulators, so they do not say which policy tracks better. The cross-check above
runs all four in one simulator, but that simulator is Isaac, the Isaac-trained policies' own training simulator
(the MuJoCo-trained ones run there out of distribution), so it favours them; a symmetric check of the Isaac actors
in MuJoCo was not run, and it does not show that the Isaac policies track better. In it both Isaac-trained policies
have a lower body and pelvis error (0.170-0.178 m against 0.236-0.306 m) and a higher joint error (0.079-0.087 rad
against 0.064-0.068 rad) than the MuJoCo-trained
ones, none falls, and their arm saturation and torque-demand shares are lower; that is 40 clips, nominal physics,
and one seed per run.

**Why Isaac did 2.5 times fewer updates per hour.** From the summed per-interval timings in `progress.csv`
(`interval_s`, `collect_s`, `wm_s`, `policy_s`, `updates`):

| run | envs | updates per hour | updates per iteration | env steps/s (mean) | per update: collect / world model / policy |
|---|---|---|---|---|---|
| `amass_s1` | 5,120 | 6,081 | 0.50 | 17,299 | 0.053 / 0.278 / 0.261 s |
| `isaac_s1` | 2,048 | 2,484 | 0.20 | 7,088 | 0.372 / 0.587 / 0.491 s |

Three things add up: fewer envs, so 2.5 times fewer updates per iteration (the replay ratio ties updates to collected
steps); slower updates, 1.08 s of world-model plus policy update against 0.54 s; and
slower simulation, 7,088 env steps/s against 17,299 (collection takes 0.074 s per iteration for
2,048 envs against 0.027 s for 5,120). Why the updates are about twice as slow is not yet measured.

## What was not done

- Export and the deploy gate for a latent policy.
- The full-length run. The long run had a wall-time budget of 8 hours, split into the base run (23,000 s) and the
  GOLEM-aware fine-tune (5,200 s); `amass_s1` took 26.4 hours in MuJoCo (94,945 s), at 2.5 times the updates per
  iteration.
