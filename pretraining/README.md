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

# continue a run that stopped (only --max-iterations, --max-seconds and --device are accepted beside it)
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
`docs/superpowers/specs/2026-11-24-pulse-cooperative-design.md`) is `LocoManip-Marl-Direct-v0` with each agent acting
in the 32-wide latent of its frozen stage-1 actor: the env decodes `z = tanh(u + prior(proprio))` with the stage-1
decoder (`marl/pulse.py`) into joint targets, default + `ACTION_SCALE` x output, at 50 Hz on main's explicit PD with
this package's gains, armature, default pose and joint ranges, under main's full randomisation, with the gait, stance
and torso posture rewards at weight 0. The latent prior (`marl/actors.py::LatentPrior`, ported from h12_train) is
fitted first on the stage-1 actors tracking the clips (`scripts/fit_prior.py`: their run's clips from its `run.json`,
frames drawn uniformly) and written beside them as `prior.pt` and `prior.json` (per agent the mean squared error, the
latent variance and the share of it explained, over the last 500 steps). The env cfg's `pulse_stage1_run` (default
`data/pretraining/runs/isaac_s1_golem_aware`, relative to the repository root) names the stage-1 run, `pulse_prior`
(default `prior.pt`, relative to that run's directory; `null` or empty: no prior, `z = tanh(u)`) the prior; both are
hydra overrides (`env.pulse_stage1_run=...`, `env.pulse_prior=null`). The skrl agent is main's yaml at discount 0.99.

```bash
PY=/home/max/miniconda3/envs/env_isaaclab/bin/python

# fit the prior (h12_train's settings: 4,096 envs, 6,000 steps) -> data/pretraining/runs/isaac_s1_golem_aware/prior.{pt,json}
timeout -s KILL 5400 $PY -m pretraining.scripts.fit_prior data/pretraining/runs/isaac_s1_golem_aware \
    [--out <file>] [--num-envs 4096] [--iterations 6000] [--hidden-dims 512 256] [--hide-last-action]

# smoke run: 300 updates (14,400 steps) at 4,096 envs -> logs/skrl/locomanip_marl/<time>_PulseSmoke
timeout -s KILL 10800 $PY scripts/skrl/train.py --headless --task LocoManip-Marl-Pulse-v0 --num_envs 4096 \
    --max_iterations 300 --name PulseSmoke

# the long run: 200k steps (4,167 updates of 48) or 12 hours, whichever comes first (checkpoints every 9,600 steps)
timeout -s INT -k 600 43200 $PY scripts/skrl/train.py --headless --task LocoManip-Marl-Pulse-v0 --num_envs 4096 \
    --max_iterations 4167 --name Pulse200k
```

Not done for a latent policy: export, the evals and the deploy gate (they assume joint-space actions; the decoder has
to be in their loop first).

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

**Early training against `amass_s1`.** At an equal number of updates (120: iteration 600 at 2,048 envs), the Isaac
measurement run's curve matches `amass_s1`'s: `Loss/state` 0.039 against 0.041, body position error 0.334 m against
0.370 m, mean episode length 60 against 50 (as logged in `progress.csv`).

**The four policies.** Errors and falls from the cross-check in Isaac on the same 40 clips; `Safety/*` shares are
the mean over the 40 clips of the share of steps with the flag.

| policy | trained in | iterations | wall time | envs | body position error | joint error | falls (of 40) | legs / arms force saturated | legs q over e-stop | arms torque demand over 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `amass_s1` | MuJoCo | 320,000 | 26.4 h | 5,120 | 0.236 m | 0.064 rad | 0 | 0.02% / 2.2% | 2.7% | 2.4% |
| `amass_s1_golem_aware` | MuJoCo | 340,000 (320,000 + 20,000) | 28.1 h (26.4 + 1.7) | 5,120 | 0.306 m | 0.068 rad | 0 | 0.13% / 2.3% | 0.19% | 2.5% |
| `isaac_s1` | Isaac | run in progress | run in progress | 2,048 | run in progress | run in progress | run in progress | run in progress | run in progress | run in progress |
| `isaac_s1_golem_aware` | Isaac | run in progress | run in progress | 2,048 | run in progress | run in progress | run in progress | run in progress | run in progress | run in progress |

## What was not done

- h12_train's stage 2 (`CommandRunner`: velocity and hand-goal commands on the world model). Its PULSE mechanics
  (the frozen decoders, the latent prior) drive `LocoManip-Marl-Pulse-v0` instead (above), trained by main's MAPPO.
- Any change to `main`'s existing `locomanip_marl` tasks, its trainer, or `evals/`.
- The full-length run. The long run had a wall-time budget of 8 hours, split into the base run (23,000 s) and the
  GOLEM-aware fine-tune (5,200 s); `amass_s1` took 26.4 hours in MuJoCo (94,945 s), at 2.5 times the updates per
  iteration.
