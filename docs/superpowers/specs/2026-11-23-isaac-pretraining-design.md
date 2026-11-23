# Isaac Lab pretraining for the H1-2 (`pretraining/`): design

Date: 2026-11-23. Branch: `pretraining`, off `origin/main` of Locomanipulation_game.

## Intent

Rerun our stage-1 pretraining in Isaac Lab: two policies, one for the legs and one for the
upper body, that make the H1-2 follow the retargeted AMASS clips. The method is
SuperMimic's: a world model learned from simulator rollouts, and policies trained by
gradients through that model. Today this exists only in `h12_train` (branch `supermimic`),
which runs on mjlab / MuJoCo-Warp through the `supermimic` submodule. `main` has neither.
The Isaac Lab version is the base that later RL fine-tuning in Isaac Lab starts from.

Decided by Max:
- Port SuperMimic's world-model gradient method; do not replace it with PPO.
- Keep the split body: a legs agent and an upper-body agent.
- The main purpose is rerunning pretraining. Fine-tuning comes later and is not in this work.
- Do not push data. Document the expected format instead.
- Approach C (below).
- Done means the full pretraining has run in Isaac Lab and produced both policies, within
  8 hours of wall time for the long run.
- The `h12_train` work is committed on `supermimic` (b175f1c) and this branch is checked
  out in place; a read-only copy of `h12_train` for reference is at `data/ref/h12_train`
  (git-ignored).

## Goals

1. A top-level `pretraining/` directory on a branch off `main` that trains, from the
   retargeted clips, the base tracker and its GOLEM-aware fine-tune in Isaac Lab.
2. The trained policies keep `h12_train`'s policy contract (below), so a policy from either
   simulator loads in the other's tooling.
3. A cross-check before the long run: the existing `amass_s1` actors, loaded into the Isaac
   task, with their tracking errors there set beside their MuJoCo numbers.
4. Both full runs done in Isaac Lab, with their tracking and safety numbers compared with
   `amass_s1` and `amass_s1_golem_aware`.
5. A README that states the data format, the commands, and the measured results.

## Non-goals

- Stage 2 (velocity and hand-goal commands) and any RL fine-tuning.
- Changing `main`'s `locomanip_marl` tasks, its trainer, or `evals/`.
- Committing motion data, policies, or anything else derived from AMASS or SMPL-X.
- Reproducing the existing weights bit for bit. PhysX is not MuJoCo; the Isaac runs are new
  policies trained by the same method.

## Anti-goals

- A tracker that silently differs from `h12_train` in observation layout, action scale,
  gains or joint order. Any difference is deliberate, listed in the README, and measured.
- Importing the `supermimic` or `mjlab` packages. They need Python 3.13; Isaac Lab here is
  Python 3.11 (isaaclab 0.54.4, torch 2.7).
- Starting the 26-hour run before the cross-check and a memory measurement.

## Constraints

- **Licence of the data.** AMASS and SMPL-X are non-commercial and not redistributable, and
  `tools/amass/README.md` extends that to the clips and to policies trained on them. Clips
  and runs live under git-ignored paths. The two existing stage-1 policies are therefore
  referenced by path and described, not committed.
- **Licence of the code.** SuperMimic and mjlab are Apache-2.0. Ported files keep a header
  naming the upstream file and commit, and `pretraining/NOTICE` lists them.
- **GPU.** One RTX 5070 Ti, 16 GB. `amass_s1`'s learner peaked at 10.8 GB with 5,120 envs
  and Isaac's simulation shares the card.
- **Environment.** `conda activate env_isaaclab`. No new dependency unless unavoidable;
  `mujoco` is not installed there and is not to be required.
- **Repo rules.** New task ids contain `LocoManip-` (`scripts/list_envs.py` lists only
  those). Never edit `/home/max/ALMI-Open`. Nothing is pushed unless Max asks.

## Approaches considered

- **A. Swap only the simulator.** Copy SuperMimic's learner, world model and replay buffer
  and `h12_train`'s two-agent runner; put an Isaac Lab env underneath that returns the same
  tensors. Narrow seam, checkable against `amass_s1`; code reads like SuperMimic.
- **B. Rebuild in `main`'s style.** A new tracking task from `main`'s terms and a new
  learner. Rejected: `main`'s gains, stance and action scales differ from `h12_train`'s, so
  the result is not interchangeable with `amass_s1` and nothing can be cross-checked.
- **C. A, on `main`'s robot asset.** As A, but the robot is `main`'s H1-2 (the CL_Assets
  USD `main` already loads) with `h12_train`'s gains, limits and action scales, and the
  tasks registered as `LocoManip-Pretrain-*`. One robot model in the repo; the cross-check
  then shows a small gap to be judged. **Chosen by Max.**

## The policy contract (from `h12_train`, unchanged)

- 50 Hz control. Joint order: Unitree motor order, 27 joints (12 legs, torso, 14 arms).
- Agents: legs = joints 0-11 (12 actions); upper body = torso + arms, joints 12-26 (15).
- Target = `default_joint_pos + action_scale * action`, clipped to GOLEM's target limits
  (`h12_constants.GOLEM_TARGET_LIMITS`); PD at `h12_constants` kp, kd; effort limits from
  the same file. Default pose, gains and scales are `h12_train`'s, not `main`'s sport-mode
  values.
- Actor observation ("proprio", history length 1): one 87-wide frame: projected gravity (3)
  and gyro (3) of the torso IMU (URDF `imu_link`, on `torso_link` at
  (-0.04452, -0.01891, 0.27756) m), `q - default` (27), `dq` (27), last action of both
  agents (27). Noise half-widths as in `h12_train`: gravity 0.05, gyro 0.2, joint position
  0.01, joint velocity 0.5.
- Each actor is SuperMimic's `MultiEncoderAE`: encoder(proprio, goal) -> 32-wide tanh
  latent -> decoder(proprio, latent); hidden sizes 2048, 1024, 512, 256. Stage 2 later
  freezes the decoders.
- Goal of each actor: the reference window SuperMimic's gencoord representation builds from
  the clip, as `h12_train.marl.runner.imitation_goal_dims` splits it between the agents.
- Tracked bodies (14): the list in `h12_train/tasks.py`. Anchor body: `pelvis`
  (`tasks.py::apply_gencoord`, the task `amass_s1` trained on).

## What gets built

```
pretraining/
  README.md            what this is, setup, data format, commands, results, differences from h12_train
  NOTICE               upstream files ported (SuperMimic, mjlab), their licence and commits
  pretraining/         the Python package (pip install -e pretraining)
    robot.py           main's H1-2 ArticulationCfg with h12_train's gains, limits, default pose, action scales
    motions.py         clip loader: the .npz format below -> per-frame joint and body tensors
    kinematics.py      forward kinematics in torch for the 27 joints, built from the CL_Assets URDF
    tasks/             the Isaac Lab tracking env and its terms; gym ids LocoManip-Pretrain-Track-v0 (+ -Play-v0)
    supermimic/        ported, mjlab-free: world model, representation, replay buffer, learners, world-model env
    marl/              ported from h12_train: two-agent actors, learner, runner, rewards, run files
    scripts/train.py   train or resume a run (base or GOLEM-aware by config)
    scripts/crosscheck.py  load an h12_train run's actors into the Isaac task and report tracking errors
    scripts/play.py    roll out a run on chosen clips and report / render
  configs/             base.yaml, golem_aware.yaml: every setting of the two runs
  tests/               unit tests with procedurally generated clips (no mocap)
```

The Isaac Lab env is manager-based, mirroring mjlab's tracking task term for term (mjlab's
tracking task is itself a port of an Isaac Lab one, and its math helpers are Isaac Lab's):
the motion command (reset to a sampled clip frame with pose and velocity perturbation,
anchor and body targets), the actor, full-state and goal observation groups, the
terminations `amass_s1` ran with (anchor position, anchor orientation, motion end,
time-out; no end-effector-height termination), its events (foot friction, torso COM,
encoder bias; no pushes), the tracking metrics, and the safety counters (`Safety/*`).
Where this document and `h12_train/tasks.py::apply_gencoord` (with
`h12_flat_tracking_env_cfg`) disagree, the code is right: the Isaac env is the env
`amass_s1` trained in. Physics step 0.005 s, decimation 4, as there. The world-model side (imagined rollouts,
rewards on predicted states, soft terminations, estop penalties) is torch code ported as
is.

Run outputs keep `h12_train`'s layout (`run.json`, `progress.csv`, `world_model.pt`,
`actors.pt`, `critics.pt`, `checkpoint.pt`, `checkpoints/`) under
`$PRETRAINING_RUNS_DIR`, default `data/pretraining/runs/` (git-ignored by `/data/`).
`actors.pt` from either simulator loads with either codebase's loader.

## Known differences from `h12_train` (to be listed in the README with numbers)

- **Robot model.** `main`'s USD has the Magpie grippers as articulated fingers with hinge
  joints; `h12_train`'s MJCF has them as rigid masses. The fingers are held closed by
  stiff actuators and are not policy joints. Collision shapes differ (USD meshes/convex
  hulls against capsules and four sole capsules per foot).
- **Physics.** PhysX contact against MuJoCo-Warp. The asset is altered to match
  `h12_train`, not `main`'s fine-tuning env (my call; the contract goal needs it): implicit
  PD at `h12_constants`' kp and kd, its per-joint armature, no joint damping or friction
  loss, physics step 0.005 s with decimation 4, self-collision on, and `main`'s start-up
  fix for the four massless frames the USD import gives 1 kg each. A later fine-tuning env
  that uses `main`'s explicit PD and gains is a change the fine-tuning work must handle.
- **IMU.** Torso IMU read from the `torso_link` body's orientation and angular velocity
  (the IMU is rigidly on that link, so gravity and gyro are the same as at the site).
- **Forward kinematics.** From the CL_Assets URDF in torch, checked against Isaac's body
  poses in a test, instead of mjlab's from the MJCF.

## Data format (documented in the README; nothing committed)

A motion directory of `.npz` files, one per clip, 50 Hz, as written by `tools/amass` and
`h12_train.scripts.clip_to_motion` (on the `supermimic` branch):

| key | shape | meaning |
|---|---|---|
| `fps` | (1,) | 50.0 |
| `joint_names` | (27,) str | Unitree motor order |
| `joint_pos`, `joint_vel` | (T, 27) float32 | rad, rad/s |
| `root_pos_w` | (T, 3) | pelvis position, m, world |
| `root_quat_w` | (T, 4) | pelvis orientation, wxyz |
| `root_lin_vel_w`, `root_ang_vel_w` | (T, 3) | m/s, rad/s, world |

Default location `data/motions/h12_amass/` (13,578 clips, 1.3 GB, on this machine); a
`--motion-dir` option overrides it. Body poses are computed from these by forward
kinematics at load. The README also says where the two existing MuJoCo-trained policies
are expected for the cross-check (`data/runs/amass_s1`, `data/runs/amass_s1_golem_aware`).

## The two runs and their settings

- **Base** (`configs/base.yaml`): `amass_s1`'s settings from its `run.json`: 5,120 envs,
  up to 320,000 iterations (stopped by the wall-time budget below), world-model lr 2e-4 / batch 2048 / horizon 8 / 10 batches, policy
  lr 2e-4 / batch 1024 / horizon 32, replay ratio 16, buffer 1000, return-weighted reset
  sampling, the reward weights and standard deviations as recorded, estop velocity and
  torque penalties -5 at margin 0.9.
- **GOLEM-aware** (`configs/golem_aware.yaml`): a fine-tune of the base run for 20,000
  iterations with the position-estop penalty -20 at margin 0.05 rad added.

If 5,120 envs do not fit in 16 GB beside Isaac, the env count drops to the largest that
fits and the README records it; the learner's batch sizes and replay ratio stay.

**Wall-time budget (Max, 2026-11-23): 8 hours at most for the long run.** The two runs
share it: the base run stops at 7.5 hours of wall time (`--max-seconds`), the GOLEM-aware
fine-tune at 0.5 hours, each at whatever iteration that is. For scale, `amass_s1` reached
iteration 97,000 of 320,000 after 8 hours in MuJoCo, with tracking errors by then close to
its final ones (body position 0.04-0.06 m throughout hours 4 to 26). The README records
the iterations reached. State the plan to Max before starting the run.

**Cross-check rule (approved by Max).** The cross-check is a diagnosis of the
port, not a bar the old policy must clear. Hard requirements before training: a clip
replayed kinematically reads zero tracking error, and the observation frame the Isaac env
builds equals `h12_train`'s for the same state. `amass_s1`'s fall rate and body-position
error in Isaac are then reported beside its MuJoCo numbers. If it falls on more than a
fifth of the fixed clips, up to half a day goes to finding a port bug or an obvious model
mismatch; after that the gap is recorded as a physics difference and training proceeds.

## Order of work and checks

1. Branch, package skeleton, robot config, motion loader, kinematics; unit tests.
2. The Isaac tracking env; a test that replays a clip kinematically and reads zero error.
3. Port the learner stack; tests on a small procedural clip set.
4. Cross-check: `amass_s1` actors in the Isaac env on a fixed clip set, errors beside
   `h12_train`'s render numbers for the same clips.
5. Short training run (a few thousand iterations): losses fall, tracking error falls,
   memory and speed measured.
6. Base run, then the GOLEM-aware fine-tune, inside the 8-hour budget. Compare final
   numbers with the MuJoCo runs.
7. README with results; commit on the branch. Not pushed unless Max asks.

## Left to the builder

File-level layout inside the package, the term names, how the config files are parsed,
test structure, and how run comparison tables are produced.
