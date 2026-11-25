# evals/

Every evaluation of the H1-2 locomanipulation policies: the legs and the arms, alone and together, in Isaac Lab and
in MuJoCo, and the gate an export passes before it goes on the robot. Run everything from the repo root as
`python -m evals.<...>` (Isaac ones in `env_isaaclab`; MuJoCo, plots and the gate in a python with mujoco, torch,
numpy, scipy, pyyaml, matplotlib and tensorboard; the gate's bag replay runs in GOLEM's ros container, docker).

```
evals/
  common/    no simulator: the metric code, the scorecard format, the command program, run discovery, plots
  isaac/     get_video.py; legs/ and arms/: get_scorecard.py, get_plot.py; arms/steerability/ (on demand, --steerability: about 1.5 h)
  mujoco/    get_video.py; legs/ and arms/: get_scorecard.py, get_plot.py; sim.py (the policy loop in MuJoCo)
  deploy/    gate.py + gate.yaml; golem/ (the run in GOLEM's sim); parity.py, bag_replay.py, stress_mujoco.py,
             test_sportmode_metrics.py
```

Results go to `logs/evals/<run folder>/<isaac|mujoco|deploy>/<legs|arms>/`.

## Scorecards

```bash
python -m evals.isaac.legs.get_scorecard --run LegsDeploy200k_s42              # every checkpoint not scored yet
python -m evals.isaac.legs.get_scorecard --checkpoint <run>/checkpoints/agent_<N>.pt [--arms_checkpoint <ckpt>|same]
python -m evals.isaac.arms.get_scorecard --run ArmsObs200k_s42                 # the IK-only reference first
python -m evals.isaac.arms.get_scorecard --checkpoint <ckpt> [--legs_checkpoint <ckpt>|same] [--steerability]
python -m evals.mujoco.legs.get_scorecard --run <run>                          # exports each checkpoint once
python -m evals.mujoco.arms.get_scorecard --run <run>
```

**Latent policies** (a policy in the frozen stage-1 decoders' latent, `isaac/latent.py`, `mujoco/latent.py`): an skrl
run of `LocoManip-Marl-Pulse-v0` (B) or a `pretraining.scripts.train_commands` run (C), both agents together, in place
of `--checkpoint`; a joint-space run that trained both agents (`LocoManip-Marl-Torso-v0`, A) is scored in its own task
with the other body `same`. The three then run the same program on the same conditions (C in the Pulse env with the
prior off and its own stage 1, and with the env overrides, IMU model and curriculum state of a Pulse skrl run,
`--settings_run`, which a C run must be given: it records none of its own):

```bash
python -m evals.isaac.legs.get_scorecard --checkpoint <A run>/checkpoints/agent_<N>.pt --arms_checkpoint same
python -m evals.isaac.arms.get_scorecard --checkpoint <A run>/checkpoints/agent_<N>.pt --legs_checkpoint same
python -m evals.isaac.legs.get_scorecard --latent_run <B run>[/checkpoints/agent_<N>.pt]
python -m evals.isaac.legs.get_scorecard --latent_run <C run> --settings_run <B run>
python -m evals.isaac.arms.get_scorecard --latent_run <B run>   # C: --settings_run <B run>, as above
python -m evals.isaac.get_video --latent_run <B or C run> [--settings_run <B run>]
python -m evals.mujoco.legs.get_scorecard --latent_run <B run>              # C: --settings_run <a Pulse skrl run>
python -m evals.mujoco.arms.get_scorecard --latent_run <C run> --settings_run <B run>
python -m evals.mujoco.get_video --latent_run <B run>
```

A latent arms scorecard has no IK-only reference (no IK). Its json carries `"latent": "skrl"|"commands"`. MuJoCo
decodes with the stage-1 run's decoders in TorchScript (`python -m pretraining.scripts.export_decoders <stage-1 run>
--out <dir> [--commands <C run>]`, which the MuJoCo evals run themselves in the isaac python, `$LMG_ISAAC_PYTHON`,
into `logs/evals/<run>/mujoco/exports/`); B's encoders are its export_marl export, C's its command encoders, and C's
deploy settings (gains, IMU model, rates, target clip) a Pulse skrl run's export (`--settings_run`: a train_commands
run records no env.yaml). `python -m evals.mujoco.test_latent` tests the MuJoCo side; its frame check needs
`pretraining/tests/fixtures/pulse_frame.json`, which a sim test records.

Each writes `scorecard_<N>.json` (the eval's results, its rows, the eval conditions and code commit) and
`scorecard_<N>.md` (one table per section: value, reference, target, met). A scored checkpoint is cached: `--force`
scores it again.

- **Legs**: sport mode's tracking metrics computed by the report's own code (`common/sportmode.py`, copied in;
  `deploy/test_sportmode_metrics.py` checks the copy still reproduces the report on the bags), direction cos and turn
  sense first; the response fit per axis; stride, foot clearance and lift, knee swing, steps/s, air time (from the
  leg angles, levelled, as the foot-lift analysis; and from foot heights and contacts); torso sway, falls, e-stops;
  per segment kind. Reference: sport mode. Targets: legs beat sport mode except on the speed and turn ratios.
- **Arms**: end position and rotation error, reach (5 cm / 20 deg, 1 mm / 5 deg, and the env's own tolerance),
  time to reach, vibration, wobble, variance. Reference: the IK alone on the same goals (`scorecard_ik_only`).
  Targets: 0-4 mm, 0-5 deg.
- **The other body's policy** is optional. Without it the legs eval holds every arm joint at 0 rad (`hold_zero`)
  and the arms eval pins the pelvis. With it (`--arms_checkpoint` / `--legs_checkpoint`, `same` for the run's own) the
  env is LocoManip-Marl-Direct-v0 and both run: the legs scorecard adds a "Coupled" section (the wrists' errors while
  walking), the arms scorecard scores the goals standing and while the legs walk. The env is the main policy's run;
  the two runs must agree on the IMU model, rates, gait period and delay, and each agent on its observation terms,
  or the eval stops and says what differs (`isaac/env.py`, `mujoco/policy.py`).
- **Conditions** are pinned in `common/conditions.py` (the legs: 84 envs, 6 segments each, seed 7, nominal motors,
  mean actions) and stamped into every json; plots join only points taken under the same ones.
- **Isaac vs MuJoCo**: the same program and metrics; MuJoCo runs the export with the deploy's settings
  (`--plant isaac` training's joint dynamics, `robocasa` GOLEM's sim's).

A new metric: compute it in `common/legs_metrics.py` / `arms_scorecard.py`, add its `Row` in the scorecard's
`rows()`, and its key to `PLOT_KEYS` to plot it. Older jsons without it show it as missing.

## Plots

```bash
python -m evals.isaac.legs.get_plot --run LegsDeploy200k_s42 [--prior 4] [--with <run>]
```

The run's scorecards against the newest prior runs of the same task (any trainer: PPO or SuperMimic), by env steps
(`plot_steps.png`) and wall-clock hours from TensorBoard (`plot_hours.png`), the reference dashed and the target
shaded; and the runs' training curves the same two ways (`plot_training_*.png`). Seeds of one run are one series
with their range as a band. Points scored under other conditions (the history imported from `logs/candidates`,
taken with the arms held by IK) are hollow. Plots never score anything.

## Videos

```bash
python -m evals.mujoco.get_video --legs <ckpt> [--arms <ckpt>]      # or --arms <ckpt> alone: pinned
python -m evals.isaac.get_video --legs <ckpt> [--arms <ckpt>|same]
```

THE command program (`common/program.py`, also what the GOLEM run sends): legs active: hung on the hoist with the
feet off the ground, lowered onto its feet, released, then forward, backward, side step left and right, angled both
ways, turn in place both ways, walk and turn, each followed by a stop. Arms active: a goal every 5 s, drawn as a frame
at the goal and one at the wrist. MuJoCo: a third-person and a side view, arrows and arcs for the commanded (green) and
actual (orange) velocity, the rope. Writes `video_<N>.mp4` and `video_<N>_small.mp4`.

## The deploy gate

```bash
python -m evals.deploy.gate --checkpoint <run>/checkpoints/agent_<N>.pt [--only ...] [--skip ...]
```

Runs and judges (thresholds in `deploy/gate.yaml`): the metric code; parity of lmg_deploy's controller with the
training env (Isaac); the MuJoCo stress grid, 180 s stands, a small command after 60 s, the pre-pose engage grid;
the robot's bags through lmg_deploy's controller; and `deploy/golem/run.sh`: the export under lmg_deploy in GOLEM's
RoboCasa sim with the full bringup, THE program on `/cmd_vel`, scored from the sim's own recording. It uses `~/GOLEM`
as it is checked out (the report records the commit): an export the checked-out lmg_deploy can't run (e.g. the
lowstate IMU on an older lmg_deploy) fails there. A check that cannot run is NOT RUN, which fails the gate.
`scripts/train_candidate.sh` runs the gate on each candidate (`GOLEM_RUN=1` adds the GOLEM run).
