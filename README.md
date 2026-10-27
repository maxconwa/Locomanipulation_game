# Locomanipulation game

An Isaac Lab external project for the Unitree H1-2 with Magpie grippers: one robot, two MAPPO agents (skrl). The
legs walk on a velocity command the way Unitree's sport mode does; the arms track a wrist pose each. Policies deploy
to the real robot through GOLEM's `lmg_deploy` package.

The recipe copies sport mode, as measured in its bags (`docs/experiments/sportmode_ablations.md`): its PD gains on
every joint (with action scales matched to them), its stance (knee 0.70 rad), its command ranges (vx 0.55, vy 0.54
m/s, wz 0.97 rad/s), its 0.5 s command low-pass and 0.70 s stride, stepping on any command. The policy runs at
100 Hz over 500 Hz physics with an explicit PD and 0-20 ms of action delay; the actors are LSTMs with a capped cell
state; episodes have drawn lengths (uniform, 90-300 s); the arm goals start once training's own sport-mode tracking
metrics pass. These are the defaults: `train.py` with no overrides trains the recipe.

## Layout

- `source/locomanipulation_game/locomanipulation_game/`
  - `assets/h1_2.py`: the robot: names, the sport stance, the PD actuators and their matched action scales.
  - `data/arm_target_tables_v3.pt`: the arm goals' tables (Git LFS).
  - `tasks/locomanip_marl/`: the task, `LocoManip-Marl-Direct-v0`.
    - `locomanip_marl_env_cfg.py`: everything configurable; `scene_cfg.py`: the scene.
    - `locomanip_marl_env.py`: a DirectMARLEnv that runs Isaac Lab's managers, the warm start and the arm goals'
      odometry.
    - `mdp/`: commands (velocity with the sport-mode metrics; arm goals with their curricula), rewards, actions,
      observations, curricula, events, terminations.
    - `recurrent.py`: the LSTM actors and recurrent MAPPO; `golem_safety.py`: GOLEM's e-stops;
      `run_files.py`: a run's saved state and overrides.
- `scripts/skrl/`: `train.py`, `play.py`, `eval_scorecard.py` (with `scorecard_metrics.py` and its check against the
  sport-mode bags, `test_scorecard_sportmode.py`), `export_marl.py`, `deploy_parity.py`.
- `scripts/train_candidate.sh`: train seeds, then score, export and check each one.
- `agent_scripts/`: the CPU checks of an export: `hw_stress_mujoco.py` (MuJoCo under measured hardware effects),
  `hw_bag_replay.py` (the deploy controller on recorded robot bags), and `hw_deploy_template.yaml`.
- `mujoco_evals/`: the policy loop in MuJoCo (`mujoco_sim.py`) and the run settings the tools share (`common.py`).
- `logs/`: everything the code writes (untracked). Runs go to `logs/skrl/locomanip_marl/<time>_<name>/`.

## Setup

See `environment.yml`: the conda env, Isaac Lab from source with skrl, then

```bash
git submodule update --init --recursive   # third_party/CL_Assets: the robot's USD, URDF and MJCF
git lfs pull                              # the arm-goal tables
conda activate env_isaaclab
pip install -e source/locomanipulation_game
```

The CPU checks run in a second python with mujoco, torch, numpy, pyyaml, pin and rosbags, and need GOLEM
(`~/GOLEM`, for `lmg_deploy`).

## Train, score, export, check

```bash
# one run (600k policy steps = 12500 updates of 48, about 21 h on an RTX 4090)
python scripts/skrl/train.py --headless --max_iterations 12500 --name <name> --seed 42
# or the whole pipeline for seeds 42 and 43: train, scorecard, export, deploy parity, MuJoCo stress, bag replays
bash scripts/train_candidate.sh <name> 42 43        # SMOKE=30 for a 30-update smoke test

# the sport-mode scorecard and the wrist phase (motors nominal) -> <run>/checkpoints/scorecard_<N>.{json,md}
python scripts/skrl/eval_scorecard.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless \
    'env.events.actuator_scales.params.gain_range=[1.0,1.0]' 'env.events.actuator_scales.params.strength_range=[1.0,1.0]'
# export for lmg_deploy: legs.pt, arms.pt, <name>.yaml (the template), export.yaml (what the run trained with)
python scripts/skrl/export_marl.py --checkpoint <run>/checkpoints/agent_<N>.pt --out ~/GOLEM/core_ws/src/lmg_deploy/policies/<name>
# the deploy controller against the env, step by step
python scripts/skrl/deploy_parity.py --checkpoint <run>/checkpoints/agent_<N>.pt --export <export dir> --headless
# CPU: MuJoCo stress tests and the replay of a robot bag
python agent_scripts/hw_stress_mujoco.py --export <export dir> --label <name>
python agent_scripts/hw_bag_replay.py --bag <bag> --deployed mid360v2 \
    --policy mid360v2=~/GOLEM/core_ws/src/lmg_deploy/policies/mid360v2 --policy <name>=<export dir> --sim_seeds 4
python scripts/skrl/play.py --checkpoint <run>/checkpoints/agent_<N>.pt
```

The scorecard's targets are sport mode's: turn sense >= 0.99, speed ratio >= 0.90, turn ratio >= 0.90, cos >= 0.95,
angular error <= 0.05 rad/s, linear error <= 0.15 m/s.

The evaluation scripts rebuild the env a run trained in from the current defaults plus its saved overrides
(`<run>/params/hydra_overrides.txt`); an override whose key the current config no longer has is dropped (its value is
now the default) and listed. Runs from before the 2026-10-25 cleanup on other recipes need the `pre-cleanup` tag.

While training, the robot walks unpushed on flat ground until the warm-start gate passes (falls per minute and the
sport-mode metrics, `warm_start_*`); then the pushes, the terrain rows and the arm goals start. `<run>/warm_start_gate.yaml`
changes its thresholds without a restart; TensorBoard shows training's sport-mode metrics under `Metrics/sport/`.

## Deploy

Copy or export into `~/GOLEM/core_ws/src/lmg_deploy/policies/<name>/`, build, and launch with `policy:=<name>` (GOLEM's
`lmg_deploy` README). The node runs at the policy's `control_dt` and reads the gains, stance, action scales, command
filter and ranges from `export.yaml`.

## Gotchas

- One Isaac process at a time on the GPU; a small job beside training is fine if memory allows. "fail to launch kernel"
  in a log means a GPU fault: rerun.
- Isaac ignores SIGTERM: `timeout -s KILL`, then check `nvidia-smi`.
- Fixed-seed rollouts are deterministic.
