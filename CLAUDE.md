# Working agreement

## Never edit this repo

Do not use Edit/Write/NotebookEdit on anything under `/home/max/ALMI-Open`.
No new files, no fixes in passing, no cleanups along the way.

# The codebase

`ALMI_RL/` is a legged_gym / rsl_rl fork for the Unitree H1-2 humanoid.

**Env lifecycle** — `legged_gym/envs/base/legged_robot.py` (840 lines) is the
spine: `step` → `post_physics_step` → `check_termination` / `compute_reward` /
`compute_observations`, with `reset_idx` for per-env resets. Per-env physics
properties are set once at creation in `_process_rigid_shape_props` /
`_process_dof_props` / `_process_rigid_body_props` (:249, :274, :307) — that's
why some randomization can't be resampled per episode.

**Rewards** — `_prepare_reward_function` (:568) looks up `_reward_<name>` for
every non-zero key in `cfg.rewards.scales`, and multiplies non-`termination`
scales by `dt` (:578). A new term is two things: the method and the cfg key.

**Config** — plain nested classes: `base_config.py` → `legged_robot_config.py`
→ per-robot `*_config.py`, flattened by `class_to_dict` in `_parse_cfg` (:734).

**Tasks** — `legged_gym/envs/__init__.py` maps `--task=` names through
`task_registry.register(name, EnvClass, Cfg, CfgPPO)`. Footgun: all three
`h1_2*` variants export identically-named `H1_2_WholeBody` / `H1_2_WholeBodyCfg`
and that file rebinds each name in turn. The class name never tells you which
variant you're looking at — check the module path.

**RL side** — `rsl_rl/rsl_rl/`: `runners/on_policy_runner*.py` (rollout loop),
`algorithms/ppo.py` (update), `storage/rollout_storage*.py`. The lower / upper /
whole-body triplicates mirror the curriculum stages in `ALMI_RL/README.md`.

# This repo (`Locomanipulation_game`)

An Isaac Lab external project (a DirectMARLEnv trained with skrl MAPPO) — separate
from `ALMI_RL`.

- **Activate the env first:** `conda activate env_isaaclab`
- **The tasks live in**
  `source/locomanipulation_game/locomanipulation_game/tasks/locomanip_marl/`:
  `LocoManip-LegsOnly-v0` and `LocoManip-ArmsOnly-v0` (one agent's task each), and
  `LocoManip-Marl-Direct-v0` (both at once, each agent on its own task; `marl_direct_env_cfg.py`).
  `locomanip_marl_env_cfg.py` is their shared base (the sport-mode recipe, README).
- **Train:** `python scripts/skrl/train.py --headless [--task <id>]`; runs go to
  `logs/skrl/locomanip_marl/`. The whole pipeline (train, scorecard, export,
  checks): `scripts/train_candidate.sh`.
- **Evaluate:** everything is in `evals/` (`evals/README.md`), run as
  `python -m evals.<...>` from the repo root: `evals/isaac/{legs,arms}/` and
  `evals/mujoco/{legs,arms}/` (`get_scorecard`, `get_plot`), `get_video` in
  `evals/isaac` and `evals/mujoco`, and `evals/deploy/gate.py` (GOLEM's sim,
  parity, bag replay, MuJoCo stress tests: PASS/FAIL). Isaac ones read the
  run's task from `<run>/params/task.txt`; the MuJoCo and deploy ones run in a
  separate python with mujoco. Results go to `logs/evals/<run>/`.
- Older recipes (the paper's reach evaluation, the whole-body baseline, the
  learned pelvis estimator) are at the `pre-cleanup` tag.
- **`pretraining/`**: a separate package (`pip install -e pretraining`), stage-1 motion-tracking pretraining of
  a legs and an upper-body agent on retargeted AMASS clips with SuperMimic's world-model method, ported from
  `h12_train` (branch `supermimic`); gym id `LocoManip-Pretrain-Track-v0`. No clips or policies in git (licence).
- Train it: `python -m pretraining.scripts.train --config pretraining/configs/base.yaml --run-name <name> --num-envs 2048`
  (base.yaml's 5,120 envs do not fit a 16 GB GPU; runs in `data/pretraining/runs/`); also
  `python -m pretraining.scripts.crosscheck`, `python -m pretraining.scripts.play`. Everything is in
  `pretraining/README.md`.
- **`LocoManip-Marl-Torso-v0`** (`marl_torso_env_cfg.py`): Marl-Direct with the arms agent also driving the torso.
  **`LocoManip-Marl-Pulse-v0`** (`marl_pulse_env_cfg.py`, `mdp/pulse_actions.py`): the same task with each agent
  acting in the latent of its frozen stage-1 decoder (PULSE), on main's plant (100 Hz), so the stage-1 run must be a
  main-plant one; fit its prior first with `python -m pretraining.scripts.fit_prior <stage-1 run>`
  (`pretraining/README.md`). `python -m pretraining.scripts.train_commands` trains the same latent policy through the
  world model. The scorecards and `get_video` take latent policies (`--latent_run`, `evals/README.md`); exports and
  the deploy gate do not.
- When adding a new task name, keep `LocoManip-` in it: `scripts/list_envs.py:54`
  only lists ids containing that substring, so a task registered under a
  different name silently won't show up.
