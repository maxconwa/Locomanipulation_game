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
- **Tasks live in**
  `source/locomanipulation_game/locomanipulation_game/tasks/locomanip_marl/`
  (`LocoManip-Marl-Direct-v0`, and the one-agent `LocoManip-WholeBody-Direct-v0`)
- **Train:** `python scripts/skrl/train.py --headless [--task <TASK>]`; runs go to
  `logs/skrl/locomanip_marl/`
- **MuJoCo transfer** lives in `mujoco_evals/`; the reach evaluation in
  `paper/reach/` (`make -C paper/reach all`) writes to `logs/reach/`.
- When adding a new task name, keep `LocoManip-` in it: `scripts/list_envs.py:54`
  only lists ids containing that substring, so a task registered under a
  different name silently won't show up.
