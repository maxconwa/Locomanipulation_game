# Locomanipulation game

An Isaac Lab external project for the Unitree H1-2 with Magpie grippers.

**This branch (`marl-direct`):** one robot and two MAPPO agents. The legs track a velocity command. The arms each
track a wrist pose, and they crouch for low goals by squatting. The task targets deployment through GOLEM's safety
layer into RoboCasa.

- `source/locomanipulation_game/locomanipulation_game/tasks/direct/locomanip_marl/`: the task, `LocoManip-Marl-Direct-v0`.
  - `locomanip_marl_env_cfg.py`: everything configurable.
  - `locomanip_marl_env.py`: a DirectMARLEnv that runs Isaac Lab's managers.
  - `mdp/commands.py`: navigation or arm goals, and the reach curriculum.
  - `odometry.py`: the pelvis-motion estimator.
  - `golem_safety.py`: GOLEM's e-stops.
- `tasks/manager_based/`: the legs-versus-adversary game (rsl_rl, `scripts/rsl_rl/`). The MARL task reuses its scene,
  legs reward, events and mdp terms.

## Setup

```bash
conda activate env_isaaclab
python -m pip install -e source/locomanipulation_game
```

## The MARL task

```bash
# train (3800 updates = 91.2k steps); --checkpoint resumes, with the estimator and curriculum saved beside it
python scripts/skrl/train.py --headless --max_iterations 3800 [--checkpoint <run>/checkpoints/agent_<N>.pt]
python scripts/skrl/play.py --checkpoint <run>/checkpoints/agent_<N>.pt

# deployment: networks + export.yaml, arm goals for the game commander, and parity with GOLEM's controller
python scripts/skrl/export_marl.py --checkpoint <agent.pt> --out <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_golem
python scripts/skrl/export_arm_goals.py --checkpoint <agent.pt> --levels 4:10,7:10,10:10 --out <same dir> --headless
python scripts/skrl/deploy_parity.py --checkpoint <agent.pt> --export <same dir> --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy --headless

# GOLEM's walking sim-to-sim test, Isaac side (scored by GOLEM's tests/locomanipulation_game/sim2sim_walk.py --score)
python scripts/skrl/eval_sim2sim_walk.py --checkpoint <agent.pt> --out <dir> --headless
```

Runs are written to `logs/skrl/locomanip_marl/<time>_mappo_torch/`. Each run directory holds:
- the skrl checkpoints;
- `estimator/` (the pelvis estimator and curriculum state);
- `arm_target_tables_v2.pt`: built on a run's first start, which takes a few minutes, then reused by every script
  pointed at that run.

## The game

Training rounds Legs-R0-v0 → Upper-Adv-Ri-v0 → Legs-Ri-v0, each against the frozen policy the previous round produced,
in one process (`scripts/rsl_rl/game.py`; see its docstring). One round on its own:

```bash
python scripts/rsl_rl/train.py --task=Legs-R0-v0 --headless
```
