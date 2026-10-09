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
- `mujoco_evals/`: the MuJoCo transfer. The env's policy loop ported to MuJoCo, the reach evaluation's MuJoCo trials,
  the Isaac-MuJoCo parity check, and the Isaac side of GOLEM's MuJoCo walking test.
- `paper/reach/`: the reach evaluation (Isaac and MuJoCo on one fixed goal grid, then the paper's tables, figures,
  numbers and result page); see its README.
- `logs/`: everything the code writes. Training runs go to `logs/skrl/locomanip_marl/`, the reach evaluation to
  `logs/reach/`. Published runs and the reach outputs are tracked.

## Setup

```bash
conda activate env_isaaclab
python -m pip install -e source/locomanipulation_game
```

## The MARL task

```bash
# train (3800 updates = 91.2k steps); --checkpoint resumes, with the estimator and curriculum saved beside it
python scripts/skrl/train.py --headless --max_iterations 3800 [--name <run>] [--checkpoint <run>/checkpoints/agent_<N>.pt]
python scripts/skrl/play.py --checkpoint <run>/checkpoints/agent_<N>.pt
# Isaac evaluation: walking and turning speeds, reach rate, end-effector error, squat depth -> <run>/eval_<N>/eval.md
python scripts/skrl/eval_final.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless

# deployment: networks + export.yaml, arm goals for the game commander, and parity with GOLEM's controller
python scripts/skrl/export_marl.py --checkpoint <agent.pt> --out <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_golem
python scripts/skrl/export_arm_goals.py --checkpoint <agent.pt> --levels 4:10,7:10,10:10 --out <same dir> --headless
python scripts/skrl/deploy_parity.py --checkpoint <agent.pt> --export <same dir> --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy --headless

```

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
