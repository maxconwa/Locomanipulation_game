# Reach evaluation of the reward-coupled H1-2 policies

Scores any `LocoManip-Marl-Direct-v0` checkpoint on one fixed grid of two-wrist goals, in Isaac Lab and in MuJoCo,
and rebuilds every table, figure, LaTeX number and the result page from the trials. Built against
`marl-direct` 72171c7 ("Paper task").

## Evaluating a run (the coworker's path)

The run directory needs `checkpoints/agent_<N>.pt`, `estimator/estimator_<N>.pt` (or `estimator_latest.pt`) and
`params/env.yaml`. λ is read from `params/env.yaml`; the label you give names the condition everywhere.

```bash
conda activate env_isaaclab51            # Isaac Sim 5.1, Isaac Lab main 2.3.2, skrl 2.1, torch 2.7
make -C paper/reach eval    RUN=logs/skrl/locomanip_marl/<run> LABEL=lambda0.5   # ~4 min: full, legs blind, arms IK
make -C paper/reach sim2sim RUN=logs/skrl/locomanip_marl/<run> LABEL=lambda0.5   # ~15 min on 8 cores
make -C paper/reach paper                                                         # tables, figures, macros, page
```

`make eval` takes the newest checkpoint; pass `CKPT=<path>` for another. Add the run to `conditions.yaml` to give it a
legend name and a "pending" row on the page before it is evaluated (not needed for the analysis: every
`results/<label>/` directory is picked up). `make smoke` checks the pipeline on a zero-action checkpoint without any run.

On another machine set `CL_ASSETS_DIR` to a CL_Assets checkout at cf87bfe with its LFS files pulled (the task's
`third_party/CL_Assets` submodule), `PY_ISAAC` / `PY` to the Isaac Lab python, and `LOCOMANIP_CODE` to the
checkout whose `source/locomanipulation_game` is installed.

## What runs

| Tool | What it does | Output |
|---|---|---|
| `tools/build_tables.py`, `tools/make_goals.py` | The task's standing wrist-pose tables (seed 0), then 96 base pose pairs × 11 lowerings (0–1.0 m) × 3 outward extensions (0, 0.1, 0.2 m) | `goals/eval_goals_v1.csv` (3168 goals) |
| `tools/eval_isaac.py` | One trial per env: 2 s standing, one grid goal placed in the standing frame and held 4 s; deterministic actors loaded with torch alone; curriculum frozen; learned odometry as deployed. Variants `full`, `legs_blind` (legs see navigation inputs), `arms_ik` (residual held at 0) | `results/<label>/isaac*/trials.csv`, `traces.npz`, `meta.json` |
| `tools/eval_mujoco.py` | The same trials in MuJoCo through `reachlib/mujoco_sim.py`, a port of the env's observations, actions, PD and odometry; RoboCasa joint dynamics or the training plant's (`--physics isaac`) | `results/<label>/mujoco*/` |
| `tools/ik_baselines.py` | GOLEM's arm IK (`h12_ros2_controller` `IKSolver.solve_ik_reduced`) on a standing robot | `results/golem_ik/kinematic/` |
| `tools/parity_check.py` | Same state in both simulators: observations term by term and the arm IK targets | `results/_parity/report.json` |
| `tools/analyze.py` | Success with Wilson intervals, errors, posture and balance by lowering, by target height and per workspace cell; paired differences; operational floor | `results/summary/*`, `tex/results_macros.tex` |
| `tools/figures.py`, `tools/render_robot.py` | Paper figures (PDF, PNG, plotted data as CSV) and orthographic H1-2 renders | `figures/` |
| `tools/build_page.py` | The result page, all numbers from `results/summary` | `../20261008-reach_cooperation.html` |

## Numbers in the paper

`tex/results_macros.tex` defines `\reach{<condition>}{<quantity>}`; `\input` it in the preamble. A quantity not yet
computed prints a bold `??`. Conditions are `<label>/<sim>` (`l1/isaac`, `l1/isaac_legs_blind`,
`l1/mujoco`, `golem_ik/kinematic`); quantities:

| Quantity | Meaning |
|---|---|
| `n`, `success`, `falls`, `floor` | trials, strict success (%), falls (%), operational floor (cm above ground) over the whole grid |
| `success_d0.3` | strict success (%) at lowering 0.3 m, all extensions |
| `success_e0_d0.3` | the same, extension 0 only (the standing-table goals lowered) |
| `drop_d0.3`, `height_d0.3` | median pelvis drop and height (cm) over the goal's last second |
| `err_d0.3` | median wrist error of reached goals (cm) |
| `com_d0.3`, `jitter_d0.3`, `falls_d0.3` | min CoM margin (cm), wrist jitter (mm), falls (%) |
| `goals`: `n`, `bases`, `depths`, `depthmax`, `exts`, `standingfloor` | the grid |

## Definitions

Strict success: both wrists within 5 cm and 0.35 rad of their world-fixed targets for 1 s without a break inside the
4 s goal (the task's reach test). Lower target height: height above the ground of the lower wrist target. CoM / DCM
margin: signed distance of the centre of mass's ground projection (or ξ = c + ċ/ω, ω = √(g/z_c)) to the support
polygon of the feet carrying more than 20 N, positive inside. Operational floor: lowest 0.1 m height bin from which
every bin above has ≥ 80% strict success. The page's "Terms and metrics" section has the full list.

## Workstation notes

- GOLEM's IK workers each map about 1.5 GB (most of it cold), so `make baselines` runs 4 workers.
- Isaac ignores SIGTERM; kill stuck runs with `timeout -s KILL`.
- `eval_isaac.py --zero_gravity` exists only for the self-test: the task's IK indexes a floating base's Jacobian,
  so pinning the root (`fix_root_link`) silently breaks the arms' IK step.
