# Handoff: Mid360v2 real-robot failure study (as of 2026-10-22 15:40 MDT)

Repo `/home/max/Locomanipulation_game`, branch `marl-direct`. Activate `conda activate env_isaaclab` for Isaac;
`conda activate base` has mujoco+torch+yaml for the MuJoCo tools. Read `CLAUDE.md` first (never edit
`/home/max/ALMI-Open`; GOLEM sims never on ROS_DOMAIN_ID 0).

## The task

Mid360v2 (agent_196800, training commit f541d1c) collapsed twice on the real H1-2 on 2026-10-20: a 2.5-3.5 Hz leg
oscillation, then an operator e-stop. A GOLEM-side failure report (pasted by the user) blamed the head MID-360's
20 Hz gyro resonance and a 12 ms LowState dq lag. The user asked, in order:

1. Propose a theory and run ablations to test the report's theories. Done.
2. Write a report. Done: Claude Doc "Mid360v2 Failure Ablation Report", https://claude.ai/artifact/LNRZfxVdPJzuWM8CqAuqZj
3. Train a deployable candidate: 60 s episodes + motor randomization, 200k steps. `LongAct200k` is done.
4. Compare it to an MLP version (no LSTM). `LongActMLP200k` is training.
5. Do not normalize h (cancelled); study the LSTM cell state c instead, and run a third experiment capping c with tanh.
   The c study is done (below); `LongActCap200k` is queued.

## Code on GitHub (2026-10-23)

The study's code is on branch `hw-ablations` (origin), branched from `marl-direct` beaaadc.
- `agent_scripts/hw_ablations.sh` no longer depends on this machine. It reads the deploy template from
  `agent_scripts/hw_deploy_template.yaml` (Mid360v2's GOLEM deploy yaml).
- The conda setup is overridable: `CONDA_SH`, `ISAAC_ENV` (default `env_isaaclab`) and `MUJOCO_ENV` (default `base`,
  any env with mujoco + torch + pyyaml).
- The tools in `logs/hw_ablations/tools/` and this file are force-added (`logs/` is gitignored); run them from the
  repo root.

## Update 2026-10-23: all three 200k runs are done and reported

The report is the Claude Doc "LSTM vs MLP vs Capped-Cell Report", https://claude.ai/artifact/T8MUT9SLzNPZEkySH2t6Yv
- **Capped c (`LongActCap200k`) is best:**
  - 0 falls in 100 stress cells, and every |c| ≤ 1;
  - arm goals from 54k steps; at 200k it reaches 0.71 goals per episode with 21% falls per 60 s episode.
- **The MLP never started arm goals** and jitters most while standing.
- **Next:** resume the capped run for 100-200k more steps. The new tool is `logs/hw_ablations/tools/smoothness.py`.
- The pause section below is history: the processes were resumed and finished.

## PAUSED: resume first (2026-10-22 15:39 MDT)

The user asked to pause everything for the handoff. Both process groups were frozen with `SIGSTOP`, not killed;
nothing was lost. GPU memory is still held by the frozen training.
- `LongActMLP200k` is frozen at 101,836 of 200,016 steps (51%), about 6 h 40 m left.
- `LongActCap200k` is still queued behind it.

Resume both:
```bash
kill -CONT -- -3138703   # launcher + LongActMLP200k training (process group 3138703; python PID 3742413)
kill -CONT -- -4016839   # queue_cap.sh (waits for "LongActMLP200k: trained" in status.txt, then starts LongActCap200k)
ps -eo pid,pgid,stat,args | grep -E 'hw_ablation[s].sh|queue_c[a]p|skrl/trai[n].py'   # STAT should no longer be T
```
After resuming, the ETAs below shift by however long the pause lasted.

Notes:
- The training's wall-clock bar jumps by the pause length, but its step count continues where it stopped.
- If the training dies on resume (e.g. a CUDA error after a long stop), restart it from its last checkpoint with
  `scripts/skrl/train.py --checkpoint <run>/checkpoints/agent_<N>.pt`. Push levels aren't in checkpoints, so they
  re-climb from 0.
- Or relaunch it from scratch: `ITER=8334 PREFIX= bash agent_scripts/hw_ablations.sh LongActMLP200k`, after killing
  both groups. Note that the queue keys on the status line `LongActMLP200k: trained`.
- No Claude watchers are running any more.

## Running right now (before the pause)

| Run | What | State | ETA (MDT) |
|---|---|---|---|
| `LongActMLP200k` | the candidate's config with MLP actors (`~agent.models.policy.rnn`) | training, about 43% at 14:40 | trained about 21:50 10-22, stress tests about 22:05 |
| `LongActCap200k` | the candidate's config + `agent.models.policy.rnn.cell_cap=1.0` (c <- tanh(c) every step) | queued (`logs/hw_ablations/queue_cap.sh`, PID 4016839) | starts about 21:50, trained about 11:30 10-23, stress tests about 11:45 |

- Launcher PID 3138703 (`bash agent_scripts/hw_ablations.sh LongAct200k LongActMLP200k`) trains the MLP. The queue
  then starts `ITER=8334 PREFIX= bash agent_scripts/hw_ablations.sh LongActCap200k`.
- Each run is trained, exported to `logs/hw_ablations/<run>/export` and stress-tested in MuJoCo
  (`logs/hw_stress_mujoco/<run>{,_long}`) automatically.
- Progress: `tail logs/hw_ablations/status.txt`; training bar: `tr '\r' '\n' < logs/hw_ablations/<run>.train.log | grep it/s | tail -1`.
- Run dirs: `logs/skrl/locomanip_marl/<time>_<run>`.

## Hazards

- **Do not edit `agent_scripts/hw_ablations.sh` while any instance of it runs** (until about 11:45 10-23).
  - Bash reads a script by byte offset as it goes, so an edit shifts what a running launcher reads next.
  - The file was realigned on 10-22 so PID 3138703 reads `wait` cleanly after its loop.
  - That is why its header comments are truncated. Restore them from `logs/hw_ablations/reference/hw_ablations.sh.bak`
    once no launcher runs, keeping the `[LongActCap200k]` entry.
  - The same applies to `queue_cap.sh`.
- **One Isaac sim at a time on the 16 GB GPU.** While a 4096-env run trains, a second Isaac process dies (CUDA OOM or
  illegal memory access), even with `--device cpu`. The MuJoCo tools are CPU-only and safe to run alongside.
- **`pkill -f '<pattern>'` killed my own shell twice** (exit 144), because the pattern was in my own command line.
  Kill by PID: `ps -eo pid,args | grep -E 'hw_ablation[s].sh'`.
- **The launcher's `TEMPLATE` default points at my session scratchpad**
  (`/tmp/claude-1000/.../scratchpad/mid360v2/mid360v2.yaml`). The queued Cap run's export copies it as `deploy.yaml`.
  A durable copy is in `logs/hw_ablations/reference/mid360v2/mid360v2.yaml`. If the scratchpad is gone before
  about 11:30 10-23, the Cap export step fails; re-export by hand with the durable template (command below).

## Findings so far (details and charts in the report doc)

**The report's theories, against the deployed policies.** These are stress tests of the deployed TorchScript in Isaac
(`agent_scripts/hw_stress_test.py`) and MuJoCo (`agent_scripts/hw_stress_mujoco.py`).
- No simulator reproduces the collapse.
- The gyro resonance (calibrated to the bags) and the dq lag change nothing.
- Weaker motors or extra torso mass reproduce bag 2's ankle-pitch target sliding to its floor, but not a fall.
- The lopsided stance shows in MuJoCo, not in Isaac, and Explicit500 has it too.

**48k-step ablations (one seed each, none reached arm goals).** Runs: base, actuator, long, standing, sensing, all.
- `base` falls just after 20 s: at 20 s its arm actions double. Zeroing the arms' LSTM every 10 s cut those falls
  from 88% to 31%.
- The arms train with `time_limit_bootstrap: False`.
- `long` (60 s episodes) and `actuator` (gains ±20%, strength ±15%) were the most robust.
- `sensing` (torso IMU + dq lag) was the least robust.
- `standing` evened the stance but crouches deepest under weak motors.
- `all` was robust but walked worst.

**`LongAct200k`, the candidate.** Run dir `logs/skrl/locomanip_marl/2026-10-21_19-28-34_LongAct200k`.
- The warm-start fall gate counts per episode, so for 60 s episodes it was raised to 6%
  (`env.warm_start_max_fall_share=0.06`). The first try at 2% never opened; it is kept as `*-aborted73k`.
- Stress tests: 0% falls everywhere, gantry starts included. The legs LSTM state stays flat (|c| about 1.7), with no
  20 s switch. Knee rise at 75% motor strength is +0.17 rad, against Mid360v2's +0.63.
- But it is undertrained on the task. Arm goals only opened at 84k steps, then training falls spiked to 99%.
- At 200k: 32% of 60 s episodes end in a fall, against Mid360v2's 0.064 falls per env-minute. It reaches 0.32 goals
  per episode and is still improving.
- **Open question for the user:** resume it for about 100k more steps? Not decided.

**The cell state c** (`tools/cell_study.py`). "Unbounded growth" was wrong: a few units with forget gate f ≈ 0.998
are slow leaky integrators. They settle at c* ≈ i·g/(1−f): Mid360v2's legs unit 9 at about −420, its arms units 6
and 43 at about ±200.
- Unit 9 is still climbing at 20 s (the training horizon) and needs about 30 s to settle.
- tanh(c) is saturated, so h barely changes. But the unit is a **latch**: under the strongest opposite drive it takes
  about 6.6 s to change sign.
- Explicit500's and the candidate's leg latches flip in about 0.6 s; the candidate's arms in about 2.3 s.
- The tanh cap (`LongActCap200k`) bounds every unit to ±1, so it can flip in a few steps. That is the hypothesis
  the third experiment tests.

## Code changes (uncommitted; all new options are off by default)

Mine:
- `.../locomanip_marl/mdp/dynamics.py`: `randomize_actuator_scales`, the reset event `events.actuator_scales`.
- `.../locomanip_marl_env.py` + `mdp/observations.py`: `joint_vel_obs_lag_ms` with `mdp.joint_vel_rel_lagged`, and
  `lagged_joint_vel()`.
- `mdp/commands.py`: `small_command_prob` and `small_command_norm` (sub-threshold commands for standing envs).
- `mdp/rewards.py` + env cfg: `stand_load_imbalance`, the term `rewards.legs.stand_load_balance`, weight 0.
- `recurrent.py` + `agents/skrl_mappo_cfg.yaml` + `scripts/skrl/export_marl.py`:
  - `rnn.normalize_hidden` (unit-length h; the run was cancelled);
  - `rnn.cell_cap` (c <- s·tanh(c/s)).
  - The checkpoints carry `normalize_hidden_flag` / `cell_cap_value` buffers, which the export reads. The exported
    TorchScript matches the policy exactly (CPU test, zero difference), so GOLEM's deploy needs no change.
- `agent_scripts/hw_stress_test.py`, `hw_stress_mujoco.py` (gantry hang start, periodic LSTM resets, LSTM |c| logging),
  `hw_ablations.sh`.

Not mine (the user's own edits from before this task): `README.md`, `mujoco_evals/{__init__,eval_sim2sim_walk,human_control}.py`,
untracked `mujoco_evals/{walk,eval_walk_mujoco}.py`, `AGENTS.md`, `paper/reach/results/`.

## Tools and data

- `logs/hw_ablations/reference/`:
  - `mid360v2/`, `explicit500/`: the deployed exports, from GOLEM git 4cbdb25 and 5928261 via
    `git show <c>:<path> | git lfs smudge`;
  - `base_overrides.txt`: the reconstructed Mid360v2 launch overrides; Mid360v2's real `params/env.yaml` is on FatMan,
    whose SSH host key isn't known here.
- `logs/hw_ablations/tools/` (run from the repo root with `conda activate base`):
  - `compare.py <run> ...` (expects `logs/hw_stress_mujoco/HWAbl_<run>`);
  - `arm_drift.py <export dir> 0` (arm/leg activity per 5 s over 60 s, the 20 s switch);
  - `cell_study.py name=<export dir> ...` (per-unit c, forget gates, latches).
- Stress results: `logs/hw_stress/` (Isaac), `logs/hw_stress_mujoco/<label>/summary.{md,json}`.

## Next steps

1. When `LongActMLP200k` is stress-tested (about 22:05 10-22), compare it to `LongAct200k`:
   - the `summary.md` files;
   - `arm_drift.py` on both exports;
   - task metrics from its TensorBoard (falls, `Metrics/arm_targets/goals_reached`, `nav_cos_vel_xy`, when the warm
     start ended: grep "Warm start over" in its train log).
   - **Watch for jitter.** Earlier MLP policies jittered on hardware, so compare action smoothness (the step-to-step
     action change), not just falls.
   - Note: `cell_study.py` only works on LSTM exports.
2. When `LongActCap200k` is stress-tested (about 11:45 10-23):
   - run the same comparison;
   - confirm with `cell_study.py` that every |c| ≤ 1;
   - check whether recovery after a gantry start or a disturbance is faster than the candidate's.
3. Write up the three-way comparison (LSTM / MLP / capped c) for the user. Optionally add it, and the c analysis,
   to the report doc via the Claude Docs connector (doc id 9cdb6431-bff8-4302-a5bd-a50d76d32e1a, tab 6b6054a7-5dbf,
   body node 174c7df9-8f69).
4. Ask the user: resume the best of the three for about 100k more steps so it learns the arm task? Only a fully
   trained one is deployable.
5. Manual re-export if needed:
   `python scripts/skrl/export_marl.py --checkpoint <run>/checkpoints/agent_196800.pt --out logs/hw_ablations/<run>/export && cp logs/hw_ablations/reference/mid360v2/mid360v2.yaml logs/hw_ablations/<run>/export/deploy.yaml`
