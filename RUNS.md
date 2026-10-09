# Runs A–R: what we learned

Branch `marl-direct`, 2026-10-05 to 10-07. One Unitree H1-2 is driven by two MAPPO agents (skrl). The **legs** track
a velocity command (vx, vy, yaw rate). The **arms** track a wrist pose each, and the legs crouch for low goals. The
deployment target is GOLEM: its `locomotion_game_deploy` controller and its `h12_safety_layer`, in RoboCasa (MuJoCo)
through ROS.

Today the task is `LocoManip-Marl-Direct-v0` (`source/.../tasks/locomanip_marl/`), configured for
[the final run](#the-final-run). The task ids named below (Flat-Curriculum, IK, IK2, IK3, Golem … Golem7) were folded into it
in commit c8ee31a; git history has each one.

Runs are in `logs/skrl/locomanip_marl/<dir>_mappo_torch/`, with logs in `logs/skrl/mappo_<name>.log`. Units:
- falls per env-minute;
- "spread" and "drop" are the two reach-curriculum levels, 0–10 each;
- `s:d` is an evaluation held at spread s, drop d;
- RoboCasa scores are e-stops and falls per minute of game.

## Before run A (runs 1–13)

Runs 1–13 (10-02 to 10-05) were on the terrain world: IPPO first, then MAPPO in a stand-and-reach task. These
lessons carried over:

- **skrl's log_std clamp** only acts in the forward pass. A parameter pushed past `max_log_std` gets no gradient, and
  the std stays pinned (run 3: 1.0 for 1077 updates). `train.py` clamps the parameter after every update. Per-agent
  bounds are in the yaml: legs std ≤ 0.6.
- **The pelvis-motion estimator** was fed the actors' domain-randomization noise and drifted 0.25 m per goal. It
  needs sensor-level noise, a 4-frame history and leg torques, plus a drift gate before arm goals use it.
- **Crouching emerges** when nothing holds the pelvis height during arm goals:
  - base_height and stand_still apply during navigation only;
  - the gait shaping that resists a crouch is scaled down during arm goals (lin_vel_z 0, ang_vel_xy 0.5, hip_pos 0.2);
  - goals are lowered by a drop curriculum.

  Run 12's legs then took 72–74% of a target drop (crouch slope 0.66). Lowering standing targets alone was not
  enough: the "lower half" of the table was still 0.9 m above the ground.
- **Raw actions past the ±10 clip** go unpunished: knees at −300 locked straight. `action_beyond_clip` charges them.
- **Stale IMU after a reset:** DirectMARLEnv resets without a kinematics update. The env refreshes the sensors
  (`_refresh_sensors_after_reset`).

## The runs

| Run | Dir | Change (commit) | Started from | Outcome |
|---|---|---|---|---|
| A | 10-05_19-19-03 | Flat world, fresh policies, walking gate (walk until navigation is tracked), rate-limited targets (fdee0a9) | fresh | Hung PhysX at 22.5k: the arms' value scaler ran away to NaN |
| B | 10-05_20-55-56 | Arms: no time-out bootstrap; user removed rate caps (603663e) | fresh | Arms' value scaler still drifted (mean −99, std 354). Stopped at ~8k |
| C | 10-05_21-07-56 | Arms critic on raw returns (34f0b9d) | fresh | Walked well (2% falls, 0.19 m/s error) but covered 69% of the path; 1.3% passed the gate. Stopped at 72k |
| D | 10-05_23-00-53 | 50/50 walking and arm goals from step 0, no gate (88a5af8) | fresh | Every episode fell within 1.5 s. Stopped at 13.8k |
| E | 10-05_23-26-01 | Legs alive 0.15 → 1.0 (064ad69) | fresh | Stopped after 5 min for the reward-floor rule |
| F | 10-05_23-28-57 | Legs reward floored at 0 in every mode (582ee65) | fresh | 360k. Walks (0.004 falls/min, 59% of path); arms stall at 12 cm, 15% reached, curriculum at 0, no crouch |
| G | 10-06_09-01-03 | IK-residual arms that observe their error; drop axis on closest approach (58cf51f) | F's legs, fresh arms | At 91.2k: 94–100% of front crouch goals reached, legs taking 78% of a target drop; wide standing reach at 8:0 only 36%, 1.18 falls/min. Stopped at 131k |
| H | 10-06_13-44-22 | IK2: goals in front, balanced tables, squat tables, goals held 4 s, split curriculum axes (a98c0af) | fresh | 91.2k. Falls 123 vs G's 597 per 2560 env-min; wrist error 3.6–4.2 cm. Commands drifted 3 → 10–15 cm within held goals |
| I | 10-06_18-28-04 | IK3: leg odometry, arm residual low-passed at 3 Hz (5424cca, 61e6482) | H 91.2k | 91.2k. Crouch at 4:10 94%, 8:5 74%, 8:10 67%; holding error 1.9 cm; best Isaac policy |
| J | 10-06_23-35-56 | Golem: GOLEM's e-stops end the episode, targets clipped like GOLEM (fa7a4a5) | I 91.2k | E-stop share of episode ends ~0.96, levels collapsed. Stopped at 17.9k |
| K | 10-07_00-12-02 | Golem2: targets bounded by PD torque at 85% of the effort limit (0759290) | I 91.2k | Trips fell to 1.9/robot-min. Every RoboCasa game hit an ankle e-stop. Stopped at ~20.5k |
| L | 10-07_00-56-49 | Golem3: passive damping U(0,10), armature U(0.01,0.12) (c534cd9) | K 19.2k | Its torque trips were a monitor artifact. Stopped at ~16k |
| M | 10-07_01-34-17 | Monitor fixed: motor torque from the current state (92f051a) | K 19.2k | 91.2k. Isaac: crouch at 4:10 74% reached (I 90%), walking path 32% (I 49%), 0.47 GOLEM trips per robot-min; RoboCasa 0.42 stops/min (2 of 8 full games) |
| N | 10-07_10-32-13 | Golem4: RoboCasa's damping U(8,12) / armature U(0.08,0.12), 0–20 ms delay, tracking std 0.25, wider ankle/knee margins (e86c5ea) | M 91.2k | Stopped at 13k to add the swing fix |
| O | 10-07_11-06-29 | Golem5: feet_swing_clearance on the gait clock (b7d7dde) | M 91.2k | 91.2k. Forward and backward walking transfer to MuJoCo; no turn in place; RoboCasa 0.86/min |
| P | 10-07_13-59-12 | Golem6: 20% pure turns, 10% pure sideways commands (1578f14) | O 91.2k | Stopped at 11.7k to add the mass fix |
| Q | 10-07_14-31-57 | Golem7: placeholder frames 1 kg → 0.01 kg (b79cd0a) | P 9.6k | 91.2k. Still no turn in place; sideways 0.1 of 0.25 m/s; crouch and reach kept |
| R | 10-07_17-36-19 | Velocity tracking on the gait-cycle mean velocity (54d935f) | Q 91.2k | Stopped at 27.4k for the final run. First policy to turn in place: +0.27 / −0.16 rad/s. See [Run R](#run-r) |

## What we learned

### Training stability (skrl MAPPO)

- **A value scaler can feed on itself.**
  - What happened: skrl's time-limit bootstrap adds γ·V(next) at time-outs, with V un-normalized through
    a RunningStandardScaler that is trained on those same returns and on the critic's own predictions. In run A the
    arms' scaler reached 6.5e9 by 4.8k steps, then NaN; the NaN actions hung PhysX with no error.
  - Fixes:
    1. no time-out bootstrap for the arms;
    2. no value preprocessor for the arms (`no_value_preprocessor: [arms]`), since run B drifted even without the
       bootstrap;
    3. the env raises on non-finite actions.
- **The legs' reward is never negative** (the user's rule). Run D, with the floor off during arm goals, netted
  −0.3/s while alive, so a fall cost about as much as living, and every episode fell within 1.5 s. Both agents are
  floored at 0 in every mode. The −5 termination penalty is added after the floor. Legs alive is 1.0.
- **Gates that wait on a hard metric stall.** Run C's walking gate needed 80% of the commanded path; the policy
  covered 69% on average, so 1.3% of envs ever alternated. 50/50 from step 0 worked once the reward floor was right.
- **Resume with the env state.** Every run from I on resumed a full checkpoint. The pelvis estimator and the
  curriculum levels are saved beside it (`estimator/`), so a resume keeps its levels and drift gate.

### Arms and reaching

- **IK plus a residual beats learning the whole map.** Run F's arms learned joint targets from scratch and stalled at
  12 cm. With one damped-least-squares IK step toward the command (step cap 0.1 rad) plus a policy residual of
  0.2 rad per unit, the arms observing their own wrist pose and error, run G's curriculum went from level 0 to
  spread 7 / drop 10.
- **How goals are sampled decides what is learned.**
  - Random joint angles put 47% of targets behind the pelvis and 29% on the outstretched shell; at drop 10, 96% of
    run G's goals were lowered standing targets. Wide standing reach regressed.
  - Fix (IK2): targets ≥ 0.1 m forward, tables balanced over 10 cm cells, half standing goals and half low goals.
  - Low goals come from tables built in feet-flat squats, 11 depths down to a pelvis drop of 0.38 m. Every target is
    reachable collision-free in that posture.
- **Hold the goal.** Replacing a goal 0.2 s after it was reached made the arms dart. Goals last 4 s, count as
  reached after 1 s inside 5 cm / 0.35 rad (the bonus is paid once), and keep paying for staying.
- **Judge each curriculum axis on its own goals.** Spread moves on standing goals' reach rate; drop on low goals'
  closest approach (≤ 12 cm up, > 18 cm down), so neither axis climbs on the other's goals.
- **Residual chatter is the policy, not the IK.** While holding, run H's arm joints reversed 21–30 times a second
  (pure IK: 5–10). A 3 Hz one-pole low-pass on the residual, which the arms observe filtered, halved joint speeds;
  run I held at 1.9 cm.

### Odometry: keeping the arm command on its world point

- The policies see the arm command in the pelvis frame, moved each step by an estimate of the pelvis motion, as on the
  robot. The learned estimator (MLP, ~0.03 m/s per step) moved held targets 10–15 cm over 4 s (run H). With true
  odometry, run H's 4:10 reach went from 86% to 94%.
- **Leg odometry:** the rotation comes from the attitude, the translation from a planted foot's ankle point,
  p(t−1) − dR·p(t). It brought drift to 0.9 cm per goal; the learned estimator covers the 0.5% of steps with no
  planted foot. The whole-foot-pose version was worse: loaded feet rock on their sole edges, and the 1 m lever turns
  that into millimetres per step.
- Run I closed the gap: true odometry adds only 2 points.

### GOLEM's safety layer

- **The sim enforces the same URDF limits as hard stops, so an unconstrained policy works right at them.** Run I
  would trip GOLEM's e-stops 28.5 times per robot-minute, mostly on torque (shoulder yaw asked for 4.4× its 18 Nm
  through the arm residual). Position trips came from the ankles, shoulder yaw and elbows.
- **The e-stop termination alone collapses training** (run J: e-stops ended ~96% of episodes, levels to 0).
  Bounding each target so the PD torque it asks for at the measured state stays within 85% of the effort limit cut
  trips to 1.9 per robot-minute (run K). The termination trips at 90%; GOLEM's controller applies the same bound.
- **Check the monitor before believing it.** Run L's torque trips came from `computed_torque`, which lags a substep
  and includes the randomized passive damping the motor doesn't produce. The fix computes motor torque as
  kp·(target − q) − kd_motor·qd at the current state.
- **Isaac audits did not predict RoboCasa.** Run M tripped 0.47 times per robot-minute in Isaac and 0.42 per minute
  in RoboCasa games, all position e-stops.

### Sim-to-sim: Isaac vs RoboCasa's MuJoCo

GOLEM's `tests/locomanipulation_game/sim2sim_walk.py` runs the deploy controller on a RoboCasa run's compiled MuJoCo
scene without ROS, on an open floor, with physics variants. `mujoco_evals/eval_sim2sim_walk.py` records the same
schedule in Isaac in the same format.

- **The slow walking in RoboCasa was the commands, not ROS or the kitchen.** Replaying a game's commands on an open
  floor reproduced it (path covered 0.09 for run M). The game asks mostly for backward, sideways and turning motion.
- **RoboCasa's joint damping (10 on every joint) killed backward walking.** Run M walked backward 0.02 of the command
  in MuJoCo and 1.17 with the damping removed. Contacts, friction cone, timestep, implicit PD and 10–20 ms of delay
  barely mattered. Training with damping U(8,12), armature U(0.08,0.12) and a 0–20 ms delay (run O) fixed it:
  MuJoCo forward 0.87–0.92 of the command and backward 0.53–0.64, against run M's 0.44 and 0.02.
- **Placeholder masses.** The USD import gave four URDF frames with no inertial (head camera, IMU, lidar, logo) 1 kg
  each. Isaac weighed 71.9 kg against RoboCasa's 67.5 kg, and with the torso randomization every training robot was
  heavier. They are 0.01 kg now (run Q onward).
- **MuJoCo yaw drift (open):**
  - Walking straight at 0.3 m/s, run O yaws −0.45 rad/s in MuJoCo against −0.02 in Isaac; the sign flips walking
    backward.
  - The left foot slips 0.11–0.22 m/s in stance (right 0.03), and the left hip yaws −0.15.
  - Ruled out: the model (left/right symmetric), torsional friction, command sign.
  - Stiffer MuJoCo contacts halve the drift and cut slip to Isaac's level, so part of it is MuJoCo's soft contacts.

### Gait

- **A swing-height term that only charges airborne feet is free to dodge by dragging.** Every run to M lifted the
  ankle ~1 cm. `feet_swing_clearance` charges a foot in its gait-clock swing phase for being below
  0.045 + 0.055·sin(π·progress) m, in contact or not. Lift rose to 1.5–3 cm (run O), still short of the 5.5 cm
  reference.
- **The tracking kernel's width decides what standing still earns.** At std 0.5, standing under a 0.25 m/s sideways
  command earned 78% of the tracking reward. At std 0.25, 37%.
- **Turning in place needs more than commands.**
  - Uniform sampling drew a near-pure turn 1.7% of the time, so runs P and Q drew 20% pure turns. After 100k steps
    of that, run Q still stood still: yaw 0.00, feet never lifting.
  - The cause is stillness. Run Q earns 7.0 per second standing under a 0.4 rad/s turn: 1.95 of 2.0 from the linear
    term for a motionless pelvis, 0.19 of 1.5 from the turn term.
  - On the instantaneous yaw rate, the pelvis twist of every step costs even straight walking half the turn term
    (0.76). A perfect turn in place would only break even, and half-learned attempts lose, so there is no gradient.
  - The standstill terms (stand_still, stance_base_vel, the gait clock) were not the cause: they test the norm of
    all three components.
  - Run R scores both tracking terms on the pelvis velocity averaged over one 0.8 s gait cycle: straight walking's
    turn term goes 0.76 → 1.19, and standing under a turn 0.19 → 0.10.

### RoboCasa through ROS (8 training-game seeds of 180 s)

| Policy | Full games | Stops per minute | What stopped them |
|---|---|---|---|
| K at 19.2k (Golem2) | 5 of 8 | 0.16 | barely walks |
| M | 2 of 8 | 0.42 | position e-stops: ankle roll ×4, knee extension, hip yaw |
| O | 0 of 8 | 0.86 | right-knee velocity in low reaches ×3, hip yaw while walking and turning ×3, ankle roll during turns in place ×2 |
| R at 24k | 0 of 8 | 0.73 | hip yaw at ±0.43 ×4, ankle roll ×3, right-knee velocity ×1 |

The better a policy walks, the more it moves, and the more these unsolved problems show: turning in place, the
MuJoCo yaw drift and foot slip, and the knee in deep squats. RoboCasa runs at real time beside training. Deploy parity
between the Isaac env and GOLEM's controller is ≤ 4e-4.

## Run R

Velocity tracking on the gait-cycle mean velocity (54d935f), resumed from run Q's agent_91200. Stopped at 27.4k steps
when the final run started; agent_24000 was evaluated. Run R is the first policy that turns in place.

**Isaac**, `scripts/skrl/eval_final.py`: 64 robots, RoboCasa's joint damping and armature, deterministic actions.

| Command | Achieved | Foot lift p90 |
|---|---|---|
| Forward 0.5 m/s | 0.43 m/s (vy −0.10, yaw −0.07 rad/s) | 1.1 cm |
| Backward −0.4 m/s | −0.35 m/s | 0.7 cm |
| Sideways ±0.25 m/s | +0.04 / −0.05 m/s | 0.1 cm |
| Turn in place ±0.4 rad/s | **+0.27 / −0.16 rad/s** (run Q: 0.00) | 0.1 cm (pivots on the soles) |
| Forward 0.3 m/s + turn 0.3 rad/s | 0.25 m/s, 0.28 rad/s | 0.6 cm |
| Forward 0.15 m/s | 0.11 m/s (run Q stood still) | 0.2 cm |

No falls and no e-stops while walking.

| Reach test | Standing goals reached | Low goals reached | Wrist error, last 2 s (median / p90) | Orientation error | Deepest pelvis drop |
|---|---|---|---|---|---|
| spread 4, drop 10 | 88% | 93% | 2.1–2.3 / 3.3–3.6 cm | 5° | 32.2 cm (p95 28.4) |
| spread 8, drop 10 | 50% | 53% | 3.1–3.2 / 6.2–13.6 cm | 9–11° | 32.2 cm |

- The closest approach is 1.1 cm at spread 4 and 1.8 cm at spread 8.
- The squat tables reach 38.2 cm.
- At spread 4, goals 30–40 cm deep are reached 86% of the time, with a median pelvis drop of 24.9 cm.
- At spread 8, 31 of the 347 goals end in an e-stop.

**MuJoCo, RoboCasa's physics** (GOLEM's sim-to-sim test, no ROS):
- **Default schedule:** forward tracking 0.69 of the command; one fall; 2 of the 3 crouch goals reached.
- **Turns:**
  - in place +0.10 / −0.06 rad/s;
  - forward 0.5 m/s gives 0.29 m/s with −0.55 rad/s of yaw drift;
  - backward −0.18 m/s;
  - sideways 0.
- **The same tests with stiff contacts:** forward 0.59 m/s at a 0.5 command, turn in place +0.30 rad/s, and stance slip down from 0.16 to 0.03 m/s.
- **Replaying the game's commands:** 8 of 11 goals reached, 3 of 3 crouch goals; one fall and 4 knee-velocity e-stop onsets.

**RoboCasa through ROS:**
- 0 of 8 full games, 0.73 stops per minute (run O 0.86).
- The stops: hip yaw at ±0.43 ×4 (it turns now), ankle roll ×3, right-knee velocity ×1.
- The sequence run reached both goals, standing and low, with no fall.
- Deploy parity ≤ 1e-4. The deploy leg odometry against the true pelvis motion during arm goals: 0.002 m/s median, 0.039 p90.

## The final run

`final`, launched 2026-10-07 19:01 from commit 7ea9d10: trained from scratch for 182.4k steps (7600 updates).
Directory `2026-10-07_19-01-04_final`, log `logs/skrl/final.log`. It is the task as run R had it, with these changes (the
user's choices):

- arm commands move by the **learned** pelvis-motion estimator, in training and in deployment;
- the always-zero `height_drop` input is gone: the actors see 95 and 123 values;
- 20% of moving commands are pure sideways walks (was 10%), 20% turns in place;
- hip yaw e-stops 0.05 rad inside GOLEM's limit (was 0.02), against run O's and run R's hip-yaw e-stops;
- `feet_swing_clearance` −25 (was −10), against the 1 cm foot lift;
- friction randomized over 0.3–1.2 static / 0.2–0.9 dynamic (was 0.6–1.2 / 0.4–0.9), against MuJoCo's stance slip.
  Contact-softness randomization was considered and dropped:
  - PhysX sets compliance per material at spawn (no runtime per-env API), and the envs are physics-replicated;
  - PhysX compliance only softens the normal direction, while MuJoCo's slip comes from its soft friction.

Not included: an e-stop warm-up and several seeds. Evaluate with `scripts/skrl/eval_final.py`.

## Open problems

1. **The MuJoCo contact gap is now the largest sim-to-sim difference.** With RoboCasa's contacts, forward walking at
   0.5 m/s reaches 0.29 m/s and yaws −0.55 rad/s, against 0.43 m/s in Isaac. With stiff contacts it is 0.59 m/s and
   the turn returns. The stance feet slip (left 0.11–0.22 m/s in run O).
2. **Turning in place is weak, one-sided, and done without stepping.** In Isaac +0.27 / −0.16 rad/s on ±0.4, the
   feet pivoting on their soles. MuJoCo's contacts allow it less (+0.10 / −0.06).
3. **Sideways walking is not learned:** about 0.04 m/s of 0.25 in every run since Q.
4. **The feet barely lift:** ~1 cm against the 5.5 cm swing reference.
5. **RoboCasa e-stops are joint limits under motion:**
   - hip yaw at ±0.43 while turning;
   - ankle roll at ±0.262;
   - the knee's 14 rad/s velocity limit as a deep squat collapses.
6. **Wide reach is half solved:** at spread 8, about 50% reached, and 9% of goals end in an e-stop.
7. **Odometry differs between training and deployment.** Leg odometry calls a foot planted by contact force (> 50 N)
   in training and by height (within 2 cm of the lowest foot) in GOLEM's controller. The final run sidesteps this
   with the learned estimator. That estimator drifted 3.7 cm per 4 s goal in run H, against leg odometry's 0.9 cm.

## Practical notes

- A fixed-seed rollout of a checkpoint is bit-for-bit deterministic in Isaac: compare before and after a refactor.
- A run started with `env.commands.arm_targets.table_file=<other run's file>` has no tables in its own directory, and
  every script pointed at it rebuilds them (minutes, on the GPU). Copy the file into the run directory.
- Isaac ignores SIGTERM: `timeout -s KILL`, and check `nvidia-smi --query-compute-apps` after tests. A leftover
  play.py ran 2.6 h beside runs Q and R.
- Evaluating an old run against changed task code: `git worktree add --detach <dir> <commit>`, symlink
  `third_party/CL_Assets` into it, and run with `PYTHONPATH=<dir>/source/locomanipulation_game`. Commits before
  the restructure into `tasks/locomanip_marl` keep the task at `tasks/direct/locomanip_marl`, so run that commit's own
  scripts in the worktree.

## Where things are

- Task: `source/locomanipulation_game/locomanipulation_game/tasks/locomanip_marl/`. Scripts: `scripts/skrl/`; MuJoCo
  transfer: `mujoco_evals/`; the reach evaluation: `paper/reach/`, writing to `logs/reach/`. See the README for the
  commands.
- Reports, published as Artifacts:
  - flat runs A–F: https://claude.ai/artifact/7d2dW2XKmSjeendsPb7DJw
  - run G by quarter: https://claude.ai/artifact/TAtpzWkN5fyPVx2rL3xpKL
  - run H: https://claude.ai/artifact/KugtNu4M4sWEaaUk5N9y22
  - run I: https://claude.ai/artifact/2QLR3M3w4nRSDr1yLjsaRg
  - GOLEM runs J–M: https://claude.ai/artifact/X6pziyKjHqYw4eF4xSpMhW

  Local copies are in `logs/report*/`.
- GOLEM work: `~/GOLEM`, branch camera_ready, local commits, not pushed.
  - `core_ws/src/locomotion_game_deploy`: the controller, with policies in `policies/marl_golem`.
  - `tests/locomanipulation_game/`: `run.sh` (RoboCasa through ROS, domain 194), `sim2sim.sh` and `sim2sim_walk.py`.
  - Results: `tests/results/`.
- Sim-to-sim recordings: `logs/report_golem/sim2sim/`.
