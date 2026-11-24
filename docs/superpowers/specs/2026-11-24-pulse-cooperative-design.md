# PULSE on the cooperative task: design

Max's request (2026-11-24): after the GOLEM-aware fine-tune, "apply pulse to learn the cooperative control task",
with the gait and stance rewards zeroed so they do not compete with the motion-retargeting prior. His answers to the
questions: main's task with skrl MAPPO; main's full domain randomisation ("we just tuned [it] based on real
deployment data"); zero the gait terms, the stance terms, the gate and the torso posture penalties; run for 12 hours
(200k steps or 12 h, whichever comes first), started unattended once a smoke run shows it trains.

## What it is

`LocoManip-Marl-Pulse-v0`: main's `LocoManip-Marl-Direct-v0` (legs on a velocity command, arms on wrist poses, MAPPO
via skrl, `scripts/skrl/train.py`), where each agent's action is a 32-wide latent and the env decodes it with the
frozen stage-1 decoder of `pretraining/` (run `data/pretraining/runs/isaac_s1_golem_aware`) into joint targets, as
h12_train's stage 2 did (`data/ref/h12_train/src/h12_train/marl/actors.py::CommandActor`, run `amass_s2_pulse`).
The reconnaissance this design rests on, with file:line citations, is in the scratchpad file `pulse-recon.md` (path
in the task brief).

## Decisions

1. **The decoders' contract, not main's.** 50 Hz (`sim.dt` 0.002, `decimation` 10: main's physics rate, so main's
   0-10 substep delay stays 0-20 ms). Joint target = pretraining's default pose + pretraining's `ACTION_SCALE` x the
   decoder's output, clipped to pretraining's `TARGET_LIMITS` intersected with GOLEM's position envelope
   (`position_envelope(cfg.golem_estop)`). Legs decoder: the 12 leg joints. Upper-body decoder: torso + 14 arm joints
   (the arms agent drives the torso; no IK, no residual). Gripper joints as main holds them.
2. **Actuators: main's explicit `IdealPDActuatorCfg` with pretraining's gains, effort limits and armature**
   (`pretraining.constants` KP, KD, EFFORT, ARMATURE), and pretraining's joint ranges (its start-up `joint_ranges`
   event) and default pose as the articulation's `init_state`. Explicit, not implicit as stage 1, because main's
   randomisation (`actuator_scales`, the delay, GOLEM's torque check on `computed_torque`) only acts on explicit PD.
   Cost: the plant differs a little from stage 1's implicit PD at 5 ms; the randomisation covers more than that.
3. **Main's full randomisation and sensing, unchanged**: friction, torso mass, base and joint resets, pre-pose resets,
   actuator scales, action delay, encoder offsets, joint noise, the lowstate IMU model, the gyro low-pass.
   Whatever in the pre-pose machinery names main's default pose must keep working with pretraining's default pose; if
   a piece cannot (say so in the report), that piece alone is switched off.
4. **The decoder's proprio is what the robot would report**: projected gravity and gyro from main's lowstate IMU
   model (the same signals the legs actor observes, before the actor-side gyro low-pass unless h12_train's deploy
   path filters it: check `export_for_evals.py` on branch `supermimic` via `data/ref` and follow it), measured joint
   positions (angle + encoder offset, main's noise) minus pretraining's default, joint velocities with main's noise,
   and the last decoded action of both agents (one shared 27-wide buffer in Unitree motor order, zeroed at reset).
   One proprio per policy step, shared by both decoders, normalised by each actor's own `obs_normalizer`.
5. **PULSE's latent prior, as `amass_s2_pulse`**: `z = tanh(u + prior(hist))`, `u` the skrl actor's output.
   `LatentPrior` and `fit_prior` are ported from h12_train into `pretraining/` (`pretraining/marl/actors.py`,
   `pretraining/scripts/fit_prior.py`); the prior is fitted on the frozen stage-1 actors tracking the clips in
   `LocoManip-Pretrain-Track-v0` and saved as `<stage-1 run>/prior.pt` (+ `.json`). No prior loss term in MAPPO (no
   clip commands in main's task); the actor's last layer starts near zero (`policy_output_init_scale` 0.01) with
   initial std 0.1 for both agents, log-std bounds [log 0.03, log 0.5], `clip_actions` 5.
6. **Rewards zeroed (weight 0, terms kept so the logs show them):** `feet_swing_height_tracking`, `contact`,
   `sport_gait`, `feet_pass`, `feet_distance`, `knee_distance`, `base_height`, `knee_standing`,
   `feet_stagger_standing`, `feet_load_balance_standing`, `orientation`, `lin_vel_z`, `ang_vel_xy`.
   Removed because they read joint-space action terms: `ankle_action_rate`, `engage_smooth`, `target_near_bounds`,
   `action_beyond_clip`. Kept: velocity and wrist tracking with the lambda shares, goal bonus, `alive`, `hold_still`,
   `action_rate` (now on the latent), `target_rate`, torques / dof_vel / dof_acc / energy / effort / speed,
   `knee_valgus`, `dof_pos_limits`, `feet_contact_forces`, `self_collision`, the three GOLEM terms, the termination
   and e-stop penalties. The warm-start gate is already off in MarlDirect (deadline 1).
7. **Observations**: main's actor and critic observations as they are; `applied_action` reads the latent term
   (the clipped `u`). The gait-phase observations stay (harmless).
8. **skrl**: main's yaml, `discount_factor` 0.99 (its own comment's value at 50 Hz), rollouts 48. 200k steps is
   `--max_iterations 4167`.
9. **Not in scope**: export, evals and the deploy gate for a latent policy; a prior loss; changes to
   `LocoManip-Marl-Direct-v0`'s behaviour (the env's two `joint_pos` assumptions get a guard that leaves the existing
   tasks bit-identical).

## Files

- `source/.../locomanip_marl/mdp/pulse_actions.py`: `PulseLatentActionCfg` / `PulseLatentAction` and the shared
  decoder state (stage-1 actors, priors, proprio, last decoded action).
- `source/.../locomanip_marl/marl_pulse_env_cfg.py`: `MarlPulseEnvCfg(MarlDirectEnvCfg)`.
- `source/.../locomanip_marl/__init__.py`: the registration. `locomanip_marl_env.py`: the guard.
- `source/.../assets/h1_2.py` or the Pulse cfg: the explicit-PD robot cfg with pretraining's values.
- `pretraining/pretraining/marl/actors.py` (+ `LatentPrior`), `pretraining/pretraining/scripts/fit_prior.py`.
- Tests beside the existing ones; a README paragraph in `pretraining/README.md` and one CLAUDE.md bullet.

## Run

1. Fit the prior (minutes). 2. Smoke: a few hundred iterations at the run's env count; it "trains" if the robot
stays up longer than a zero-latent rollout and the velocity-tracking reward rises. If the frozen decoders cannot
stand under the full randomisation, report with numbers and do not start the long run. 3. The long run: 4,096 envs
if they fit, `--max_iterations 4167`, stopped at 12 h wall time (checkpoints every 9,600 steps).
