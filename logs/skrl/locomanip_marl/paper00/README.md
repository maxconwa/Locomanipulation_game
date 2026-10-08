# paper00

The paper's method (Sec. Method) with the reward share lambda_U = lambda_L = 0, trained from scratch: task
`LocoManip-Marl-Direct-v0` at commit 3d9c8c3 (branch marl-direct), 91,200 policy steps x 4096 envs (3800 MAPPO
updates), 2 h 55 min on one GPU, 2026-10-08. Copied from `2026-10-08_14-44-30_lambda_0`.

The same run as paper05 except for lambda: the code's `REWARD_SHARE` stayed 0.5 and the nine shared copies were zeroed
by Hydra overrides (`env.rewards.legs.arms_<term>.weight=0.0` for the seven arm tracking terms,
`env.rewards.arms.legs_<term>.weight=0.0` for the two velocity tracking terms). `params/env.yaml` differs from
paper05's in those nine weights only.

- `checkpoints/agent_91200.pt`: both agents (skrl MAPPO). `estimator/estimator_91200.pt`: the pelvis estimator and
  the curriculum state, found from the checkpoint by `odometry.estimator_checkpoint_for`.
- `arm_target_tables_v3.pt`: the standing-reachable wrist-pose tables the goals are drawn from.
- `params/`: the env and agent configs as trained. `events.out.tfevents.*`: the training curves.

Training ended (means over the last 2.4k steps): depth level 9.9 of 10 (z_max about 1 m, mean goal offset 49 cm);
pelvis 28 cm below standing during arm goals and 12 cm while walking; closest approach 8.5 cm per goal; 4.5% of arm
goals reached (orientation error 0.53 rad against the 0.35 rad tolerance), 2.3% of the goals that need a crouch;
pelvis yaw rate 0.51 rad/s during arm goals; walking speed error 0.24 m/s; 1.6% of episodes end in a fall.

No GOLEM e-stops, RoboCasa joint damping or action delay in training; not yet exported for GOLEM. No video yet.
