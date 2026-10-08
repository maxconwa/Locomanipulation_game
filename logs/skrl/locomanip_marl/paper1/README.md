# paper1

The paper's method (Sec. Method) with the reward share lambda_U = lambda_L = 1, trained from scratch: task
`LocoManip-Marl-Direct-v0` at commit 3d9c8c3 (branch marl-direct), 91,200 policy steps x 4096 envs (3800 MAPPO
updates), 2 h 46 min on one GPU (RTX 5070 Ti), 2026-10-08. Copied from `2026-10-08_15-00-19_lambda1`.

The same run as paper05 except for lambda: `REWARD_SHARE` was set to 1.0 in the working tree (not committed), so the
nine shared copies carry their source term's full weight (the seven arm tracking terms in the legs' reward, the two
velocity tracking terms in the arms'). `params/env.yaml` differs from paper05's in those nine weights only.

- `checkpoints/agent_91200.pt`: both agents (skrl MAPPO). `estimator/estimator_91200.pt`: the pelvis estimator and
  the curriculum state, found from the checkpoint by `odometry.estimator_checkpoint_for`.
- `arm_target_tables_v3.pt`: the standing-reachable wrist-pose tables the goals are drawn from.
- `params/`: the env and agent configs as trained. `events.out.tfevents.*`: the training curves.

Training ended (means over the last 2.4k steps): depth level 9.9 of 10 (z_max about 1 m, mean goal offset 49 cm);
pelvis 24 cm below standing during arm goals and 14 cm while walking; closest approach 8.1 cm per goal; 6.8% of arm
goals reached (orientation error 0.50 rad against the 0.35 rad tolerance), 2.4% of the goals that need a crouch;
pelvis yaw rate 0.50 rad/s during arm goals; walking speed error 0.30 m/s; 1.2% of episodes end in a fall.

No GOLEM e-stops, RoboCasa joint damping or action delay in training; not yet exported for GOLEM. No video yet.
