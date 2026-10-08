# paper05

The paper's method (Sec. Method) with the reward share lambda_U = lambda_L = 0.5, trained from scratch: task
`LocoManip-Marl-Direct-v0` at commit 3d9c8c3 (branch marl-direct), 91,200 policy steps x 4096 envs (3800 MAPPO
updates), 2 h 25 min on one GPU, 2026-10-08. Copied from `2026-10-08_14-34-36_paper_yaw_91k`.

- `checkpoints/agent_91200.pt`: both agents (skrl MAPPO). `estimator/estimator_91200.pt`: the pelvis estimator and
  the curriculum state, found from the checkpoint by `odometry.estimator_checkpoint_for`.
- `arm_target_tables_v3.pt`: the standing-reachable wrist-pose tables the goals are drawn from.
- `params/`: the env and agent configs as trained. `events.out.tfevents.*`: the training curves.
- `videos/`: the final checkpoint, deterministic: walk 0.4 m/s, stop, a crouch reach to a goal 0.40 m below the rest
  pose (5 s), walk. `summary.txt` has its numbers.

Training ended (means over the last 2.4k steps): depth level 9.9 of 10 (z_max about 1 m, mean goal offset 49 cm);
pelvis 23 cm below standing during arm goals and 13 cm while walking; closest approach 8.7 cm per goal; 7% of arm
goals reached (orientation error 0.48 rad against the 0.35 rad tolerance), 2.5% of the goals that need a crouch;
pelvis yaw rate 0.44 rad/s during arm goals; walking speed error 0.28 m/s; 2.7% of episodes end in a fall.

No GOLEM e-stops, RoboCasa joint damping or action delay in training; not yet exported for GOLEM.
