# golemlim120k

A partly trained checkpoint of `golemlim_zrange_200k`, the first run trained to stay inside GOLEM's safety layer: task
`LocoManip-Marl-Direct-v0`, branch marl-direct (the commit that adds this directory), 120,000 policy steps x 4096
envs (5000 MAPPO updates) of a 200k-step run, from scratch, 2026-10-11. Copied from
`2026-10-11_09-13-13_golemlim_zrange_200k` while it was still training.

- `checkpoints/agent_120000.pt`: both agents (skrl MAPPO). `estimator/estimator_120000.pt`: the pelvis estimator and
  the curriculum state, found from the checkpoint by `odometry.estimator_checkpoint_for`.
- `arm_target_tables_v3.pt`: the wrist-pose tables the goals are drawn from, built by `kernels_gtodom_200k` and reused
  by this run. `params/env.yaml` names it relative to this directory (the run itself pointed at the kernels run's
  copy); nothing else in the configs is changed.
- `params/`: the env and agent configs as trained. `events.out.tfevents.*`: the training curves up to about 120k steps.

What it trained with (see the README's "Changes since the paper runs"):
- every joint target clipped 0.02 rad inside GOLEM's edges (the URDF range shrunk by 0.021 rad);
- GOLEM e-stops logged, not terminal, and penalised: measured joints within 0.03 rad of GOLEM's position e-stops,
  velocity or motor torque beyond 0.8 of GOLEM's thresholds;
- arm speed and effort as squared shares of GOLEM's velocity and torque thresholds;
- goal heights uniform in [z_min, 1.7 m], z_min falling from 1.0 m to 0.2 m with the depth level;
- policies on the estimated pelvis motion, the estimator fit on arm goals only; the tracking-kernel curriculum.

At 120k steps (means over the last 2.4k steps; in brackets `estodom_clip_200k`, the same setup without the penalties
and with the old depth curriculum, at the same step):
- policy steps that would trip a GOLEM e-stop: position 4.6% (26.5%), velocity 0.9% (1.3%), torque 0.4% (0.9%).
  Leg position trips are 12x fewer; arm position trips are about 8x more and not yet attributed to a joint;
- depth level 7.4 of 8 (z_min 0.26 m); pelvis 24 cm below standing during arm goals (27 cm);
- closest approach 5.3 cm per goal (7.5 cm); kernel stage 4 (3);
- walking speed error 0.25 m/s; 1.2% of episodes end in a fall (0.8%); estimator drift 1.0 cm.

Not yet exported for GOLEM, evaluated in MuJoCo or run through the reach evaluation.
