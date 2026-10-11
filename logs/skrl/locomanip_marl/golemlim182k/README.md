# golemlim182k

The last good checkpoint of `golemlim_zrange_200k`, the first run trained to stay inside GOLEM's safety layer: task
`LocoManip-Marl-Direct-v0`, branch marl-direct at 82a65cb (the setup `golemlim120k` describes), 182,400 policy steps x
4096 envs, from scratch, 2026-10-11. Copied from `2026-10-11_09-13-13_golemlim_zrange_200k`.

The run went on to 200k, but at about 187k steps the legs took one update that spiked their value loss and raised
falls from about 1% to 6-10%; the depth curriculum then fell back (z_min 0.25 m -> 0.37 m). This checkpoint is from
before that.

- `checkpoints/agent_182400.pt`: both agents (skrl MAPPO). `estimator/estimator_182400.pt`: the pelvis estimator (a
  4-frame window) and the curriculum state, found from the checkpoint by `odometry.estimator_checkpoint_for`.
- `arm_target_tables_v3.pt`: the wrist-pose tables, built by `kernels_gtodom_200k`; `params/env.yaml` names it
  relative to this directory, otherwise the configs are as trained.
- `events.out.tfevents.*`: the whole run's training curves, to 200k.
- `joint_trips.txt`: which joints cross which GOLEM e-stop threshold, per physics substep, 512 envs x 30 s under the
  training distribution (scratch `joint_trips.py`).

At 182.4k steps (means over the last 2.4k steps; in brackets `estodom_clip_200k` at the same step):
- policy steps that would trip a GOLEM e-stop: position 3.7% (24.0%), velocity 1.2% (1.7%), torque 0.4% (1.0%);
- per `joint_trips.txt`, 4.5% of arm-goal steps and 1.6% of walking steps would e-stop (`estodom_clip_200k`'s final
  checkpoint: 52.5% and 1.1%). The most: the right knee past its upper e-stop in 1.24% of arm-goal substeps, the
  right ankle pitch past its lower one in 0.59%; each arm joint 0.34% or less;
- depth level 7.5 of 8 (z_min 0.25 m); pelvis 22.5 cm below standing during arm goals;
- closest approach 5.0 cm per goal (8.0 cm); kernel stage 4 (3); 1.2% of episodes end in a fall (3.3%);
- walking speed error 0.25 m/s; estimator drift over a goal 1.0 cm.

Not yet exported for GOLEM or run through the reach evaluation.
