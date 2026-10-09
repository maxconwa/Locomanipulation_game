# paperwb256

The whole-body baseline with the game players' network size: paperwb with actor and critic [256, 128, 64] (each the
size of one player's network) instead of [512, 256, 128]. Task `LocoManip-WholeBody-Direct-v0`, task code as at commit
7a43084 (branch marl-direct), 91,200 policy steps x 4096 envs (3800 MAPPO updates), 2 h 22 min on one GPU,
2026-10-08. Copied from `2026-10-08_21-00-30_paperwb256`.

**Setup:** paperwb's, except the two network overrides
`agent.models.policy.network=[{name:net,input:OBSERVATIONS,layers:[256,128,64],activations:elu}]` and the same for
`agent.models.value.network` with `input:STATES`. `params/env.yaml` is identical to paperwb's; `params/agent.yaml`
differs only in the layers and the experiment name.

**Training ended** (means over the last 2.4k steps):

| | paper05 (two agents, lambda 0.5) | paperwb (one agent, [512, 256, 128]) | paperwb256 (one agent, [256, 128, 64]) |
|---|---|---|---|
| Depth level (of 10) | 9.9 | 9.4 | 9.7 |
| Closest approach per goal | 8.7 cm | 11.0 cm | 9.9 cm |
| Arm goals reached | 7.4% | 1.4% | 2.5% |
| Goals needing a crouch reached | 2.5% | 0.3% | 0.6% |
| Orientation error during goals | 0.48 rad | 0.72 rad | 0.50 rad |
| Pelvis drop during arm goals / walking | 23 / 13 cm | 21 / 13 cm | 21 / 13 cm |
| Walking speed error | 0.28 m/s | 0.31 m/s | 0.33 m/s |
| Falls (share of episode ends) | 2.7% | 1.6% | 4.4% |

**Sample efficiency** (millions of environment transitions to reach each target, on curves smoothed over 50 updates,
computed as `logs/figures/sample_efficiency/milestones.csv` is; that figure does not include this run):

| | paper05 | paperwb | paperwb256 |
|---|---|---|---|
| Closest approach <= 12 cm | 79 | 245 | 177 |
| Depth level 5 | 134 | 282 | 237 |
| Depth level 9 | 186 | 353 | 304 |

The smaller network learns faster and ends closer to the two-agent runs than paperwb, but still behind paper05 on every
reach metric, and falls more. No video was recorded for this run.

- `checkpoints/agent_91200.pt`, `estimator/estimator_91200.pt`, `arm_target_tables_v3.pt`, `params/` and the
  TensorBoard events. Binaries in Git LFS.
