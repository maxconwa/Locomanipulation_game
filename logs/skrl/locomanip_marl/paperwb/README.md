# paperwb

The whole-body baseline for the paper's two-agent game: the same task with ONE agent for all 26 joints, trained from
scratch. Task `LocoManip-WholeBody-Direct-v0` at commit 7a43084 (branch marl-direct), 91,200 policy steps x 4096 envs
(3800 MAPPO updates), 2 h 23 min on one GPU, 2026-10-08. Copied from `2026-10-08_17-54-06_paperwb`.

**Setup (what differs from paper05):**
- One agent, `whole`, drives both action terms (legs q0 + 0.25 a, arms IK step + filtered residual): 26 actions.
- It observes both agents' observations merged: the shared vector once, both previous actions, wrist poses and
  wrist errors (135 values). The critic sees that plus the same privileged state.
- Reward: every term of both agents' own rewards once, none of the shared copies:
  r = r(T^goal) + r(v^cmd) + r_U^shaping + r_L^shaping, floored at 0, -5 on a fall.
- Actor and critic [512, 256, 128] (paper05: [256, 128, 64] per agent). The actor has 1.7x the two actors' parameters,
  actor and critic together 1.4x the two agents'.
- Entropy bonus 0.01 on the 12 leg dimensions only, time-out bootstrapping, log-std caps 0.6 (legs) and 1.0 (arms), as
  the two agents had them. KL target 0.04, the two agents' 0.02 each combined.
- Everything else (commands, warm start, goal tables, depth curriculum, estimator, physics) is paper05's.

A first attempt (stopped at 11k steps) had the entropy bonus on all 26 dimensions and a 0.02 KL target; it is not this run.

**Training ended** (means over the last 2.4k steps; paper05 and paper00 over the same steps):

| | paper05 (two agents, lambda 0.5) | paper00 (two agents, lambda 0) | paperwb (one agent) |
|---|---|---|---|
| Depth level (of 10) | 9.9 | 9.9 | 9.4 |
| Closest approach per goal | 8.7 cm | 8.5 cm | 11.0 cm |
| Arm goals reached | 7.4% | 4.5% | 1.4% |
| Goals needing a crouch reached | 2.5% | 2.3% | 0.3% |
| Orientation error during goals | 0.48 rad | 0.53 rad | 0.72 rad |
| Pelvis drop during arm goals / walking | 23 / 13 cm | 28 / 12 cm | 21 / 13 cm |
| Walking speed error | 0.28 m/s | 0.24 m/s | 0.31 m/s |
| Falls (share of episode ends) | 2.7% | 1.6% | 1.6% |

During training the single agent's depth level stayed below 1 until about 50k steps (paper05: 9.3 at 48k), then
climbed to 9.4. Compare runs on task metrics, not total reward: the two-agent total includes the shared copies.

**Deterministic video** (`videos/`): the policy does not walk at the 0.4 m/s command; it steps in place (forward speed
0.00 m/s, paper05 0.32-0.36 m/s in the same sequence). In the crouch reach the pelvis drops to 0.76 m and the wrists
come within 6.9 cm (paper05: 0.72 m and 1.8 cm). Its linear-velocity tracking reward in training ended at 1.28 per
episode-second against paper05's 1.47.

- `checkpoints/agent_91200.pt`, `estimator/estimator_91200.pt`, `arm_target_tables_v3.pt`, `params/`, the TensorBoard
  events, and `videos/` (deterministic walk / crouch reach / walk, as paper05's).
