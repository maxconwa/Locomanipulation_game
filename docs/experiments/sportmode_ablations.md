# Sport-mode ablations for the legs policy (100k-step runs)

2026-10-20 to 10-23, branch marl-direct (uncommitted). Goal: make the H1-2 legs policy follow velocity commands as well
as Unitree's sport mode (h1_ai_sport, recorded in the sport-mode bags). Each run trains LocoManip-Marl-Direct-v0 for
100,008 env steps with one change against base, and is scored at the final checkpoint (agent_100008) and at 50.4k
(agent_50400) with `scripts/skrl/eval_scorecard.py`.

Rebuild the tables: `python scripts/sportmode_summary.py --out <file>` (status: `logs/sportmode_ablations/status.tsv`).

## Summary

- No single change and no combination met the targets. The best single change, gait period 0.70 s (gait070), meets
  2 of 6 at 100k and comes closest overall. It is the only change that moved every metric the right way at both
  checkpoints, but with two seeds only its turn-ratio gain clears the noise floor.
- Turn emphasis (yaw) is the only change that met the turn targets: turn sense 0.993, angular error 0.046 at 100k;
  turn ratio 0.94 at 50.4k. It costs linear tracking at 100k (speed ratio 0.36, cos 0.62), and 8 of 168 envs fell at 50.4k.
- Five changes made things worse.
  - The command schedule (cmd) and both combos that include it are worse. Resampling every 1–3 s runs the
    velocity-kernel curriculum far faster per step, and the arm goals never opened. Slowing the kernel window 5×
    (cmd_tw, combo_tw) did not fix it: cos 0.55 and 0.27 at 100k.
  - Sport-mode PD gains (gains) marched in place: speed ratio 0.22, cos 0.39.
  - The 100 Hz policy (hz100) was good at 50.4k (speed 1.06, cos 0.93). It collapsed once arm goals opened at about
    67k steps: training falls reached 99% and never recovered.
- Every run finished with exit 0 and no "fail to launch kernel" in any training or eval log.

Recommended 200k run: base + gait070, two seeds. Add gate if the deploy wants sport mode's stand/walk behaviour
(details under Recommendation).

## Setup

**Defaults.** On 2026-10-20 the env defaults were switched to Mid360v2's training config (Explicit500's physics), and
base is that config with no overrides:

- 500 Hz explicit PD (dt 0.002, decimation 10), 0–10 substep action delay
- no fixed observation scales, no link-mass or CoM randomization, torso mass −2 to +4 kg
- contact force as the peak over a 3-substep window
- flat terrain row 0 and a gated terrain curriculum
- golemlim120k arm-goal table

A dump of the bare defaults matches Mid360v2's env.yaml except for the seed and the new switch fields. The first
queue (13:38 that day) had run on the older ALMI-Open defaults (200 Hz, fixed obs scales, mass/CoM randomization,
rows 0–5). It was aborted at 33k (falls 18%, walk cos 0.05) and is not reported. Its partial run is
`logs/skrl/locomanip_marl/2026-10-20_13-38-04_sm_base`.

**Switches.** Each is a hydra override on base (`scripts/sportmode_ablations.sh`).

| Tag | Change | Overrides |
| --- | --- | --- |
| base, base_s2 | none (seed 42, 43) | |
| gate | 1. Stand/walk gate | `zero_cmd_eps` 0.1 → 0.03; 20% of moving commands have norm 0.05–0.15 |
| cmd | 2. Command schedule | resampled every U(1, 3) s (was 10 s), low-passed with τ 0.5 s for observations and rewards |
| range | 3. Ranges | vy ±0.3 → ±0.5, wz ±0.5 → ±1.0 m/s, rad/s; turn-in-place 0.2–0.5 → 0.2–1.0, sideways draws widened |
| yaw | 4. Turn emphasis | yaw tracking weight → 2.0 (as linear); last yaw-kernel stage 0.2 → 0.14 rad/s |
| gait070 | 5. Gait period | 0.8 → 0.70 s (tracking windows follow it) |
| gains | 6. Sport-mode PD | kp/kd hip 250/6.5, knee 400/10, ankle pitch 80/1.5, ankle roll 40/0.4, torso 600/6 (base: 200/2.5, 300/4, 40/2, 40/2, 300/3) |
| hz100 | 7. 100 Hz policy | decimation 5; per-step rate penalties ×4, arm IK step /2, discount 0.995, GAE λ 0.975 |
| combo | 1–5 | |
| combo_all | 1–7 | |
| cmd_tw | 2 + kernel pace | `kernel_window_segments` 20,000 → 100,000 (base's pace in seconds) |
| combo_tw | 1–5 + kernel pace | |

**Scorecard.** `eval_scorecard.py` runs 168 envs on flat terrain row 0 with a scripted command sequence (settle, steps,
flicks, small commands) inside base's command ranges. That is narrower than range's training ranges, so range is
scored only on what base also trained. Velocity is scored against the lag-allowed command, raw: no creep correction
(sport mode's 0.96 cos has its +0.10 m/s forward creep removed; raw it is 0.65). The stride window is the run's own
gait period. On the sport-mode bags the scorecard reproduces the bag analysis
(`scripts/skrl/test_scorecard_sportmode.py`, passing after the defaults switch).

**Reference.** Mid360v2 (196.8k steps, the defaults' source), re-scored under the new defaults:

- turn sense 0.988, speed ratio 0.973, turn ratio 0.835, cos 0.945, angular error 0.056, linear error 0.068
- 0 falls; not stepping 13.6% (small commands 46.3%)

The 100k-step base runs are below it on every metric, as expected at half the training.

## Scorecards

### Final checkpoint (100k steps)

| Run | Changes | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err | Met | Stride (IQR) s | Roll / pitch sd ° | Falls | Commanded, not stepping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Target | | ≥ 0.99 | ≥ 0.90 | ≥ 0.90 | ≥ 0.95 | ≤ 0.05 | ≤ 0.15 | 6 | | | | |
| Sport mode (bags) | | 0.994 | 0.93 | 0.91 | 0.96* | 0.050 | 0.147 | | 0.700 (0.01) | 0.83 / 0.75 | | |
| Explicit500 (robot) | | 0.896 | 0.96 | 0.61 | 0.54 | 0.137 | 0.151 | | 0.790 (0.08) | 2.03 / 2.22 | | 28% |
| base | none | 0.970 | 0.799 | 0.733 | 0.906 | 0.072 | **0.088** | 1 | 0.800 (0.06) | 3.39 / 1.59 | 0/168 | 14% (small 46%) |
| base_s2 | none (seed 43) | 0.951 | 0.802 | 0.730 | 0.904 | 0.087 | **0.087** | 1 | 0.780 (0.10) | 3.82 / 1.71 | 0/168 | 14% (small 45%) |
| gate | 1 | 0.966 | 0.655 | 0.706 | 0.872 | 0.077 | **0.101** | 1 | 0.800 (0.06) | 2.51 / 1.46 | 0/168 | 3% (small 1%) |
| cmd | 2 | 0.839 | 0.292 | 0.242 | 0.531 | 0.092 | **0.107** | 1 | 0.780 (0.72) | 4.91 / 6.48 | 0/168 | 51% (small 55%) |
| range | 3 | 0.889 | **0.947** | 0.524 | 0.903 | 0.106 | **0.084** | 2 | 0.780 (0.12) | 3.94 / 1.31 | 0/168 | 15% (small 44%) |
| yaw | 4 | **0.993** | 0.361 | 0.856 | 0.623 | **0.046** | **0.146** | 3 | 0.800 (0.08) | 2.26 / 1.67 | 0/168 | 16% (small 46%) |
| gait070 | 5 | 0.978 | **0.908** | 0.845 | 0.939 | 0.065 | **0.065** | 2 | 0.680 (0.32) | 3.43 / 1.62 | 0/168 | 13% (small 45%) |
| gains | 6 | 0.940 | 0.219 | 0.580 | 0.385 | 0.080 | 0.160 | 0 | 0.800 (0.08) | 2.06 / 1.37 | 0/168 | 10% (small 36%) |
| hz100 | 7 | 0.514 | 0.184 | 0.013 | −0.126 | 0.146 | **0.149** | 1 | 0.370 (0.17) | 2.44 / 2.72 | 27/168 | 0% (small 0%) |
| combo | 1–5 | 0.728 | 0.715 | 0.293 | 0.040 | 0.163 | 0.202 | 0 | 0.660 (0.64) | 6.16 / 11.25 | 3/168 | 24% (small 27%) |
| combo_all | 1–7 | 0.671 | **1.640** | 0.366 | −0.216 | 0.201 | 0.273 | 1 | 0.450 (0.38) | 8.38 / 8.01 | 33/168 | 8% (small 17%) |
| cmd_tw | 2 + pace | 0.836 | 0.416 | 0.273 | 0.548 | 0.101 | **0.106** | 1 | 0.720 (0.50) | 5.08 / 7.62 | 1/168 | 22% (small 28%) |
| combo_tw | 1–5 + pace | 0.690 | 0.305 | 0.165 | 0.274 | 0.107 | **0.121** | 1 | 0.700 (0.66) | 6.60 / 7.79 | 0/168 | 21% (small 17%) |

### 50.4k checkpoint

| Run | Changes | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err | Met | Stride (IQR) s | Roll / pitch sd ° | Falls | Commanded, not stepping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Target | | ≥ 0.99 | ≥ 0.90 | ≥ 0.90 | ≥ 0.95 | ≤ 0.05 | ≤ 0.15 | 6 | | | | |
| base | none | 0.982 | 0.692 | 0.844 | 0.918 | 0.066 | **0.082** | 1 | 0.760 (0.34) | 3.75 / 2.24 | 0/168 | 14% (small 43%) |
| base_s2 | none (seed 43) | 0.990 | **1.019** | 0.856 | **0.963** | 0.053 | **0.057** | 3 | 0.780 (0.22) | 3.34 / 2.58 | 0/168 | 13% (small 43%) |
| gate | 1 | 0.982 | **0.924** | 0.777 | 0.886 | 0.062 | **0.080** | 2 | 0.800 (0.06) | 2.51 / 1.62 | 0/168 | 1% (small 0%) |
| cmd | 2 | 0.673 | 0.353 | 0.133 | 0.158 | 0.122 | **0.148** | 1 | 0.800 (0.68) | 5.43 / 13.35 | 0/168 | 35% (small 47%) |
| range | 3 | 0.820 | 0.498 | 0.345 | 0.746 | 0.113 | **0.133** | 1 | 0.800 (0.08) | 3.12 / 1.38 | 0/168 | 15% (small 43%) |
| yaw | 4 | **0.997** | 0.735 | **0.940** | 0.894 | **0.040** | **0.090** | 4 | 0.780 (0.34) | 4.09 / 1.91 | 8/168 | 8% (small 27%) |
| gait070 | 5 | 0.989 | 0.856 | 0.860 | **0.954** | 0.057 | **0.066** | 2 | 0.680 (0.20) | 3.78 / 1.87 | 0/168 | 10% (small 35%) |
| gains | 6 | 0.900 | 0.241 | 0.496 | 0.413 | 0.100 | 0.162 | 0 | 0.820 (0.18) | 1.66 / 1.57 | 0/168 | 15% (small 41%) |
| hz100 | 7 | 0.971 | **1.062** | 0.788 | 0.932 | 0.065 | **0.074** | 2 | 0.560 (0.41) | 4.36 / 2.09 | 0/168 | 12% (small 42%) |
| combo | 1–5 | 0.710 | 0.369 | 0.256 | 0.207 | 0.139 | **0.145** | 1 | 0.720 (0.68) | 5.68 / 7.51 | 3/168 | 25% (small 31%) |
| combo_all | 1–7 | 0.642 | **1.425** | 0.294 | −0.393 | 0.221 | 0.308 | 1 | 0.490 (0.39) | 8.15 / 7.75 | 93/168 | 5% (small 10%) |
| cmd_tw | 2 + pace | 0.748 | 0.419 | 0.237 | 0.489 | 0.128 | **0.127** | 1 | 0.800 (0.62) | 4.86 / 6.04 | 0/168 | 20% (small 31%) |
| combo_tw | 1–5 + pace | 0.683 | 0.374 | 0.157 | 0.389 | 0.119 | **0.126** | 1 | 0.760 (0.70) | 4.14 / 6.19 | 0/168 | 17% (small 16%) |

Bold: meets the target. *Sport mode's cos with its forward creep removed (0.65 raw); ours are raw. A speed ratio
above 1 meets the ≥ 0.90 target but is overshoot: combo_all's 1.64 comes with a negative cos.

## Noise floor and what helps beyond it

The noise floor is |base − base_s2| per metric, at each checkpoint. At 100k the two seeds agree almost exactly;
at 50.4k they differ widely in speed ratio and cos (base_s2 was ahead, base caught up and passed it in turn ratio by 100k):

| | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err |
| --- | --- | --- | --- | --- | --- | --- |
| Noise, 100k | 0.019 | 0.003 | 0.003 | 0.002 | 0.014 | 0.001 |
| Noise, 50.4k | 0.008 | 0.327 | 0.013 | 0.045 | 0.013 | 0.025 |
| Floor used (max) | 0.019 | 0.327 | 0.013 | 0.045 | 0.014 | 0.025 |

A 100k-only floor would call almost any difference significant, so the floor used here is the larger of the two.
Two seeds still make it a rough floor. Change against base, with ✓ when the change beats the floor in the right direction:

| Run | Ckpt | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate | 100k | −0.004 | −0.144 | −0.027 | −0.034 | +0.004 | +0.013 |
| | 50.4k | +0.000 | +0.232 | −0.066 | −0.032 | −0.004 | −0.002 |
| cmd | 100k | −0.131 | −0.507 | −0.491 | −0.375 | +0.020 | +0.019 |
| | 50.4k | −0.309 | −0.339 | −0.710 | −0.760 | +0.056 | +0.066 |
| range | 100k | −0.081 | +0.148 | −0.209 | −0.003 | +0.034 | −0.005 |
| | 50.4k | −0.162 | −0.194 | −0.498 | −0.171 | +0.047 | +0.051 |
| yaw | 100k | +0.022 ✓ | −0.438 | +0.123 ✓ | −0.283 | −0.026 ✓ | +0.058 |
| | 50.4k | +0.015 | +0.043 | +0.097 ✓ | −0.024 | −0.026 ✓ | +0.008 |
| gait070 | 100k | +0.007 | +0.109 | +0.112 ✓ | +0.033 | −0.008 | −0.023 |
| | 50.4k | +0.007 | +0.164 | +0.016 ✓ | +0.037 | −0.009 | −0.016 |
| gains | 100k | −0.030 | −0.579 | −0.153 | −0.521 | +0.008 | +0.072 |
| | 50.4k | −0.082 | −0.450 | −0.347 | −0.505 | +0.034 | +0.080 |
| hz100 | 100k | −0.456 | −0.615 | −0.719 | −1.032 | +0.073 | +0.061 |
| | 50.4k | −0.010 | +0.370 (overshoot) | −0.056 | +0.015 | −0.001 | −0.008 |
| combo | 100k | −0.242 | −0.084 | −0.440 | −0.866 | +0.091 | +0.114 |
| | 50.4k | −0.272 | −0.322 | −0.587 | −0.711 | +0.073 | +0.062 |
| combo_all | 100k | −0.299 | +0.841 (overshoot) | −0.366 | −1.122 | +0.129 | +0.185 |
| | 50.4k | −0.340 | +0.733 (overshoot) | −0.549 | −1.310 | +0.155 | +0.226 |
| cmd_tw | 100k | −0.134 | −0.383 | −0.460 | −0.358 | +0.028 | +0.018 |
| | 50.4k | −0.234 | −0.273 | −0.607 | −0.429 | +0.062 | +0.045 |
| combo_tw | 100k | −0.280 | −0.494 | −0.568 | −0.632 | +0.034 | +0.032 |
| | 50.4k | −0.299 | −0.318 | −0.687 | −0.529 | +0.053 | +0.044 |

What helps beyond noise:

- **gait070 (5).** All 12 deltas point the right way. At both checkpoints its turn-ratio gain clears the floor;
  its speed ratio (+0.11/+0.16), cos (+0.03/+0.04) and linear error (−0.02) gains fall just inside it. Its stride is
  0.68 s, close to sport mode's 0.70 s, though more variable (IQR 0.20–0.32 s against 0.01 s). It also has the best
  arm numbers of the clean runs (reach 8.3%, falls 11% in training). This is the one change worth keeping on this evidence.
- **yaw (4).** Turn sense, turn ratio and angular error all clear the floor, and yaw is the only run that met the turn
  targets. At 100k linear tracking fell (speed 0.36, cos 0.62); at 50.4k it was still fine (0.74 / 0.89) but 8 envs
  fell. It is a trade, not a gain: the weight 2.0 pulls the policy toward turning at linear tracking's cost by 100k.
- **gate (1).** It cuts "commanded but not stepping" from 14% to 3% (small commands 46% → 1%). That is what it is for,
  and the only behavioural change as large as that in the study. It moves none of the six targets beyond noise
  (speed ratio −0.14 at 100k, +0.23 at 50.4k).
- **range (3).** Nothing beyond noise; turn ratio is far worse (−0.21 / −0.50). The scorecard commands only base's
  ranges, so range is scored on a subset of what it trained. It had the best arm reach (9.2%) and training cos (0.72).
- **Worse:** cmd (2), gains (6), hz100 (7) at 100k, combo, combo_all. Causes are below.

## Why runs failed

**cmd, combo, combo_all: the velocity-kernel curriculum.** ModalVelocityCommand promotes the kernels after judging
20,000 command segments. Resampling every U(1, 3) s instead of 10 s makes about 5× as many segments per step, so
the curriculum reached its narrowest stage by about 10–12k steps, long before the gait could track it. A 24-env raw
eval of cmd confirmed the command filter is working (the filtered command ramps to 0.6 while actual vx stays near 0),
and nothing outside commands.py reads the raw command.

None of the three ever opened the arm goals, which need a walk cos of 0.75. combo's training cos fell steadily from
0.36 at 10k to −0.09 at 100k. combo_all adds hz100 on top: 33 of 168 eval envs fell, with 40% e-stop steps. cmd_tw
and combo_tw slowed the kernel window 5× (`kernel_window_segments` 100,000); that was not enough (below).

**gains: stiffer PD at the same action scale.** The legs marched in place: speed ratio 0.22, cos 0.39, and
training cos 0.27 at 100k. The legs' action scale stayed 0.25 rad, so the same action now asks for more torque (kp hip 200 → 250, knee 300 → 400, ankle
pitch 40 → 80, torso 300 → 600) against far more damping (kd hip 2.5 → 6.5, knee 4 → 10). Sport mode's policy was
trained around its own gains. A fair test of change 6 would rescale the action or retune the gait rewards with it.

**hz100: collapse after the arm goals opened.** Up to about 67k steps hz100 trained better than any other run: walk
cos 0.76 and 2–3% falls. The 50.4k checkpoint scores speed 1.06 and cos 0.93 with no falls. Arm goals opened at about
67k. Falls then jumped to 62%, reached about 99% by 81k, and never recovered; wrist error grew to 0.78 m.

base also had a fall spike when the goals opened (65% at 50k), but it recovered. Several settings count in steps or
frames, not seconds, so at 100 Hz they cover half the time. Untested candidates:

- the arm-goal ramp, which reaches its full share of goals in half the simulated time
- the 24-step rollouts and LSTM training sequence (0.24 s instead of 0.48 s)
- the estimator's 25-frame window (0.25 s)
- the per-step reward floor at 0, which clips twice as often per second when the arm penalties are large

Also, 100k steps at 100 Hz is only half the simulated time of the 50 Hz runs.

## Arms and falls in training

Means over the 2400 env steps before each checkpoint, from the training logs. Reach rate = goals reached / (reached + missed).

| Run | Hours | Arm goals on | Reach rate 100k | Wrist error 100k (m) | Closest approach 100k (m) | Fall share 100k | Fall share 50k | Training cos 100k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 5.95 | yes | 0.049 | 0.191 | 0.068 | 0.154 | 0.649 | 0.611 |
| base_s2 | 5.92 | yes | 0.010 | 0.233 | 0.091 | 0.133 | 0.030 | 0.546 |
| gate | 5.77 | yes | 0.005 | 0.314 | 0.152 | 0.289 | 0.022 | 0.473 |
| cmd | 5.75 | no | — | — | — | 0.028 | 0.071 | 0.396 |
| range | 6.42 | yes | 0.092 | 0.143 | 0.044 | 0.110 | 0.253 | 0.718 |
| yaw | 6.57 | yes | 0.007 | 0.251 | 0.119 | 0.137 | 0.747 | 0.365 |
| gait070 | 6.24 | yes | 0.083 | 0.169 | 0.054 | 0.109 | 0.540 | 0.694 |
| gains | 6.66 | yes | 0.100 | 0.145 | 0.046 | 0.123 | 0.279 | 0.269 |
| hz100 | 3.43 | yes | 0.000 | 0.782 | 0.521 | 0.835 | 0.026 | 0.037 |
| combo | 6.07 | no | — | — | — | 0.107 | 0.106 | −0.086 |
| combo_all | 3.59 | no | — | — | — | 0.194 | 0.178 | −0.035 |
| cmd_tw | 5.99 | no | — | — | — | 0.038 | 0.060 | 0.456 |
| combo_tw | 6.25 | no | — | — | — | 0.083 | 0.079 | 0.181 |

At 100k every run is early in the arm phase: the goals open between about 45k and 67k steps, so reach rates are
low (0–10%) and noisy (base 4.9% against base_s2 1.0%). Fall shares at 50k mostly show when each run's goals opened,
not how stable it is. No arm metric separates the changes beyond the seed gap except hz100's collapse.

## cmd_tw and combo_tw

Both reran a cmd-based config with the kernel window at 100,000 segments, 5× the default, to put the curriculum back at
base's pace in seconds. Neither recovered:

- cmd_tw at 100k: cos 0.55, speed ratio 0.42, turn ratio 0.27 (cmd: 0.53 / 0.29 / 0.24; base: 0.91 / 0.80 / 0.73).
  Training cos peaked at 0.53 near 80k, so the arm goals never opened.
- combo_tw at 100k: cos 0.27, speed 0.31, turn ratio 0.17. combo's cos was 0.04, but combo_tw's speed and turn
  ratios are worse. Training cos stayed between 0.18 and 0.43; the arm goals never opened.
- Both have fewer falls than their originals (1 and 0 of 168 in eval) and step more on command (22% and 21% not
  stepping, against cmd's 51%).

The window fix did not slow the curriculum enough. The linear kernel stage at 20k steps was 2 in base, 5 in cmd and
4 in both tw runs. A 5× window should have matched base, so either short segments of a filtered command pass the
promotion test much more easily than 10 s segments, or the window does not count what was assumed. That is not
checked here. Either way, change 2 as tested does not train well in 100k steps, and its curriculum interaction needs
fixing before it is tried again.

## Recommendation for the 200k run

Base + gait070 (`env.gait_period=0.7`), two seeds (42, 43). That is the only change that improved every metric at
both checkpoints. Optional additions, each as its own seed pair if the budget allows:

- **+ gate** (`env.zero_cmd_eps=0.03 env.commands.base_velocity.small_command_prob=0.2`), if the deploy should step
  on small commands as sport mode does; on the robot Explicit500 did not step for 28% of commanded time. It costs no
  target beyond noise.
- **+ yaw at a lower weight** (e.g. 1.5 instead of 2.0), to keep its turn gains without the linear loss seen at 100k.
- Leave out **cmd**: neither kernel pace trained it well. Its curriculum interaction needs fixing first (see cmd_tw and combo_tw).
- Leave out gains (needs an action-scale retune first) and hz100 (needs the step-counted settings converted to
  seconds first; see above).

## GOLEM lmg_deploy changes needed (not made)

GOLEM was not touched. For a policy trained with these switches, the deploy needs:

1. **Command filter** (change 2): a first-order low-pass on the velocity command in the `velocity_command` path before
   the observation, with τ = `command_filter_tau` (0.5 s) at the control rate. Reset it to the current command when
   the policy engages, and hold it at zero in arm mode. Training filters both the observation and the reward command,
   so feeding the raw joystick would be a train/deploy mismatch.
2. **Gains from the export** (change 6): export.yaml already carries the per-joint kp/kd the run trained with
   (`scripts/skrl/export_marl.py`, via `common.read_deploy_settings`). lmg_deploy must apply them to LowCmd instead of
   its template's table, or a sport-gain policy runs on base gains and the reverse.
3. **100 Hz loop** (change 7): a `control_hz` launch parameter (or export.yaml's `control_dt`) driving the policy
   timer at 100 Hz, with the observation history, action delay and gyro low-pass at the matching dt.
4. **The other export.yaml keys**: export.yaml already writes `gait_period`, `zero_cmd_eps`, `command_ranges`,
   `control_dt` and `command_filter_tau`. The node must use them: the gait clock's phase rate (0.70 s for gait070),
   the stand/walk threshold (0.03 for gate) and the joystick clip to the trained ranges (wider for range). Today the
   deploy does not apply `command_filter_tau` at all (item 1).

gait070 alone needs only item 4's `gait_period` to be honoured by the node's gait clock.

## Method notes

- 100k steps (4167 updates × 24) is half the default 200k; checkpoints every 2400 steps, plus the final one saved by
  train.py (agent_100008). At 100 Hz the same steps cover half the simulated time.
- One training process at a time, about 5.2 steps/s at 500 Hz physics (6–6.7 h per 50 Hz run, 3.4–3.6 h at 100 Hz).
  Every switch was smoke-tested for 30 updates before the queue (`logs/sportmode_ablations/smoke/status.tsv`).
- PD stability at 500 Hz: knee kd·dt/I was checked at 200 Hz (0.16, no chatter); 500 Hz is more stable still.
  This includes the sport-mode gains.
- The legs' per-step reward stays floored at 0 in every run; changes shape behaviour through weights only.
- Logs: `logs/sportmode_ablations/` (train_/eval_ logs per tag, queue logs, status.tsv); runs in
  `logs/skrl/locomanip_marl/*_sm_<tag>`, scorecards in each run's `checkpoints/scorecard_<step>.{json,md}`.
