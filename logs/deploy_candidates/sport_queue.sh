#!/usr/bin/env bash
# The sport-mode deploy candidate (2026-10-24, evening): the user's call after DeployRand300k / Deploy300k_s42 —
# copy Unitree's sport mode (gait period, command ranges, gains, its standing knee, a 100 Hz policy) and gate the arm
# goals on the sport-mode scorecard's numbers; keep the drawn episode lengths; physics, PD and the simulated delays
# stay at 500 Hz. Two seeds of 600k policy steps (100 Hz: the 300k-step 50 Hz runs' simulated time), each then
# scored on the sport-mode scorecard and exported. No MuJoCo stress test or bag replay: mujoco_evals/mujoco_sim.py
# runs 50 Hz policies at the nominal gains and one action scale, so it cannot run these yet (nor can lmg_deploy).
#
#   bash logs/deploy_candidates/sport_queue.sh            (tmux sportcand)
#   SMOKE=<updates> bash logs/deploy_candidates/sport_queue.sh   (seed 42 only, into logs/skrl/sport_smoke, no scoring)
set -u
set -f  # the [a,b] overrides are not globs
cd "$(dirname "$0")/../.."
OUT=$PWD/logs/deploy_candidates
STATUS=$OUT/status.txt
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

ITERATIONS=12500   # x 48 = 600,000 policy steps (6000 s per env)
FINAL=600000
MID=393600         # a checkpoint (every 9600) near two thirds
SEEDS=(42 43)
EXTRA=()
if [ -n "${SMOKE:-}" ]; then
  ITERATIONS=$SMOKE
  SEEDS=(42)
  # the arm goals start at 600 steps whatever the gate says, so the switch-on path runs too
  EXTRA=(agent.agent.experiment.directory=sport_smoke env.warm_start_deadline_steps=600)
fi

# Kept from the deploy queue (queue.sh): capped LSTM cell, episodes drawn from an exponential (mean 90 s, cap 300 s),
# motor gains 0.8-1.2 and strength 0.85-1.15 drawn at reset and every 20-30 s, gait 0.70 s, fall gate 8%.
KEPT=(
  agent.models.policy.rnn.cell_cap=1.0
  env.episode_length_s=300.0 env.episode_length_mean_s=90.0 "env.actuator_redraw_interval_s=[20.0,30.0]"
  "env.events.actuator_scales.params.gain_range=[0.8,1.2]" "env.events.actuator_scales.params.strength_range=[0.85,1.15]"
  env.gait_period=0.7
  env.warm_start_max_fall_share=0.08
)
# Sport mode, from its bags (assets/h1_2.py SPORT_MODE_*):
#  - its PD gains on every joint, the arms' too; each action scale matched so an action unit asks for the torque it
#    did at our gains (the ablations' gains run, without that, marched in place)
#  - its standing pose (knee 0.70 rad, from 0.5) as the legs' default and the posture term's target, the gait-cycle
#    mean of hip pitch, knee and ankle pitch outside arm goals (-2: seed 42 sank to a 1.2 rad knee once the arm goals
#    started); base_height's target follows to that stance's 0.98 m
#  - its command ranges (full stick: vx 0.55, vy 0.54 m/s, wz 0.97 rad/s), with the turn weight 1.0 -> 1.5 (the
#    ablations' wide ranges cost turn tracking)
#  - its command low-pass, tau 0.5 s (alone: the ablations' cmd run also resampled every 1-3 s, which broke the
#    velocity curriculum)
#  - stepping on any input: the stand/walk threshold 0.03, a fifth of moving commands small (0.05-0.15)
#  - its 6 cm swing clearance (the clearance term's lift 0.055 -> 0.06 m)
#  - its steadier torso (roll/pitch sd 0.83/0.75 deg against our ~3.4/1.6): orientation -0.25 -> -10, ang_vel_xy
#    -0.05 -> -0.2 (at -0.25 a 3.4 deg sway cost 0.001 per second)
SPORT=(
  env.pd_gains=sport_all env.retune_action_scales=True
  env.default_leg_pose=sport env.rewards.legs.posture.weight=-2.0
  "env.commands.base_velocity.ranges.lin_vel_x=[-0.55,0.55]" "env.commands.base_velocity.ranges.lin_vel_y=[-0.54,0.54]"
  "env.commands.base_velocity.ranges.ang_vel_z=[-0.97,0.97]"
  "env.commands.base_velocity.pure_turn_speed=[0.2,0.97]" "env.commands.base_velocity.pure_lateral_speed=[0.15,0.54]"
  env.rewards.legs.track_ang_vel_z.weight=1.5
  env.commands.base_velocity.command_filter_tau=0.5
  env.zero_cmd_eps=0.03 env.commands.base_velocity.small_command_prob=0.2
  env.rewards.legs.feet_swing_clearance.params.lift_height=0.06
  env.rewards.legs.orientation.weight=-10.0 env.rewards.legs.ang_vel_xy.weight=-0.2
)
# The arm goals start on sport-mode tracking measured in training (LocoManipMarlEnv._sport_gate): turn sense >= 0.95,
# speed ratio 0.85-1.15, turn ratio 0.75-1.25, cos >= 0.90, angular error <= 0.10 rad/s, linear error <= 0.12 m/s,
# three checks in a row 100 steps apart, falls under 8%; or at 240k steps (2400 s). Mid360v2's scorecard: 0.988 /
# 0.973 / 0.835 / 0.945 / 0.056 / 0.068, without pushes or randomization; training's numbers will sit below it.
# To change a threshold while it trains: <run>/warm_start_gate.yaml, e.g. "warm_start_min_sport_cos: 0.85".
GATE=(env.warm_start_sport_gate=True env.warm_start_deadline_steps=240000)
# 100 Hz policy, physics 500 Hz (decimation 5; the 0-20 ms action delay and the contact window count 2 ms substeps,
# so they stay as they were). What counted policy steps now counts the same seconds:
#  - rollouts 24 -> 48 (0.48 s), discount 0.99 -> 0.995 and GAE lambda 0.95 -> 0.975 (0.99^0.5, 0.95^0.5)
#  - the arm-goal ramp 10k -> 20k steps (0.1 -> 0.5, the default: the slower 30k ramp did not stop seed 42's crash),
#    the terrain gate's row interval 8000 -> 16000, checkpoints and estimator saves every 9600
#  - the estimator's input window 25 frames -> 0.5 s (50 frames); its buffer 100 -> 50 steps, the windowed loss
#    50 -> 50 steps and a fit every 25 steps (each step still in two fits): at 100 steps the doubled input filled the
#    GPU to 23.8 of 24.5 GB in the smoke test
#  - the arm IK step per policy step 0.1 -> 0.05 rad
#  - the rate penalties on raw actions x4 (a smooth trajectory's step-to-step change halves, squared), and x the mean
#    squared change of each term's action scale (the same joint-target change costs what it did): legs action_rate
#    -0.01 x 4 x 0.622 = -0.025, ankle_action_rate -0.02 x 4 x 0.625 = -0.05, arms action_rate -0.1 x 4 x 2.224 =
#    -0.89; arms target_rate (joint targets) -20 x 4 = -80
#  - the push curriculum one level per 20 s of a tracked episode (90 s mean episodes climbed 4.5x slower per second)
HZ100=(
  env.decimation=5
  agent.agent.rollouts=48 agent.agent.discount_factor=0.995 agent.agent.gae_lambda=0.975
  env.commands.arm_targets.arm_goal_ramp_steps=20000 env.curriculum.terrain_levels.params.steps_per_row=16000
  agent.agent.experiment.checkpoint_interval=9600 env.estimator.save_every=9600
  env.odometry_history_s=0.5 env.estimator.buffer_steps=50 env.estimator.window_steps=50 env.estimator.train_every=25
  env.actions.arm_pos.max_ik_step=0.05
  env.rewards.legs.action_rate.weight=-0.025 env.rewards.legs.ankle_action_rate.weight=-0.05
  env.rewards.arms.action_rate.weight=-0.89 env.rewards.arms.target_rate.weight=-80.0
  env.curriculum.push_levels.params.seconds_per_level=20.0
)

note() { [ -z "${SMOKE:-}" ] && echo "$(date '+%F %T') $*" >> "$STATUS"; echo "$(date '+%F %T') $*"; }

for seed in "${SEEDS[@]}"; do
  name=Sport100Hz_s$seed
  [ -n "${SMOKE:-}" ] && name=SportSmoke_s$seed
  log=$OUT/$name.train.log
  note "$name: training (seed $seed, $ITERATIONS updates of 48)"
  python scripts/skrl/train.py --headless --name "$name" --seed "$seed" --max_iterations "$ITERATIONS" \
    "${KEPT[@]}" "${SPORT[@]}" "${GATE[@]}" "${HZ100[@]}" "${EXTRA[@]}" > "$log" 2>&1
  code=$?
  run=$(grep -o "Logging to /[^ ]*/[^ ]*_$name" "$log" | head -1 | sed 's/^Logging to //')
  kernels=$(grep -c "fail to launch kernel" "$log")
  final=$((ITERATIONS * 48))
  if [ $code -ne 0 ] || [ ! -f "$run/checkpoints/agent_$final.pt" ]; then
    note "$name: training FAILED (exit $code, kernel failures $kernels, run ${run:-?}); see $log"
    continue
  fi
  note "$name: trained ($run, kernel failures $kernels)"
  [ -n "${SMOKE:-}" ] && continue
  for step in $FINAL $MID; do
    python scripts/skrl/eval_scorecard.py --checkpoint "$run/checkpoints/agent_$step.pt" --headless \
      "env.events.actuator_scales.params.gain_range=[1.0,1.0]" "env.events.actuator_scales.params.strength_range=[1.0,1.0]" \
      > "$OUT/$name.scorecard_$step.log" 2>&1
    note "$name: scorecard agent_$step exit $? (kernel failures $(grep -c 'fail to launch kernel' "$OUT/$name.scorecard_$step.log"))"
  done
  export=$OUT/$name/export
  mkdir -p "$export"
  if python scripts/skrl/export_marl.py --checkpoint "$run/checkpoints/agent_$FINAL.pt" --out "$export" \
      > "$OUT/$name.export.log" 2>&1; then
    note "$name: exported to $export"
  else
    note "$name: export FAILED; see $OUT/$name.export.log"
  fi
done
note "sport queue finished"
