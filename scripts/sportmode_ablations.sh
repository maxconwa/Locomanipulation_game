#!/usr/bin/env bash
# Sport-mode ablations (docs/experiments/sportmode_ablations.md): 100k-step runs of LocoManip-Marl-Direct-v0, one at a
# time, each followed by its scorecards (scripts/skrl/eval_scorecard.py) at the last checkpoint and at 50.4k.
#
#   bash scripts/sportmode_ablations.sh [tag ...]       (default: every tag below, in order)
#
# Runs land in logs/skrl/locomanip_marl/<time>_sm_<tag>; logs, a status line per run and a kernel-failure check in
# logs/sportmode_ablations/. A tag whose run already finished (status "done") is skipped, so the queue can be restarted.
# SMOKE=<iterations> runs each tag that many updates into logs/skrl/sm_smoke/ and scores its last checkpoint only.
set -u
set -f  # no globbing: the overrides carry [ ] lists
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

OUT=logs/sportmode_ablations
# 4167 updates x 24 steps = 100,008 env steps; checkpoints (and the estimator) every 2400 steps, so 50,400 is one
ITERATIONS=4167
COMMON=(agent.agent.experiment.checkpoint_interval=2400 env.estimator.save_every=2400)
EXPERIMENT=locomanip_marl
if [ -n "${SMOKE:-}" ]; then
  ITERATIONS=$SMOKE
  EXPERIMENT=sm_smoke
  COMMON+=(agent.agent.experiment.directory=sm_smoke)
  OUT=$OUT/smoke
fi
FINAL_STEP=$((ITERATIONS * 24))
mkdir -p "$OUT"
STATUS="$OUT/status.tsv"
touch "$STATUS"

# 1 stand/walk gate: walk from a command norm of 0.03; 20% of moving commands of norm 0.05-0.15
GATE=(env.zero_cmd_eps=0.03 env.commands.base_velocity.small_command_prob=0.2)
# 2 command schedule: resampled every U(1, 3) s, low-passed with tau 0.5 s (observations and rewards both)
CMD=("env.commands.base_velocity.resampling_time_range=[1.0,3.0]" env.commands.base_velocity.command_filter_tau=0.5)
# 2 with the velocity-kernel curriculum at base's pace in seconds: it judges every 20,000 command segments, and
# resampling every U(1, 3) s instead of 10 s makes 5x as many (cmd reached the narrowest kernels by 12k steps)
CMD_TW=("${CMD[@]}" env.commands.base_velocity.kernel_window_segments=100000)
# 3 ranges: vy +-0.5, wz +-1.0 (the turn-in-place and sideways draws widened to match)
RANGE=("env.commands.base_velocity.ranges.lin_vel_y=[-0.5,0.5]" "env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0]"
       "env.commands.base_velocity.pure_turn_speed=[0.2,1.0]" "env.commands.base_velocity.pure_lateral_speed=[0.15,0.5]")
# 4 turn emphasis: yaw tracking weight 2.0 (as linear); the yaw kernel's last stage 0.2 -> 0.14 rad/s (-30%)
YAW=(env.rewards.legs.track_ang_vel_z.weight=2.0 "env.commands.base_velocity.ang_vel_stds=[1.5,1.0,0.7,0.5,0.4,0.3,0.14]")
# 5 gait period 0.70 s (the tracking windows follow it)
GAIT=(env.gait_period=0.7)
# 6 sport-mode PD gains for legs, feet and torso
GAINS=(env.pd_gains=sport)
# 7 100 Hz policy: decimation 5 of the 500 Hz physics (the action delay and the contact window count 2 ms substeps,
# so they stay 0-20 ms and 6 ms); the per-step rate penalties x4 and the arm IK step /2, so a given joint-target
# trajectory costs and moves as at 50 Hz; discount and GAE lambda for the same horizon in seconds (0.99^0.5, 0.95^0.5)
HZ100=(env.decimation=5 env.actions.arm_pos.max_ik_step=0.05
       env.rewards.legs.action_rate.weight=-0.04 env.rewards.legs.ankle_action_rate.weight=-0.08
       env.rewards.arms.action_rate.weight=-0.4 env.rewards.arms.target_rate.weight=-80.0
       agent.agent.discount_factor=0.995 agent.agent.gae_lambda=0.975)

ALL_TAGS=(base base_s2 gate cmd range yaw gait070 gains hz100 combo combo_all cmd_tw combo_tw)
overrides() {
  case "$1" in
    base) echo "";;
    base_s2) echo "--seed 43";;
    gate) echo "${GATE[*]}";;
    cmd) echo "${CMD[*]}";;
    range) echo "${RANGE[*]}";;
    yaw) echo "${YAW[*]}";;
    gait070) echo "${GAIT[*]}";;
    gains) echo "${GAINS[*]}";;
    hz100) echo "${HZ100[*]}";;
    combo) echo "${GATE[*]} ${CMD[*]} ${RANGE[*]} ${YAW[*]} ${GAIT[*]}";;
    combo_all) echo "${GATE[*]} ${CMD[*]} ${RANGE[*]} ${YAW[*]} ${GAIT[*]} ${GAINS[*]} ${HZ100[*]}";;
    cmd_tw) echo "${CMD_TW[*]}";;
    combo_tw) echo "${GATE[*]} ${CMD_TW[*]} ${RANGE[*]} ${YAW[*]} ${GAIT[*]}";;
    *) echo "unknown tag $1" >&2; return 1;;
  esac
}

kernel_check() {  # $1 log
  local n
  n=$(grep -c "fail to launch kernel" "$1")
  [ "$n" -eq 0 ] && echo ok || echo "FAIL($n)"
}

TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=("${ALL_TAGS[@]}")
for tag in "${TAGS[@]}"; do
  args=$(overrides "$tag") || continue
  if awk -F'\t' -v t="$tag" '$1 == t && $2 == "done"' "$STATUS" | grep -q .; then
    echo "[queue] $tag: done already, skipped"; continue
  fi
  log="$OUT/train_$tag.log"
  echo "[queue] $(date '+%F %T') $tag: training ($args)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  python scripts/skrl/train.py --headless --name "sm_$tag" --max_iterations "$ITERATIONS" "${COMMON[@]}" $args > "$log" 2>&1
  code=$?
  hours=$(awk -v s="$start" -v e="$(date +%s)" 'BEGIN { printf "%.2f", (e - s) / 3600 }')
  run=$(grep -o "Logging to /[^ ]*$EXPERIMENT/[^ ]*_sm_$tag" "$log" | head -1 | sed 's/^Logging to //')
  kernels=$(kernel_check "$log")
  final="$run/checkpoints/agent_$FINAL_STEP.pt"
  if [ $code -ne 0 ] || [ -z "$run" ] || [ ! -f "$final" ]; then
    printf '%s\tfailed\t%s\t%s h\texit %s\tkernels %s\n' "$tag" "${run:-?}" "$hours" "$code" "$kernels" >> "$STATUS"
    echo "[queue] $tag: FAILED (exit $code, kernels $kernels), see $log"
    continue
  fi
  echo "[queue] $(date '+%F %T') $tag: trained in $hours h (kernels $kernels); scoring"
  scores=""
  checkpoints=("$final")
  [ -z "${SMOKE:-}" ] && checkpoints+=("$run/checkpoints/agent_50400.pt")
  for ckpt in "${checkpoints[@]}"; do
    elog="$OUT/eval_${tag}_$(basename "$ckpt" .pt).log"
    python scripts/skrl/eval_scorecard.py --checkpoint "$ckpt" --headless > "$elog" 2>&1
    ecode=$?
    scores="$scores $(basename "$ckpt" .pt):exit$ecode,kernels-$(kernel_check "$elog")"
  done
  printf '%s\tdone\t%s\t%s h\texit 0\tkernels %s\t%s\n' "$tag" "$run" "$hours" "$kernels" "$scores" >> "$STATUS"
  echo "[queue] $(date '+%F %T') $tag: done ($scores)"
done
echo "[queue] $(date '+%F %T') finished"
