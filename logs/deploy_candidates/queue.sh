#!/usr/bin/env bash
# The deploy candidate (2026-10-24): the synthesis of the sport-mode ablations and the Mid360v2 failure study, two
# seeds of 300k env steps, each then scored, exported, stress-tested and replayed on the 10-20 failure bags.
#
#   bash logs/deploy_candidates/queue.sh            (tmux deploycand)
#
# Per seed: train (GPU) -> sport-mode scorecard at 300k and 196.8k with nominal motors (GPU, before the next seed
# trains: one Isaac process at a time) -> export -> in the background on CPU, while the next seed trains: MuJoCo
# stress tests (standard, 180 s stands, a small command after a 60 s stand) and the bag replay on both failure bags.
# Status lines in logs/deploy_candidates/status.txt.
set -u
set -f  # the [a,b] overrides are not globs
cd "$(dirname "$0")/../.."
REPO=$PWD
OUT=$REPO/logs/deploy_candidates
STATUS=$OUT/status.txt
source ~/anaconda3/etc/profile.d/conda.sh

ITERATIONS=12500   # x 24 = 300,000 env steps
FINAL=300000
# cap: no slow latch can charge (bag 1); 60 s episodes + motor randomization down to 0.75 strength: the failure
# study's robust pair, widened to bag 2's weak-motor drift; load balance: both bags' lopsided stance; gait 0.70 s:
# the sport-mode ablations' one tracking win; gate 0.03 + small commands: bag 1 failed on a small command after a
# stand; arm goals ramped 0.05 -> 0.5 over 30k steps: every LSTM run crashed at the switch-on; warm-start fall
# gate scaled to the drawn episode lengths.
# Episode lengths (10-24): drawn per episode from an exponential, mean 90 s, at least 2 s, capped at 300 s
# (episode_length_s), so no network can learn when its episode ends (the robot never resets; the failure study saw
# behaviour change at the 20 s end, and the arms train without time-out bootstrapping); the motor draw repeats every
# 20-30 s so long episodes keep the randomization. Warm-start fall gate: 6% of 60 s episodes is about 0.001 falls/s;
# with exponential lengths of mean 90 s that is a fall share of 0.001 / (0.001 + 1/90) = 8%
OVERRIDES=(
  agent.models.policy.rnn.cell_cap=1.0
  env.episode_length_s=300.0 env.episode_length_mean_s=90.0 "env.actuator_redraw_interval_s=[20.0,30.0]"
  "env.events.actuator_scales.params.gain_range=[0.8,1.2]" "env.events.actuator_scales.params.strength_range=[0.75,1.15]"
  env.rewards.legs.stand_load_balance.weight=-2.0
  env.gait_period=0.7
  env.zero_cmd_eps=0.03 env.commands.base_velocity.small_command_prob=0.2
  env.commands.arm_targets.arm_goal_ramp_start=0.05 env.commands.arm_targets.arm_goal_ramp_steps=30000
  env.warm_start_max_fall_share=0.08
)
# the scorecard (one long episode: it zeroes episode_length_mean_s) with the motors nominal, as Mid360v2's 0.988 / 0.973 / 0.835 / 0.945 / 0.056 / 0.068 was scored
NOMINAL_MOTORS=("env.events.actuator_scales.params.gain_range=[1.0,1.0]" "env.events.actuator_scales.params.strength_range=[1.0,1.0]")
GOLEM_POLICIES=$HOME/GOLEM/core_ws/src/lmg_deploy/policies
BAGS=$HOME/bags/mid360v2_failure_2026-10-20

note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

cpu_checks() {  # $1 name, $2 export dir
  local name=$1 export=$2
  conda activate base
  python agent_scripts/hw_stress_mujoco.py --export "$export" --label "$name" --seeds 8 --workers 12 \
    > "$OUT/$name.stress.log" 2>&1
  python agent_scripts/hw_stress_mujoco.py --export "$export" --label "${name}_long" \
    --conditions nominal,hw_no_gyro,hw,hang,hang_hw --profiles zero --seeds 8 --seconds 180 --workers 12 \
    > "$OUT/$name.stress_long.log" 2>&1
  python agent_scripts/hw_stress_mujoco.py --export "$export" --label "${name}_small60" \
    --conditions nominal,hw,hang,hang_hw,strength085 --profiles small60 --seeds 8 --seconds 80 --workers 12 \
    > "$OUT/$name.stress_small60.log" 2>&1
  for bag in mid360v2_deploy mid360v2_deploy_2; do
    PYTHONPATH=$OUT/pylib python agent_scripts/hw_bag_replay.py --bag "$BAGS/$bag" --deployed mid360v2 \
      --policy "mid360v2=$GOLEM_POLICIES/mid360v2" --policy "$name=$export" --sim_seeds 4 \
      --out "$OUT/$name/replay_$bag" > "$OUT/$name.replay_$bag.log" 2>&1
  done
  note "$name: CPU checks done (stress: logs/hw_stress_mujoco/$name*, replay: $OUT/$name/replay_*)"
}

for seed in 42 43; do
  name=DeployRand300k_s$seed
  log=$OUT/$name.train.log
  conda activate env_isaaclab
  note "$name: training (seed $seed, $ITERATIONS updates)"
  python scripts/skrl/train.py --headless --name "$name" --seed "$seed" --max_iterations "$ITERATIONS" \
    "${OVERRIDES[@]}" > "$log" 2>&1
  code=$?
  run=$(grep -o "Logging to /[^ ]*locomanip_marl/[^ ]*_$name" "$log" | head -1 | sed 's/^Logging to //')
  kernels=$(grep -c "fail to launch kernel" "$log")
  ckpt=$run/checkpoints/agent_$FINAL.pt
  if [ $code -ne 0 ] || [ ! -f "$ckpt" ]; then
    note "$name: training FAILED (exit $code, kernel failures $kernels, run ${run:-?}); see $log"
    continue
  fi
  note "$name: trained ($run, kernel failures $kernels); scoring"
  for step in $FINAL 196800; do
    python scripts/skrl/eval_scorecard.py --checkpoint "$run/checkpoints/agent_$step.pt" --headless \
      "${NOMINAL_MOTORS[@]}" > "$OUT/$name.scorecard_$step.log" 2>&1
    note "$name: scorecard agent_$step exit $? (kernel failures $(grep -c 'fail to launch kernel' "$OUT/$name.scorecard_$step.log"))"
  done
  export=$OUT/$name/export
  mkdir -p "$export"
  if python scripts/skrl/export_marl.py --checkpoint "$ckpt" --out "$export" > "$OUT/$name.export.log" 2>&1; then
    cp agent_scripts/hw_deploy_template.yaml "$export/deploy.yaml"
    note "$name: exported to $export; CPU checks start in the background"
    (cpu_checks "$name" "$export") &
  else
    note "$name: export FAILED; see $OUT/$name.export.log"
  fi
done
wait
note "queue finished"
