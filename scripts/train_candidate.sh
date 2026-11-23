#!/usr/bin/env bash
# Train a deploy candidate and check it: for each seed, train (the defaults are the recipe), score it on the
# sport-mode scorecard (and the wrist phase) at the last checkpoint and at MID, export it, check the deploy controller
# against the env (deploy_parity, which fails over its tolerance), then on the CPU, while the next seed trains: the MuJoCo stress tests (standard,
# 180 s stands, a small command after a 60 s stand) and the replay of the Mid360v2 failure bags beside the deployed
# Mid360v2. One Isaac process at a time on the GPU.
#
#   bash scripts/train_candidate.sh [name] [seed ...]           (default: Candidate, seeds 42 43; e.g. in tmux)
#   SMOKE=<updates> bash scripts/train_candidate.sh              (one seed, into logs/skrl/smoke, no checks)
#   ITERATIONS=<updates> MID=<step> OVERRIDES="<hydra args>" ...  (a shorter run, its scorecard step, its overrides)
#
# Exits 1 when any step of any seed failed (each is a FAILED line in the status file). Status lines in
# logs/candidates/status.txt; per run: logs/candidates/<name>_s<seed>/ (logs, the export in a folder
# of the same name, parity, replays);
# stress results in logs/hw_stress_mujoco/<name>_s<seed>*. Isaac in conda env_isaaclab; the CPU checks in an env with
# mujoco, torch, pin, pyyaml and rosbags (CPU_ENV, default base; ROSBAGS_PATH is put on its PYTHONPATH if set).
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
OUT=$REPO/logs/candidates
STATUS=$OUT/status.txt
CONDA_SH=${CONDA_SH:-$( [ -f $HOME/miniconda3/etc/profile.d/conda.sh ] && echo $HOME/miniconda3 || echo $HOME/anaconda3)/etc/profile.d/conda.sh}
CPU_ENV=${CPU_ENV:-base}
GOLEM=${GOLEM:-$HOME/GOLEM}
BAGS=${BAGS:-$HOME/bags/mid360v2_failure_2026-10-20}
ROSBAGS_PATH=${ROSBAGS_PATH:-}
source "$CONDA_SH"
mkdir -p "$OUT"

NAME=${1:-Candidate}
shift || true
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(42 43); fi
ITERATIONS=${ITERATIONS:-12500}   # x 48 = 600,000 policy steps (6000 s per env at 100 Hz)
read -r -a EXTRA <<< "${OVERRIDES:-}"
if [ -n "${SMOKE:-}" ]; then
  ITERATIONS=$SMOKE
  SEEDS=(${SEEDS[0]})
  # the arm goals start at 600 steps whatever the gate says, so the switch-on path runs too
  # (the curriculum state is saved with every checkpoint)
  EXTRA+=(agent.agent.experiment.directory=smoke env.warm_start_deadline_steps=600
          agent.agent.experiment.checkpoint_interval=480)
fi
FINAL=$((ITERATIONS * 48))
MID=${MID:-$((FINAL * 2 / 3 / 9600 * 9600))}   # the checkpoint (every 9600) at or before two thirds: 393600 of 600000
FAILED=$OUT/.failed_$NAME.$$   # one line per failed step, the background CPU checks' too
: > "$FAILED"

note() { [ -z "${SMOKE:-}" ] && echo "$(date '+%F %T') $*" >> "$STATUS"; echo "$(date '+%F %T') $*"; }
fail() { echo "$*" >> "$FAILED"; note "$* FAILED"; }

cpu_checks() {  # $1 run name, $2 export dir, $3 out dir
  local name=$1 export=$2 dir=$3
  local bad=""
  conda activate "$CPU_ENV"
  python -m evals.deploy.stress_mujoco --export "$export" --label "$name" --seeds 8 --workers 12 \
    > "$dir/stress.log" 2>&1 || bad+=" stress.log"
  python -m evals.deploy.stress_mujoco --export "$export" --label "${name}_long" \
    --conditions nominal,hw_no_gyro,hw,hang,hang_hw --profiles zero --seeds 8 --seconds 180 --workers 12 \
    > "$dir/stress_long.log" 2>&1 || bad+=" stress_long.log"
  python -m evals.deploy.stress_mujoco --export "$export" --label "${name}_small60" \
    --conditions nominal,hw,hang,hang_hw,strength085 --profiles small60 --seeds 8 --seconds 80 --workers 12 \
    > "$dir/stress_small60.log" 2>&1 || bad+=" stress_small60.log"
  # the export gate's engage from the robot's pre-pose (the pp_* grid: hung, settled, the 2026-11-20 bag's sagged
  # pose, random draws; rope kept and released; hw and the torso IMU's vibration; the quaternion bias range's ends)
  python -m evals.deploy.stress_mujoco --export "$export" --label "${name}_prepose" --conditions pp \
    --plants robocasa --seeds 4 --workers 12 > "$dir/stress_prepose.log" 2>&1 || bad+=" stress_prepose.log"
  for bag in mid360v2_deploy mid360v2_deploy_2; do
    PYTHONPATH=$ROSBAGS_PATH python -m evals.deploy.bag_replay --bag "$BAGS/$bag" --deployed mid360v2 \
      --golem "$GOLEM" --policy "mid360v2=logs/deploy_baselines/mid360v2" --policy "$name=$export" \
      --sim_seeds 4 --out "$dir/replay_$bag" > "$dir/replay_$bag.log" 2>&1 || bad+=" replay_$bag.log"
  done
  # (a check that ran says nothing here about what it found: read its summary)
  if [ -n "$bad" ]; then
    fail "$name: CPU checks that did not run to the end (see $dir/):$bad"
  else
    note "$name: CPU checks ran (stress: logs/hw_stress_mujoco/$name*, replays: $dir/replay_*)"
  fi
}

for seed in "${SEEDS[@]}"; do
  name=${NAME}_s$seed
  dir=$OUT/$name
  mkdir -p "$dir"
  conda activate env_isaaclab
  note "$name: training (seed $seed, $ITERATIONS updates)"
  python scripts/skrl/train.py --headless --name "$name" --seed "$seed" --max_iterations "$ITERATIONS" "${EXTRA[@]}" \
    > "$dir/train.log" 2>&1
  code=$?
  run=$(grep -ao "Logging to /[^ ]*_$name" "$dir/train.log" | head -1 | sed 's/^Logging to //')
  kernels=$(grep -ac "fail to launch kernel" "$dir/train.log")
  if [ $code -ne 0 ] || [ ! -f "$run/checkpoints/agent_$FINAL.pt" ]; then
    fail "$name: training (exit $code, kernel failures $kernels, run ${run:-?}; see $dir/train.log)"
    continue
  fi
  note "$name: trained ($run, kernel failures $kernels)"
  [ -n "${SMOKE:-}" ] && continue
  for step in $FINAL $MID; do
    # the motors nominal, as the reference scorecards were taken
    python -m evals.isaac.legs.get_scorecard --checkpoint "$run/checkpoints/agent_$step.pt" --headless \
      "env.events.actuator_scales.params.gain_range=[1.0,1.0]" "env.events.actuator_scales.params.strength_range=[1.0,1.0]" \
      > "$dir/scorecard_$step.log" 2>&1
    code=$?
    kernels=$(grep -ac 'fail to launch kernel' "$dir/scorecard_$step.log")
    if [ $code -ne 0 ]; then
      fail "$name: scorecard agent_$step (exit $code, kernel failures $kernels; see $dir/scorecard_$step.log)"
    else
      note "$name: scorecard agent_$step done (kernel failures $kernels)"
    fi
  done
  export=$dir/$name   # a candidate export; lmg_deploy's policies/h12 takes one only when chosen
  if python scripts/skrl/export_marl.py --checkpoint "$run/checkpoints/agent_$FINAL.pt" --out "$export" \
      > "$dir/export.log" 2>&1; then
    python -m evals.deploy.parity --checkpoint "$run/checkpoints/agent_$FINAL.pt" --export "$export" \
      --deploy_pkg "$GOLEM/core_ws/src/lmg_deploy" --out "$dir/parity" > "$dir/parity.log" 2>&1 \
      || fail "$name: deploy parity (over its tolerance, or it did not run; see $dir/parity.log)"
    note "$name: exported to $export; CPU checks start in the background"
    (cpu_checks "$name" "$export" "$dir") &
  else
    fail "$name: export (see $dir/export.log)"
  fi
done
wait
failures=$(wc -l < "$FAILED")
rm -f "$FAILED"
note "$NAME: finished, $failures failed step(s)"
[ "$failures" -eq 0 ]
