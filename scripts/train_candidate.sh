#!/usr/bin/env bash
# Train a deploy candidate and check it: for each seed, train (the defaults are the recipe), score its legs on the
# Isaac scorecard (evals/isaac/legs) at the last checkpoint and at MID, export it, then the deploy gate
# (evals/deploy/gate.py, thresholds in evals/deploy/gate.yaml): parity on the GPU in line, then on the CPU, while the
# next seed trains, the MuJoCo stress grids, the pre-pose engage grid, the bag replays and the metric code; with
# GOLEM_RUN=1 also the run in GOLEM's RoboCasa sim (docker, GPU) in line. One Isaac process at a time on the GPU.
#
#   bash scripts/train_candidate.sh [name] [seed ...]           (default: Candidate, seeds 42 43; e.g. in tmux)
#   SMOKE=<updates> bash scripts/train_candidate.sh              (one seed, into logs/skrl/smoke, no checks)
#   ITERATIONS=<updates> MID=<step> OVERRIDES="<hydra args>" ...  (a shorter run, its scorecard step, its overrides)
#
# Exits 1 when any step of any seed failed (each is a FAILED line in the status file). Status lines in
# logs/candidates/status.txt; per run: logs/candidates/<name>_s<seed>/ (logs, the export in a folder of the same
# name, the gate's reports gate_gpu/ and gate_cpu/ with each check's output); scorecards in logs/evals/<run>/. A gate
# FAIL or NOT RUN is a FAILED line. Isaac in conda env_isaaclab; the gate in an env with mujoco, torch, scipy and
# pyyaml (CPU_ENV, default base); its bag replay runs in GOLEM's ros container (docker). GOLEM: $GOLEM, kept on main.
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
OUT=$REPO/logs/candidates
STATUS=$OUT/status.txt
CONDA_SH=${CONDA_SH:-$( [ -f $HOME/miniconda3/etc/profile.d/conda.sh ] && echo $HOME/miniconda3 || echo $HOME/anaconda3)/etc/profile.d/conda.sh}
CPU_ENV=${CPU_ENV:-base}
GOLEM=${GOLEM:-$HOME/GOLEM}
# the bags the gate replays: evals/deploy/gate.yaml bag_replay
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
  # a checkpoint (and the curriculum state saved with it) every 480 steps
  EXTRA+=(agent.agent.experiment.directory=smoke agent.agent.experiment.checkpoint_interval=480)
fi
FINAL=$((ITERATIONS * 48))
MID=${MID:-$((FINAL * 2 / 3 / 9600 * 9600))}   # the checkpoint (every 9600) at or before two thirds: 393600 of 600000
FAILED=$OUT/.failed_$NAME.$$   # one line per failed step, the background CPU checks' too
: > "$FAILED"

note() { [ -z "${SMOKE:-}" ] && echo "$(date '+%F %T') $*" >> "$STATUS"; echo "$(date '+%F %T') $*"; }
fail() { echo "$*" >> "$FAILED"; note "$* FAILED"; }

cpu_checks() {  # $1 run name, $2 checkpoint, $3 export dir, $4 out dir: the gate's CPU checks
  local name=$1 ckpt=$2 export=$3 dir=$4
  conda activate "$CPU_ENV"
  if GOLEM=$GOLEM python -m evals.deploy.gate --checkpoint "$ckpt" --export "$export" --out "$dir/gate_cpu" \
      --only metric_code,stress,bag_replay > "$dir/gate_cpu.log" 2>&1; then
    note "$name: deploy gate, CPU checks PASS ($dir/gate_cpu/report.md)"
  else
    fail "$name: deploy gate, CPU checks (see $dir/gate_cpu/report.md)"
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
    # the pinned conditions (evals/common/conditions.py: nominal motors, 84 envs), into logs/evals/<run>/isaac/legs
    python -m evals.isaac.legs.get_scorecard --checkpoint "$run/checkpoints/agent_$step.pt" --headless \
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
    gpu_checks=parity
    [ "${GOLEM_RUN:-0}" = 1 ] && gpu_checks=parity,golem
    (conda activate "$CPU_ENV" && GOLEM=$GOLEM python -m evals.deploy.gate --checkpoint "$run/checkpoints/agent_$FINAL.pt" \
      --export "$export" --out "$dir/gate_gpu" --only "$gpu_checks" > "$dir/gate_gpu.log" 2>&1) \
      || fail "$name: deploy gate, $gpu_checks (see $dir/gate_gpu/report.md)"
    note "$name: exported to $export; the gate's CPU checks start in the background"
    (cpu_checks "$name" "$run/checkpoints/agent_$FINAL.pt" "$export" "$dir") &
  else
    fail "$name: export (see $dir/export.log)"
  fi
done
wait
failures=$(wc -l < "$FAILED")
rm -f "$FAILED"
note "$NAME: finished, $failures failed step(s)"
[ "$failures" -eq 0 ]
