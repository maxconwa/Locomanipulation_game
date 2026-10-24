#!/usr/bin/env bash
# CPU checks for the sport-mode queue (sport_queue.sh, which predates the MuJoCo evals' 100 Hz support, 5217e87):
# waits for each seed's export, then runs the MuJoCo stress tests (standard, 180 s stands, a small command after a
# 60 s stand) and the replay of both Mid360v2 failure bags beside the deployed Mid360v2, as queue.sh's cpu_checks.
#
#   bash logs/deploy_candidates/sport_checks.sh            (tmux sportchecks)
set -u
cd "$(dirname "$0")/../.."
OUT=$PWD/logs/deploy_candidates
STATUS=$OUT/status.txt
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
GOLEM_POLICIES=$HOME/GOLEM/core_ws/src/lmg_deploy/policies
BAGS=$HOME/bags/mid360v2_failure_2026-10-20

note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

for seed in 42 43; do
  name=Sport100Hz_s$seed
  until grep -q "$name: exported to\|$name: training FAILED\|$name: export FAILED\|sport queue finished" "$STATUS"; do
    sleep 300
  done
  grep -q "$name: exported to" "$STATUS" || continue
  export=$OUT/$name/export
  cp agent_scripts/hw_deploy_template.yaml "$export/deploy.yaml"
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
done
