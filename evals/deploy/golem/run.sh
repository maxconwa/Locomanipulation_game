#!/usr/bin/env bash
# The deploy test in GOLEM: an export run by lmg_deploy's marl_deploy_node in GOLEM's RoboCasa sim, under the full
# bringup (lmg_sim.launch.py: the safety layer, FAST-LIO, ...), driven by THE command program
# (evals/common/program.py) on /cmd_vel, then scored (evals/deploy/golem/score.py).
#
#   bash evals/deploy/golem/run.sh <export dir> <out dir>       (out new or empty)
#
# Uses ~/GOLEM as it is checked out ($GOLEM): its tests/sim/common.sh sets the containers up (the RoboCasa sim and a
# ros container in its network namespace, no network of their own, the core_ws build cached, a DDS domain that is
# never 0), and its test_lmg_explore.py records the sim (the pelvis, the whole qpos, contacts, model.mjb). Nothing in
# GOLEM is changed: the export is staged into <out>/policy (its yaml as h12.yaml) and bind-mounted over
# core_ws/src/lmg_deploy/policies/h12 in the ros container, and this repo's evals/ is mounted beside it for the
# driver (drive.py). GPU needed (the sim's cameras and lidar render with EGL); about 5 minutes.
#
# Env: GOLEM (default ~/GOLEM), ROS_DOMAIN_ID (default 196), SIM_IMAGE, ROS_IMAGE (GOLEM's defaults),
#   IMU_CALIBRATION_FILE: mounted over lmg_deploy's config/imu_calibration.yaml (default: GOLEM's own, as checked out),
#   SCORE_PYTHON: the python that scores (mujoco, numpy, scipy, yaml; default python3).
# Exit: 0 when the drive and the scoring pass; 1 when a check fails or a step does; 2 bad arguments; 3 build failure.
set -eo pipefail
EXPORT=${1:?usage: run.sh <export dir> <out dir>}
OUT_ARG=${2:?usage: run.sh <export dir> <out dir>}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
GOLEM=${GOLEM:-$HOME/GOLEM}
EXPORT=$(cd "$EXPORT" && pwd)
[ -f "$EXPORT/export.yaml" ] || { echo "[lmg_evals] $EXPORT has no export.yaml" >&2; exit 2; }
[ -f "$GOLEM/tests/sim/common.sh" ] || { echo "[lmg_evals] no GOLEM checkout at $GOLEM" >&2; exit 2; }
if [ -n "${GOLEM_SIM_ODOM:-}" ]; then
    echo "[lmg_evals] unset GOLEM_SIM_ODOM: the deploy runs on FAST-LIO's odometry" >&2
    exit 2
fi
export OUT=$OUT_ARG
SIM_ENV=(-e GOLEM_SIM_ODOM=0 -e GOLEM_LIDAR=)
source "$GOLEM/tests/sim/common.sh"
sim_test_setup h12 sim lmg_evals lmg-evals program 196
sim_test_log_load

# the export, staged as policies/h12 (lmg_sim's launch reads policies/h12/h12.yaml)
mkdir -p "$OUT/policy"
cp -r "$EXPORT"/. "$OUT/policy/"
yaml=$(ls "$OUT"/policy/*.yaml | grep -v '/export.yaml$' | head -1)
[ -n "$yaml" ] || { echo "[lmg_evals] no deploy yaml in $EXPORT" >&2; exit 2; }
[ "$(basename "$yaml")" = h12.yaml ] || mv "$yaml" "$OUT/policy/h12.yaml"
cat > "$OUT/lmg_evals_compose.yml" <<EOF
# evals/deploy/golem/run.sh: the export over policies/h12, and this repo's evals/ for the driver
services:
  ros:
    volumes:
      - {type: bind, source: "$OUT/policy", target: /home/code/core_ws/src/lmg_deploy/policies/h12, read_only: true}
      - {type: bind, source: "$REPO/evals", target: /home/code/lmg_evals/evals, read_only: true}
EOF
if [ -n "${IMU_CALIBRATION_FILE:-}" ]; then
    cal=$(cd "$(dirname "$IMU_CALIBRATION_FILE")" && pwd)/$(basename "$IMU_CALIBRATION_FILE")
    printf '      - {type: bind, source: "%s", target: /home/code/core_ws/src/lmg_deploy/config/imu_calibration.yaml, read_only: true}\n' \
        "$cal" >> "$OUT/lmg_evals_compose.yml"
fi
COMPOSE+=(-f "$OUT/lmg_evals_compose.yml")
{
    echo "golem_commit: $(git -C "$GOLEM" rev-parse HEAD)"
    echo "golem_branch: $(git -C "$GOLEM" rev-parse --abbrev-ref HEAD)"
    echo "lmg_deploy_commit: $(git -C "$GOLEM/core_ws/src/lmg_deploy" log -1 --format=%H -- . 2>/dev/null)"
    echo "export: $EXPORT"
    echo "imu_calibration_file: ${IMU_CALIBRATION_FILE:-the one in GOLEM}"
} > "$OUT/lmg_evals.yaml"

echo "[lmg_evals] DDS domain $ROS_DOMAIN_ID, output in $OUT"
sim_test_build_ros
sim_test_start_sim test_lmg_explore.py "${SIM_ENV[@]}"
echo "[lmg_evals] sim up; starting lmg_sim"
sim_test_run_ros 1500 "ros2 launch golem_bringup lmg_sim.launch.py start_position_verified:=true use_rviz:=false > /out/launch.log 2>&1 &
    LAUNCH=\$!
    PYTHONPATH=/home/code/lmg_evals:\$PYTHONPATH python3 -u /home/code/lmg_evals/evals/deploy/golem/drive.py --out /out
    rc=\$?
    touch /out/stop
    for _ in \$(seq 1 60); do [ -e /out/contacts.json ] && break; sleep 1; done
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"
sim_test_stop_sim
# the legs' places in the recorded qpos, by the sim's own mujoco (model.mjb is its version's format)
"${COMPOSE[@]}" run --rm -T --no-deps robocasa python3 -c "
import json, mujoco
m = mujoco.MjModel.from_binary_path('/out/model.mjb')
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '' for j in range(m.njnt)]
json.dump({n: int(m.jnt_qposadr[j]) for j, n in enumerate(names) if n}, open('/out/joint_qpos.json', 'w'))
" > "$OUT/joint_qpos.log" 2>&1 || echo "[lmg_evals] could not read the joints' places (see joint_qpos.log)" >&2
# the containers wrote as root
"${COMPOSE[@]}" run --rm -T --no-deps --entrypoint chown robocasa -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
set +e
(cd "$REPO" && ${SCORE_PYTHON:-python3} -m evals.deploy.golem.score --out "$OUT") | tee "$OUT/score.log"
REPORT_RC=${PIPESTATUS[0]}
set -e
[ "$DRIVE_RC" = 0 ] && [ "$REPORT_RC" = 0 ]
