#!/usr/bin/env bash
# Training ablations for the Mid360v2 real-robot failure (2026-10-20): one factor at a time on Mid360v2's setup
# (Explicit500's launch overrides), 48k steps each, then each run exported and put through the MuJoCo stress tests
# (agent_scripts/hw_stress_mujoco.py). Sequential on one GPU, about 3 h per run.
#
#   bash agent_scripts/hw_ablations.sh [RUN ...]      # default: all of RUNS in order
#
# Runs go to logs/skrl/locomanip_marl/<time>_HWAbl_<RUN>, exports to logs/hw_ablations/<RUN>/export, stress results
# to logs/hw_stress_mujoco/HWAbl_<RUN>{,_long}. Progress in logs/hw_ablations/status.txt.
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
ITER=${ITER:-2000}   # x 24 rollouts: 48k steps by default; ITER=8334 is a full 200k run
PREFIX=${PREFIX-HWAbl_}   # run directory name = <time>_<PREFIX><RUN>
# the deploy yaml the stress tests read (control_dt, gains; export.yaml overrides the rest): Mid360v2's GOLEM lmg_deploy config
TEMPLATE=${TEMPLATE:-$REPO/agent_scripts/hw_deploy_template.yaml}
# conda: the Isaac Lab env trains and exports, an env with mujoco + torch + pyyaml runs the MuJoCo stress tests
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
ISAAC_ENV=${ISAAC_ENV:-env_isaaclab}
MUJOCO_ENV=${MUJOCO_ENV:-base}

BASE=(
  env.commands.arm_targets.table_file=$REPO/logs/skrl/locomanip_marl/golemlim120k/arm_target_tables_v3.pt
  env.observation_scales=null env.explicit_pd=True env.sim.dt=0.002 env.decimation=10
  env.action_delay_range=[0,10] env.action_delay_substeps=True
  env.rewards.legs.feet_contact_forces.params.peak_over_substeps=True env.rewards.legs.feet_contact_forces.params.window=3
  env.events.scale_link_masses=null env.events.shift_pelvis_com=null
  env.events.add_torso_mass.params.mass_distribution_params=[-2.0,4.0]
  env.scene.terrain.max_init_terrain_level=0 env.scene.terrain.terrain_generator.sub_terrains.rough.height_at_0=0.0
  env.curriculum.terrain_levels.params.gated=True
)
ACTUATOR=(env.events.actuator_scales.params.gain_range=[0.8,1.2] env.events.actuator_scales.params.strength_range=[0.85,1.15])
LONG=(env.episode_length_s=60.0)
GATE60=(env.warm_start_max_fall_share=0.06)
STANDING=(env.commands.base_velocity.rel_standing_envs=0.2 env.commands.base_velocity.small_command_prob=0.3
          env.rewards.legs.stand_load_balance.weight=-2.0)
# (escaped: ARGS goes through eval, and Hydra needs the quotes around the braces)
SENSING=("env.scene.imu.prim_path=\"'{ENV_REGEX_NS}/Robot/imu_link'\"" env.gyro_lowpass_hz=null
         env.events.imu_mount.params.pitch_deg=2.0 env.joint_vel_obs_lag_ms=[0.0,15.0])

declare -A ARGS=(
  [base]=""
  [actuator]="${ACTUATOR[*]}"
  [long]="${LONG[*]}"
  [standing]="${STANDING[*]}"
  [sensing]="${SENSING[*]}"
  [all]="${ACTUATOR[*]} ${LONG[*]} ${STANDING[*]} ${SENSING[*]}"
  # the deployable candidate (run with ITER=8334 PREFIX=): the two changes that helped, nothing else, with the
  # warm start's fall gate scaled to 60 s episodes (2% of 20 s episodes = 6% of 60 s ones at the same fall rate;
  # at 2% the first try was still walking-only at 73k steps with 9.4% of its episodes ending in a fall)
  [LongAct200k]="${ACTUATOR[*]} ${LONG[*]} ${GATE60[*]}"
  # its twin with MLP actors (no LSTM, so no memory to carry past the training horizon)
  [LongActMLP200k]="${ACTUATOR[*]} ${LONG[*]} ${GATE60[*]} \"~agent.models.policy.rnn\""
  # and with the LSTM's cell state squashed after every step, c <- tanh(c) (recurrent.py cell_cap 1): it cannot count
  # time past the training episode the way Mid360v2's did (mean |c| 4 -> 11 over 180 s); h stays o tanh(c)
  [LongActCap200k]="${ACTUATOR[*]} ${LONG[*]} ${GATE60[*]} agent.models.policy.rnn.cell_cap=1.0"
)
RUNS=(base actuator long standing sensing all)
[ $# -gt 0 ] && RUNS=("$@")

mkdir -p logs/hw_ablations
STATUS=logs/hw_ablations/status.txt
source "$CONDA_SH"

for run in "${RUNS[@]}"; do
  name=$PREFIX$run
  echo "$(date '+%F %T') $run: training" >> $STATUS
  conda activate "$ISAAC_ENV"
  set -f  # the [a,b] overrides are not globs
  eval python scripts/skrl/train.py --headless --name $name --max_iterations $ITER "${BASE[@]}" ${ARGS[$run]} \
    > logs/hw_ablations/$run.train.log 2>&1
  code=$?
  set +f
  dir=$(ls -dt logs/skrl/locomanip_marl/*_"$name" 2>/dev/null | head -1)
  # the last step's checkpoint (numeric sort on the step after agent_)
  ckpt=$(ls "$dir"/checkpoints/agent_*.pt 2>/dev/null | awk -F'agent_' '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
  if [ $code -ne 0 ] || [ ! -f "$ckpt" ]; then
    echo "$(date '+%F %T') $run: training failed (exit $code, $ckpt)" >> $STATUS
    continue
  fi
  echo "$(date '+%F %T') $run: trained ($dir); exporting" >> $STATUS
  out=logs/hw_ablations/$run/export
  mkdir -p "$out"
  python scripts/skrl/export_marl.py --checkpoint "$ckpt" --out "$out" > logs/hw_ablations/$run.export.log 2>&1 \
    && cp "$TEMPLATE" "$out/deploy.yaml" \
    || { echo "$(date '+%F %T') $run: export failed" >> $STATUS; continue; }
  conda activate "$MUJOCO_ENV"
  # the stress tests on CPU, in the background while the next run trains
  (
    python agent_scripts/hw_stress_mujoco.py --export "$out" --label $name --seeds 8 --workers 12 \
      > logs/hw_ablations/$run.stress.log 2>&1
    python agent_scripts/hw_stress_mujoco.py --export "$out" --label ${name}_long \
      --conditions nominal,hw_no_gyro,hw,hang,hang_hw --profiles zero --seeds 8 --seconds 180 --workers 12 \
      > logs/hw_ablations/$run.stress_long.log 2>&1
    echo "$(date '+%F %T') $run: stress tests done (exit $?)" >> $STATUS
  ) &
done
wait
echo "$(date '+%F %T') all done" >> $STATUS
