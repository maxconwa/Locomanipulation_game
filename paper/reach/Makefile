# Reach evaluation of the LocoManip-Marl-Direct-v0 policies: evaluate a run, then rebuild every table, figure,
# LaTeX number and the result page from the trials. Run from anywhere: make -C paper/reach <target> RUN=... LABEL=...
#
#   make all                                     every run in conditions.yaml without results, then the paper
#   make eval    RUN=<run dir> LABEL=lambda1     Isaac Lab: full policy, legs blind, arms IK only (3 seeds)
#   make sim2sim RUN=<run dir> LABEL=lambda1     MuJoCo: RoboCasa joint dynamics and the training plant's
#   make paper                                   analysis -> figures (with the crouch render) -> page
#   make smoke                                   pipeline self-test on a zero-action checkpoint (no run needed)
#   make baselines                               GOLEM's arm IK on the goal grid (needs pinocchio + pink)
#   make goals                                   rebuild the standing tables and the goal grid (changes every result)

PY_ISAAC ?= $(HOME)/miniconda3/envs/env_isaaclab51/bin/python
PY_IK    ?= $(HOME)/miniconda3/bin/python3
PY       ?= $(PY_ISAAC)
SEEDS    ?= 0,1,2
WORKERS  ?= 8
CKPT     ?= $(shell ls -v $(RUN)/checkpoints/agent_*.pt 2>/dev/null | tail -1)
# the robot assets: the checkout's submodule when it holds the USD, else the lab workstation's pinned clone
SUBMODULE_USD := $(abspath ../../third_party/CL_Assets/isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd)
export CL_ASSETS_DIR ?= $(if $(wildcard $(SUBMODULE_USD)),$(abspath ../../third_party/CL_Assets),$(HOME)/Programs/CL_Assets-cf87bfe)
export OMNI_KIT_ACCEPT_EULA = YES
export MUJOCO_GL ?= egl
# On the lab workstation every Isaac job goes through the memory guard; elsewhere it runs directly.
GUARD    ?= $(if $(wildcard $(HOME)/.claude/bin/resguard.sh),$(HOME)/.claude/bin/resguard.sh run --mem 12G --cpu 400 --,)

DEVICE   ?= cuda:0

.PHONY: all eval sim2sim paper analyze figures render page smoke baselines goals check-run

# every run in conditions.yaml without current results: Isaac (full, legs blind, arms IK), MuJoCo; then make paper
all:
	$(PY) tools/run_all.py --device $(DEVICE) --seeds $(SEEDS) --workers $(WORKERS)

check-run:
	@test -n "$(LABEL)" || (echo "set LABEL=<name>, e.g. LABEL=lambda1" && exit 1)
	@test -f "$(CKPT)" || (echo "no checkpoints/agent_<N>.pt under RUN=$(RUN)" && exit 1)
	@echo "checkpoint $(CKPT)"

eval: check-run
	$(GUARD) $(PY_ISAAC) tools/eval_isaac.py --checkpoint $(CKPT) --label $(LABEL) --seeds $(SEEDS)
	$(GUARD) $(PY_ISAAC) tools/eval_isaac.py --checkpoint $(CKPT) --label $(LABEL) --seeds $(SEEDS) --variant legs_blind
	$(GUARD) $(PY_ISAAC) tools/eval_isaac.py --checkpoint $(CKPT) --label $(LABEL) --seeds 0 --variant arms_ik

sim2sim: check-run
	$(PY) tools/eval_mujoco.py --checkpoint $(CKPT) --label $(LABEL) --workers $(WORKERS)
	$(PY) tools/eval_mujoco.py --checkpoint $(CKPT) --label $(LABEL) --workers $(WORKERS) --physics isaac

paper: analyze render figures page

analyze:
	$(PY) tools/analyze.py

render:
	$(PY) tools/render_robot.py --pose standing --out figures/render/render_standing.png
	$(PY) tools/render_robot.py --auto --out figures/render/render_crouch.png || true

figures:
	$(PY) tools/figures.py

page:
	$(PY) tools/build_page.py

smoke:
	$(PY) tools/smoke_checkpoint.py
	$(GUARD) $(PY_ISAAC) tools/eval_isaac.py --checkpoint results/_smoke_run/checkpoints/agent_0.pt --label _smoke \
		--odometry true --seeds 0 --limit 33 --zero_gravity
	$(PY) tools/eval_mujoco.py --checkpoint results/_smoke_run/checkpoints/agent_0.pt --label _smoke \
		--odometry true --limit 8 --workers 4

baselines:
	$(PY_IK) tools/ik_baselines.py --workers 4

goals:
	$(GUARD) $(PY_ISAAC) tools/build_tables.py --num_envs 1024 --seed 0
	$(PY) tools/make_goals.py --bases 96 --depths 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
