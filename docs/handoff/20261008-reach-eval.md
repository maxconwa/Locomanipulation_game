# Hand-off: reach evaluation of the lambda runs (2026-10-08)

Page: paper/20261008-reach_cooperation.html (https://claude.ai/artifact/LHVKbdmtDGtoQpmfghMZBn, v5 at 18:28).
Pipeline and commands: paper/reach/README.md. Local marl-direct is rebased on origin/marl-direct 65708f5 (paper1)
with the paper/reach commits on top; NOT pushed: this machine has no GitHub credentials for maxconwa (memory
locomanip-push-access), so the coworker gets a git bundle.

State (18:35): conditions.yaml l1 = paper1, l0.5 = paper05, l0 = paper00 (all agent_91200), whole = not trained.
- Done (Isaac, CPU physics, seed 0): l0.5 full, l0 full; MuJoCo l0.5 (RoboCasa, training joints, legs blind), l0
  (RoboCasa, training joints). GOLEM IK kinematic.
- Fixed at 18:10: eval_isaac logged `alive` by reference, so on CPU a trial that fell at any time counted as fallen
  from step 0. l0.5 full (6 falls) is barely affected; l0.5 legs_blind (746 falls) is being re-run.
- Running: paper/reach/logs/queue_paper1.sh: l1 full (ends ~18:50) -> l1 MuJoCo x2 -> walk l1, l0.5, l0 ->
  legs_blind l1, l0.5, l0 -> l0.5 full re-run; ends ~20:45 with "=== done" in logs/queue_paper1.log.
- Findings so far (within 10 cm and 0.6 rad for 1 s): below the standing table lambda 0 17%, lambda 0.5 11%, GOLEM IK
  0.4%; above it 41%, 49%, 52%. lambda 0 holds one deep crouch (pelvis ~0.71 m for every goal); lambda 0.5's pelvis
  rises with the goal (0.71 -> 0.83 m).

Next:
1. When l1 lands: `make -C paper/reach paper`, check fig_hero (l1 is the hero), republish the page (same URL).
2. When the queue ends: `make -C paper/reach paper`, republish, commit, `git bundle create` of
   origin/marl-direct..marl-direct for the coworker.
3. Revise the tex's "Effect of reward sharing" paragraph with lambda 1 (marked % RESULT).

Waiting on owner: the whole-body baseline run (conditions.yaml `whole`), push access for badinkajink.
