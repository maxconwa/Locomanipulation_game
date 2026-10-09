# Hand-off: reach evaluation of the lambda runs (2026-10-08)

Page: paper/20261008-reach_cooperation.html (https://claude.ai/artifact/LHVKbdmtDGtoQpmfghMZBn). Pipeline and
commands: paper/reach/README.md. Local marl-direct carries 008c29e, 200e1dc on top of 2a10ffb; origin/marl-direct
has since gained b701ed0 (LocoManip-WholeBody-Direct-v0), not yet rebased onto. NOT pushed: this machine cannot push
to maxconwa/Locomanipulation_game (memory locomanip-push-access).

State (17:50): label l0.5 = paper05 agent_91200 (lambda 0.5, final). Isaac full done: strict 4.3% (below table 1.5%,
above 8.4%); within 10 cm 27% (below 11%, above 49%); 10 cm floor 0.8 m vs GOLEM IK 1.0 m. MuJoCo (RoboCasa joints)
0.3% strict, 11% at 10 cm. l0.5_52k (52.8k steps) is kept as a dotted, non-hero line.
- Running: paper/reach/logs/queue_paper05.sh (PID 3192511), CPU physics (GPU held by the hand project's
  rl_train_cube.py): Isaac legs_blind -> MuJoCo legs_blind, isaacphys, isaacphys+trueodom -> walk -> Isaac arms_ik
  --ext0 -> MuJoCo trueodom -> Isaac trueodom --ext0; ends ~18:45 with "=== done" in logs/queue_paper05.log.
- Hero figure now shows the 10 cm criterion (fig_hero); the strict version is fig_hero_strict.

Next:
1. When the queue log ends with "done": `make -C paper/reach paper`; check fig_hero (blue = legs blind), tradeoff,
   walking, sim2sim; republish the page (same file path, same URL).
2. Rebase onto origin/marl-direct (b701ed0), then teach the evaluators the one-agent "whole" checkpoint
   (load_actors over the checkpoint's agents; eval_isaac/eval_walk act per agent; MuJoCo port's whole observation =
   shared terms + leg actions + arm residual + wrist poses + wrist errors, 135).
3. Commit results + figures + page + tex; git bundle origin/marl-direct..marl-direct; tell the user push needs
   badinkajink added as a collaborator.

Waiting on owner: the lambda 1 and 0 runs (conditions.yaml `run:`), the whole-body baseline run, and push access.
