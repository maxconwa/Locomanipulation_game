# Hand-off: reach evaluation of the lambda runs (2026-10-08)

Page: paper/20261008-reach_cooperation.html (https://claude.ai/artifact/LHVKbdmtDGtoQpmfghMZBn, v9 at 23:15).
Pipeline and commands: paper/reach/README.md. Pushed to origin/marl-direct (914af59, 19:00) over SSH:
`git push git@github.com:maxconwa/Locomanipulation_game.git marl-direct` (origin's HTTPS URL has no credentials).

State (23:15): every planned evaluation is done and pushed for l1 = paper1, l0.5 = paper05, l0 = paper00 and
whole = paperwb (agent_91200; Isaac on CPU physics, seed 0): full, legs blind (not for whole), MuJoCo with RoboCasa's
and the training plant's joints, walking; l1 arms_ik on the unextended goals. Page v9.
- Within 10 cm (and 0.6 rad, 1 s), all goals: lambda 1 30%, lambda 0.5 27%, lambda 0 27%, whole-body agent 7%,
  GOLEM IK 21%. Below the standing table: 13%, 11%, 17%, 2%, 0.4%. Above: 54%, 49%, 41%, 14%, 52%.
- lambda 1 and the whole-body agent do not walk (velocity error 23 and 24 cm/s; lambda 0.5 and 0: 5 and 4 cm/s).
- Blind legs: 18%, 9%, 7% within 10 cm with 0%, 24%, 41% falls. lambda 1 with IK arms: 19% of the unextended goals
  within 10 cm, against 41% with its residual.

Next:
1. lambda 1 and the whole-body agent standing still: read the legs' reward terms during navigation in their
   TensorBoard logs (Episode_Reward/legs/arms_* for lambda 1).
2. More seeds per goal on the GPU when it is free (`make -C paper/reach all SEEDS=0,1,2` after deleting results).

Waiting on owner: nothing.
