# Hand-off: reach evaluation of the lambda runs (2026-10-08)

Page: paper/20261008-reach_cooperation.html (https://claude.ai/artifact/LHVKbdmtDGtoQpmfghMZBn, v8 at 21:45).
Pipeline and commands: paper/reach/README.md. Pushed to origin/marl-direct (914af59, 19:00) over SSH:
`git push git@github.com:maxconwa/Locomanipulation_game.git marl-direct` (origin's HTTPS URL has no credentials).

State (21:45): every planned evaluation is done for l1 = paper1, l0.5 = paper05, l0 = paper00 (agent_91200, Isaac
on CPU physics, seed 0): full, legs blind, MuJoCo with RoboCasa's and the training plant's joints, walking. Page v8.
- Within 10 cm (and 0.6 rad, 1 s): lambda 1 30% of all goals (13% below the standing table, 54% above), lambda 0.5
  27% (11%, 49%), lambda 0 27% (17%, 41%), GOLEM IK 21% (0.4%, 52%). Blind legs: 18%, 9%, 7%, with 0%, 24%, 41%
  falls. lambda 1's legs do not walk (23 cm/s velocity error; 0.5 and 0: 5 and 4 cm/s).
- l1 arms_ik --ext0 running (logs/queue_l1_armsik.log) for the table's IK-arms row.

Next:
1. Commit and push the arms_ik result (SSH URL, see memory locomanip-push-access).
2. lambda 1 standing still: read the legs' shared arm terms during navigation in its TensorBoard log.
3. The whole-body baseline, when trained: set `run:` of `whole` in conditions.yaml, `make -C paper/reach all`.

Waiting on owner: the whole-body baseline run.
