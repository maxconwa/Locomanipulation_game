# LocoManip-VlaUpper-v0: upper-body VLA block stacking (2026-11-13/14)

One night's build of the task, its teacher, the policy and its DAgger trainer. **The pipeline runs end to end and the
teacher stacks 93% of the time; the trained policy does not work yet (about 1%).** What was built, the numbers, what
each failed run showed, and what to do next.

## What exists

| Piece | Where | State |
|---|---|---|
| Task: pinned pelvis, table, 6 blocks per env (2-4 on the table), 3 cameras, instruction | `tasks/locomanip_marl/vla_upper_env_cfg.py`, `vla_upper_env.py` | works |
| Delta end-effector IK action, Magpie gripper action, block setup, proprioception | `tasks/locomanip_marl/mdp/vla.py` | works |
| Success-guided reset sampling (54 bins: blocks on the table x picked size x target size x arm) | `vla_upper_env.py` `SuccessGuidedSampler` | works; untested as a curriculum (the policy succeeds too rarely) |
| Scripted privileged teacher | `vla/teacher.py` | 93.4% of 256 episodes |
| Frozen SmolVLM2-500M + DINOv2-S encoders, trained keypoint encoder, action head | `vla/encoders.py`, `vla/model.py` | trains; see results |
| DAgger trainer, replay, policy runner | `scripts/vla/train.py`, `vla/buffer.py`, `vla/runner.py` | works |
| Evaluation, training plot, demos | `scripts/vla/eval.py`, `plot_training.py`, `teacher_demo.py`, `gripper_demo.py` | work |

The existing tasks are untouched: the only shared file changed is `tasks/locomanip_marl/__init__.py` (one
`gym.register` appended). `mdp/vla.py` is imported by the new config only.

    conda activate env_isaaclab
    python scripts/vla/train.py --headless --enable_cameras --name <name> --grad_steps 2 --batch 256
    python scripts/vla/eval.py --headless --enable_cameras --checkpoint logs/vla/<run>/checkpoints/policy_<N>.pt
    python scripts/vla/eval.py --headless --enable_cameras --teacher --out logs/vla/teacher_eval

## Results

Run `VlaUpperV6_s42` (`logs/vla/2026-11-14_05-53-36_VlaUpperV6_s42`), step 12,000 (64 envs; 768,000 env steps over the chain of resumed runs V3 -> V6).
Uniform configurations, 20 s episodes, every env the same number of episodes.

| Evaluation | Episodes | Stacked | Lifted, then misplaced | Wrong block | Never grasped |
|---|---|---|---|---|---|
| Policy, the 64 training block layouts | 256 | 0.8% | 35.2% | 0% | 64.1% |
| Policy, new block layouts (`--block_seed 1`) | 128 | 1.6% | 31.3% | 0% | 67.2% |
| Teacher | 256 | 93.4% | 6.3% | 0% | 0.4% |

In training, the policy-driven episodes stacked 5-8% of the time from step 8,300 on (550 episodes). That is under the
success-guided sampler, which draws the configurations the policy succeeds on more often; the uniform evaluation is
the number to trust.

The teacher by configuration: small targets 81% (medium 98%, large 99%), small picked blocks 88% (large 98%).

Where the policy fails (its action against the teacher's at the same states, training layouts):

| Teacher's phase | Steps | Policy's position delta | Teacher's | Cosine | Gripper command agrees |
|---|---|---|---|---|---|
| approach | 56,061 | 0.23 | 0.59 | -0.08 | 91% |
| descend | 25,507 | 0.22 | 0.25 | 0.39 | 90% |
| close | 8,251 | 0.22 | 0.53 | 0.26 | 15% |
| lift | 5,202 | 0.20 | 0.60 | -0.11 | 33% |
| carry | 3,747 | 0.19 | 0.60 | -0.16 | 63% |

- It reaches the block (about 40% of the steps are in descend or later) and lifts the instructed block in a third of the
  episodes; it almost never lifts another block.
- When the teacher closes the jaws the policy keeps them open 85% of the time. The teacher's close and open phases are
  timed, not triggered by anything the policy observes, so on the policy's own states the label flips with nothing in
  the input to predict it.
- On the replay, the policy's first position delta agrees with the teacher's (cosine 0.76-0.80); in rollouts it starts
  that way (0.88 on an episode's first step) and then hovers around the waypoint, where the teacher's deltas are small
  corrections it does not reproduce.
- The keypoint encoder places the instructed block and its target within 0.6-0.7 cm on the training layouts, and
  5.7 cm on layouts with other sizes and colours (measured with 32 envs): it has not generalized beyond the 64 layouts.

Training plot: `logs/vla/2026-11-14_05-53-36_VlaUpperV6_s42/training.png`. Camera video of the policy: `logs/vla/2026-11-14_05-53-36_VlaUpperV6_s42/eval_12000/cameras.mp4`; of the teacher:
`logs/vla/2026-11-14_05-53-36_VlaUpperV6_s42/eval_teacher/cameras.mp4`.

## Departures from the brief

1. **A trained vision part.** The brief has a frozen VLM for the head camera and a frozen (or lightly adapted) encoder
   for the wrists. On 10,000 scenes, probes of the frozen encoders placed the instructed block no better than the
   centre of its zone:

   | Probe's input | Median error placing the instructed block |
   |---|---|
   | VLM tokens (head image + instruction) | 7.3 cm (which arm's side: 99%) |
   | DINOv2 patch tokens of the head image, the block's colour and size given | 7.2 cm |
   | Small CNN on the pixels, colour and size given | 0.3 cm |
   | Small CNN queried by the VLM's word embeddings, 6,000 steps | 4.7 cm, falling |

   So the policy has a small CNN keypoint encoder on the three images (`KeypointEncoder`), queried by the frozen VLM's
   input embeddings of the instruction's words, with an auxiliary loss in training on the two blocks' true positions.
   The frozen VLM tokens and DINOv2 wrist tokens are still inputs. Run V2, the brief's architecture exactly, is the
   baseline: 0% and a flat loss at 3,000 steps (`logs/vla/VlaUpperV2_s42_log.jsonl`).
2. **A state branch beside the action head.** An MLP from the standardized state (proprioception, the keypoints' block
   positions, each commanded grasp point's offset to each block) to the chunk, added to the transformer head's output.
   The transformer given the same features as a token did not learn the arm's motion in 5,000 steps; offline, an MLP
   on them reached a cosine of 0.90 with the teacher's deltas.
3. **The commanded grasp point is an observation** (52 values, not 34). The action's deltas add to the controller's
   commanded point and the teacher steers that point, so the policy has to see it; on the robot it is the
   controller's own state.
4. **Task id** `LocoManip-VlaUpper-v0`, not `vla-upper-v0` (`scripts/list_envs.py` lists only ids with `LocoManip-`).
5. **20 Hz policy steps on 200 Hz physics**, not the other tasks' 100 Hz on 500 Hz.
6. **`num2words` was pip-installed** into `env_isaaclab` while testing SmolVLM's processor; the code uses the
   tokenizer only, so it can be uninstalled.

## What each failed run showed

| Run | Result | Cause found |
|---|---|---|
| V1 | 0%, position loss at the predict-nothing level | Envs overlapped: on the walking task's generated terrain all envs start on 20 tiles, so each camera saw other envs' blocks. Now a plane and a grid. |
| V1, V2 | same | The teacher steers the commanded grasp point, which the policy could not see. |
| V2 | same | The arms' ready pose hid the blocks from the head camera; the frozen encoders don't place a block (above). |
| V3 | blocks placed to 1.2 cm, arm motion not learned | The teacher closed the whole distance to its waypoint each step and its executed actions carried noise of 0.15: near a waypoint its action was "undo the last step's noise". Now half the distance, noise 0.04. |
| V4 | cosine with the teacher 0.0 in rollouts | Given the commanded point's lead over the measured one, the head learned to keep moving the way it was moving. Removed. |
| V5 | cosine 0.8 on the replay, 0% in rollouts | The teacher left its approach only within 1.2 cm and 0.08 rad of the waypoint; a policy hovering 1-2 cm off was never shown a later phase. Gates widened (3 cm, 0.25 rad; carry 1.2 cm). |
| V6 | 5-8% in training, 1% evaluated | Open: see below. |

Other things worth knowing: a contact sensor with a history recomputes at every physics substep (it was most of a
step's time); `DirectMARLEnv` does not render after a reset (the env does); the closed jaws' tips reach 3.5 cm past
the grasp frame and swing forward as they close, so the grasp point sits 4.7 cm above the table and small blocks are
held by the tips; neither arm reaches the strip between them, so an instructed pair is always on one arm's side.

## Next steps, in the order I would take them

1. **Make the teacher's phases a function of what the policy sees.** Close when the jaws are around the block (the
   grasp point within a tolerance), open when the block rests on the target, not on timers; then the labels at the
   policy's own states are predictable. This is the gripper's 15% agreement above.
2. **A lower-gain or velocity-like teacher near waypoints**, or an absolute-target action for the last centimetres, so
   small position errors don't flip the label's direction (the hovering).
3. **More block layouts.** The layouts are 64 fixed sets; resample sizes and colours across envs (more envs, or
   several runs' layouts) so the keypoint encoder generalizes. `block_layout` also depends on the env count, which
   the evaluation has to match.
4. **Then** longer training and the success-guided sampler's effect (uniform vs guided), held-out layouts as the
   standing test.
5. Not started: export, the MuJoCo plant (it has no table or blocks), the robot-side inference loop, wrist-encoder
   LoRA, visual domain randomization (lighting, textures), a second phrasing of the instruction.

## V7 (2026-11-14): state-based teacher, controller fix, colours per episode

The three next steps above, implemented, and a retrain from scratch (`VlaUpperV7_s42`, 64 envs, 16,000 steps,
teacher noise 0.015).

- **Teacher** (`vla/teacher.py`, rewritten): no timers and no carried phase. The phase is recomputed each step from
  the state (block between the jaws, jaw closure against the block's size, block resting on its target); waypoints
  slide in along a funnel; the action is a low gain (0.15) on the *measured* grasp point's error.
- **Controller** (`mdp/vla.py`, `TcpKinematics.step`): the null-space posture pull was projected with the step's
  damping (0.05) and leaked into the task: a zero action crept ~2 mm/step and arms stopped 1-2 cm short. It now
  has its own damping (1e-3).
- **Colours** are drawn per episode (`block_colors_per_episode`); sizes stay fixed per env (PhysX).

| | training sizes (256 ep.) | new sizes (128 ep.) |
|---|---|---|
| teacher | 97.7% (was 93.4%) | |
| V7 policy, step 16,000 | 2.3% (V6: 0.8%) | 3.1% (V6: 1.6%) |

Failures: 53% lifted the block and misplaced it, 44% never grasped. The gripper now agrees with the teacher in
95-100% of the steps of every phase but close (54%) and open (1%).

**The in-training success (about 20% from step 12,000) is not the policy's success.** The replay holds the last
2,500 steps, about 400 episodes, the running ones included, and the policy fits them: on the replay the block
estimate is 0.7 cm off and the cosine with the teacher 0.93; on fresh episodes (`scripts/vla/heldout_check.py`)

| fresh episodes driven by | instructed block | target block | cosine with the teacher |
|---|---|---|---|
| the teacher | 2.1 cm | 7.3 cm | 0.39 |
| the policy | 5.8 cm | 9.5 cm | 0.14 |

So the limit is now the vision: the keypoint encoder memorizes a few hundred scenes instead of localizing blocks.
Next: train the localizer on many more scenes than a rolling replay holds (a reset is a new scene: tens of
thousands of reset-only frames cost minutes), with a held-out error logged while training; then DAgger on top with
that encoder, and a held-out evaluation at each checkpoint as the number to read.

## V8 (2026-11-14): a bigger replay and scene envs

`VlaUpperV8_s42`, resumed from V7's step 16,000 to step 48,000 with two changes in `scripts/vla/train.py` and
`vla/buffer.py`:

- **Replay** of 12,000 steps (was 2,500), images stored and sampled every 4th step (actions every step), 125 GB.
- **Scene envs**: the last 16 of the 64 envs are teacher-driven and their episode is ended every 8 steps
  (`env.end_now`), so each shows a newly drawn scene that often: about 24,000 scenes in the replay, from 400. A
  scene's first frame is scored before it is trained on: `fresh_pick_cm`, `fresh_target_cm`, `fresh_cosine` in the
  log and the plot.

The block estimate on new scenes went from 3.3 / 10.4 cm (instructed / target) at the resume to 0.7 / 0.5 cm.

| | training sizes (256 ep.) | new sizes (128 ep.) |
|---|---|---|
| teacher | 97.7% | |
| V7, step 16,000 | 2.3% | 3.1% |
| V8, step 48,000 | 41.4% | 42.2% |

In training the policy's episodes stacked 48-54% from step 20,000 on (guided sampler), so the in-training number is
now close to the evaluation's. Failures (training sizes): 46% lifted and misplaced, 13% never grasped, 0 wrong
block. What is left:

- **The right arm: 12% against the left's 71%** (14% / 70% on new sizes). The teacher has no such gap.
- **Lowering and releasing**: 24,000 steps in the teacher's lower phase at cosine 0.08, and the gripper agrees with
  the teacher's open in 10% of the steps: the policy hovers over the target holding the block.
- Small targets 24%, large 51%; large instructed blocks 33%.

## Dataset and visual augmentation (2026-11-15)

- `logs/vla/datasets/teacher_100k`: 58,644 teacher episodes (6.49 M steps, 40 GB; stopped there on request), over a
  chroma-green floor, images as video (`scripts/vla/collect_dataset.py`, format in the folder's README).
- On recorded images (`vla/augment.py`, nothing rendered): the floor's pixels take any texture; the table is
  retextured with each pixel's brightness kept (shadows stay); the blocks' colours are swapped between names with the
  instruction's words swapped to match. Textures are drawn on the floor's and table's planes from the cameras' poses
  (head: fixed; wrists: 0.10 m behind the grasp point, fitted to the blocks' pixels, 3.5-3.9 px median error of 168).
- At render time (`mdp/vla.py VisualRandomizer`, `collect_dataset.py --visuals <texture folder>`): per episode the
  table's texture/colour/roughness, each block's roughness, metal share and (grey, tinted) texture, and a lamp per env
  beside a dimmer dome. The draws are saved in `meta.jsonl`. The floor is made matte (`make_matte`): with the lamps
  its red/blue rose from 14 to 80 and left the key's box; matte it is (0, 232, 0) everywhere.
- Textures: Poly Haven's 870 diffuse maps at 1k in `logs/vla/textures/polyhaven` (CC0), tags in `index.json`.
- Open: bright squares in one wrist camera's images where a lamp's highlight falls on a glossy surface, in some
  launches (VisualRandomizationCfg.lamp_radius's comment). The hue-based block and table masks are for the plain
  dataset: they do not hold on textured tables or metallic blocks.
- Nothing in training uses either yet.
