# Pure RL against pretraining + PULSE on one POMDP: design

Max (2026-11-24): "What we ultimately want is a fair comparison between the pure rl cooperative task and this
pretrained and pulse cooperative task. It would be great if the pulse was trained on the same POMDP as the pure rl
cooperative task", and "I like training the policy via propagating gradients through the world model".

Why now: the first PULSE run (`LocoManip-Marl-Pulse-v0`, MAPPO, 158k steps) plateaued with 10% of ending episodes
falls and no arm goal reached. It ran on the decoders' plant (50 Hz, pretraining's gains, ten arm joints on implicit
PD), not main's, and the frozen decoders fell on it 23% of the time in 10 s with no policy acting.

## Max's decisions

1. **Plant: stage 1 is retrained on main's plant.** The baseline's plant is not changed.
2. **Torso: the arms agent drives it in every arm of the comparison**, so the pure-RL baseline is retrained with a
   driven torso.
3. **Three arms:** (A) pure RL + MAPPO, (B) PULSE + MAPPO, (C) PULSE + world-model gradients.
4. **Rewards: each method its own training rewards; one scorecard for all.** PULSE keeps the gait, stance and
   torso-posture terms at zero; the baseline keeps main's full set.
5. **Development runs default to 40k steps** (about 3 h; the first PULSE run's legs had plateaued by 30k).

## The shared POMDP

Everything below is identical in A, B and C, and is main's `LocoManip-Marl-Direct-v0` as it stands except the torso:

- 100 Hz control (`sim.dt` 0.002, decimation 5), explicit `IdealPDActuatorCfg` on every driven joint with main's
  sport-mode gains, effort limits, armature and default pose;
- main's randomisation and sensing (friction, torso mass, resets, pre-pose resets, actuator scales, 0-10 substep
  delay, encoder offsets, joint noise, lowstate IMU, gyro low-pass);
- main's commands and curricula (velocity command, wrist-pose goals), terminations and GOLEM e-stop model;
- main's actor and critic observations;
- the torso joint driven by the arms agent.

What differs is the method only: the action each agent emits (A: main's joint/IK-residual actions plus a torso
target; B, C: a 32-wide latent decoded by the frozen stage-1 decoder), the training rewards (decision 4) and the
learner (A, B: skrl MAPPO; C: gradients through the world model).

## Pieces to build

1. **Baseline with a driven torso** (`LocoManip-Marl-Direct-v0` gains a torso action for the arms agent behind a cfg
   field, default off so the existing task is unchanged; a new id, e.g. `LocoManip-Marl-Torso-v0`, turns it on).
   My call: a joint-position residual on the torso with the same scale convention as GOLEM's uncommitted
   `torso_action_scale` work.
2. **Stage 1 on main's plant** (`pretraining/`): a second track-env configuration with main's robot cfg (gains,
   explicit PD, default pose), 100 Hz, the 50 Hz clips interpolated to 100 Hz. Contract: target = main's default
   pose + `ACTION_SCALE` x output, 87-wide proprio as now. Two runs, as before: a nominal base run, then a
   fine-tune with main's randomisation, delay and the GOLEM limits on. My calls, to be checked in a short run
   before the long one: the world model's horizon goes from 8 to 16 steps (the same 0.16 s), and the iteration
   budget doubles for the same seconds of motion.
3. **`LocoManip-Marl-Pulse-v0` rebuilt on main's plant** (arm B, and the evaluation env for C): the overrides that
   moved it to pretraining's plant are removed (rate, gains, implicit joints, default pose, standing height); the
   latent action, the prior and the zeroed rewards stay.
4. **Stage 2 through the world model** (arm C): h12_train's `train_commands.py` ported into `pretraining/`. A
   command encoder per agent maps main's command and main's actor observation into the frozen decoder's latent,
   trained by back-propagating main's velocity and wrist tracking rewards (functions of the state the world model
   predicts) plus the GOLEM e-stop penalties. Data is collected in the piece-3 env, so the world model sees main's
   randomisation. Kept from the reference: a quarter of segments are clip segments with the latent pulled toward
   the stage-1 encoder's (weight 0.1). My call: the encoder is recurrent (an LSTM as main's actor), because main's
   delay and noise make one step of proprio a weaker state estimate than in the reference's nominal sim.
5. **One scorecard:** `evals/isaac` and `evals/mujoco` scorecards and `get_video` take a latent policy (decoder in
   the loop), so A, B and C are scored by the same code on the same conditions. The deploy gate is out of scope.

## Approaches considered

- **Move the baseline to the decoders' plant** (retrain pure RL at 50 Hz with pretraining's gains): no new stage 1,
  but the baseline stops being the policy tuned from deployment data. Rejected by decision 1.
- **Keep the plants different and report it:** cheapest; not the same POMDP. Rejected.
- **Two arms (A against C):** a difference could come from pretraining or from the learner. Rejected by decision 3.

## Runs and estimates (one GPU, 4,096 envs unless memory says otherwise)

| Run | Estimate |
|---|---|
| Stage 1 base on main's plant | 8-13 h (the 50 Hz one took 6.4 h for 15.8k updates; twice the iterations per second of motion) |
| Stage 1 fine-tune (randomisation, GOLEM limits) | 2-3 h |
| Prior fit | 0.5 h |
| A, B, C development runs, 40k steps each | about 3 h each; C's speed is unknown until measured |
| A, B, C final runs, 200k steps each | about 13 h each (A's past runs; B and C to be measured) |

Before the stage-1 long run: a clip-replay check that the new decoders' wrist poses match main's wrist-goal frames
(the first PULSE run showed 120-140 degree wrist orientation errors, cause not established).

## Success

A scorecard table for A, B and C at 200k steps on the same conditions (velocity error, wrist position and
orientation error, falls, e-stop trips), with the 40k-step development runs showing each arm trains before its
long run is started.

## Not in scope

The deploy gate and GOLEM export for latent policies; a zeroed-reward baseline; changing main's randomisation.
Nothing AMASS-derived is committed.
