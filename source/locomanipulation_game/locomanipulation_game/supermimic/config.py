"""SuperMimic hyperparameters (arXiv 2610.28480, Table A.5), as one plain dataclass that `asdict` turns into yaml.

Where this file departs from Table A.5 it says why: the replay holds 400 steps per env (not 1000) so 4096 envs fit on
a 24 GB GPU beside the model; the fields after the paper's (normalize_delta, detach_head_inputs, trip_penalty,
range_margin, policy_start_*, critic_warmup_iters) are ours, each with the measurement that called for it.
Network sizes are the paper's (Table A.4) for the dynamics and the critic; the observation, reward and termination
heads are not given in the paper, so their widths here are ours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SuperMimicCfg:
    # agents the SHAC update trains (the others act with zeros, in the sim and in imagination); empty = all agents
    trained_agents: list[str] = field(default_factory=list)

    # world model (Eq. 1)
    window: int = 8                     # k: autoregressive replay window length
    wm_batches: int = 10                # gradient steps per world-model update
    wm_batch_size: int = 2048           # windows per gradient step
    dynamics_hidden: tuple[int, ...] = (2048, 2048, 2048)
    obs_hidden: tuple[int, ...] = (1024, 1024)
    reward_hidden: tuple[int, ...] = (512, 512)
    termination_hidden: tuple[int, ...] = (64, 64)     # also the trip heads' (wider ones overfit their rare positives)
    activation: str = "elu"             # elu | relu | silu | gelu | tanh
    # the reward head predicts r / reward_scale, and its loss is in those units: set it to a typical |reward| per
    # step so the reward term weighs about as much as a normalised state dimension
    reward_scale: float = 1.0
    # floor of every running normaliser's std, so a (nearly) constant input does not blow up to huge normalised values
    norm_std_floor: float = 1e-2
    # the dynamics network predicts the state change in units of its own running std (s' = s + delta_std * net + mean),
    # not of the state's: slow dimensions (height, gravity, clocks) then cost the network as much as fast ones and an
    # untrained network predicts the mean change. Its floor only guards a division (outputs are multiplied by it).
    normalize_delta: bool = True
    delta_std_floor: float = 1e-6
    # the observation, reward, trip and termination heads read the rolled-out state detached in Eq. 1, so their losses
    # do not train the dynamics: with them attached (the paper's joint loss) the reward's error dominated the shared
    # global-norm clip and the dynamics' 1-step error stalled at 0.61 against 0.17 detached (2026-11-20 diagnosis)
    detach_head_inputs: bool = True
    # GOLEM e-stop trips: their cost per trip (the env's golem_estop_penalty; 0 = no trip head). With it the reward
    # head fits the reward without the trip cost and a Bernoulli trip head per agent its probability, which imagination
    # charges at this cost, as the termination head's probability is charged the fall penalty
    trip_penalty: float = 0.0
    # imagination clamps each predicted state dimension and each predicted reward to the range seen in real data,
    # widened by this fraction of it on each side (negative: no clamp), so the policy cannot climb the model's
    # extrapolation (the 2026-11-20 sanity run's imagined H-step return reached +5275 against a real -5..0)
    range_margin: float = 0.1
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"state": 1.0, "obs": 1.0, "reward": 1.0, "termination": 1.0})

    # imagination and policy (Eq. 2)
    horizon: int = 32                   # H
    gamma: float = 0.99                 # the paper's
    imagine_batch: int = 4096           # imagined rollouts per policy update
    imagine_chunks: int = 1             # sequential passes over the rollouts: the same gradient, less memory
    policy_updates: int = 1             # actor + critic updates per iteration, each on freshly sampled starts
    fall_penalty: float = -20.0         # added as fall_penalty * d_hat to every agent's imagined reward
    actor_betas: tuple[float, float] = (0.9, 0.95)
    actor_grad_clip_elem: float = 0.1   # after the global-norm clip

    # critic (Eq. 3)
    gae_lambda: float = 0.95
    critic_hidden: tuple[int, ...] = (256, 128)
    critic_epochs: int = 5
    critic_minibatches: int = 4

    # shared optimisation
    lr: float = 2e-4                    # world model, actors and critics
    grad_clip: float = 1.0              # global norm, everywhere
    lr_decay_iters: int = 0             # cosine decay over this many iterations, then constant; 0 = constant lr
    lr_final_ratio: float = 0.03

    # when the policy starts: the actors and critics wait at least policy_start_iters iterations and until the world
    # model's 1-step state error (normalised RMSE, the train metric) is below policy_start_rmse, at the latest
    # 5 x policy_start_iters; then the critics alone are fit for critic_warmup_iters iterations before the actors move
    # (on a few dozen steps of data the error reads low by overfitting: the minimum wait is what guards against that)
    policy_start_rmse: float = 0.3
    policy_start_iters: int = 100
    critic_warmup_iters: int = 50

    # data collection
    action_noise_std: float = 0.1       # N(0, std^2) added to the trained agents' mean actions while collecting
    # > action_noise_std: each env's collection noise std is drawn once, log-uniform in [action_noise_std, this], so
    # the data covers actions away from the policy's (ShacTrainer.collection_noise). With 0.1 everywhere the model,
    # fit on near-policy actions only, predicted that N(0, 1) actions cost nothing and fell less, the opposite of the
    # sim (2026-11-20, SmcFixLegs256), and SHAC drove |a| from 0.08 to 1.2 while the real reward halved and trips
    # doubled; with 0.1-0.5 (SmcFixNoiseLegs256) trips stayed flat and |a| reached 0.96. 0 = the paper's fixed 0.1
    action_noise_std_max: float = 0.5
    buffer_steps: int = 400             # replay capacity per env
    collect_steps: int = 8              # env steps per iteration


def lr_factor(cfg: SuperMimicCfg, iteration: int) -> float:
    """Multiplier of cfg.lr at an iteration: cosine from 1 to lr_final_ratio over lr_decay_iters, then constant."""
    if cfg.lr_decay_iters <= 0:
        return 1.0
    x = min(iteration / cfg.lr_decay_iters, 1.0)
    return cfg.lr_final_ratio + (1.0 - cfg.lr_final_ratio) * 0.5 * (1.0 + math.cos(math.pi * x))


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
