"""SuperMimic (arXiv 2610.28480) for the LocoManip tasks: a deterministic world model fit to logged simulator
transitions and SHAC-style policy improvement through it. Pure torch, no simulator imports: the Isaac side lives in
scripts/supermimic/train.py and the env's hooks.
"""

from .buffer import ReplayBuffer, flatten_lstm, standard_spec, unflatten_lstm
from .config import SuperMimicCfg, lr_factor, set_lr
from .shac import Critic, Imagination, ShacTrainer, actor_step
from .world_model import AgentDims, GoalFns, RunningNorm, WorldModel, mlp, train_world_model

__all__ = [
    "AgentDims", "Critic", "GoalFns", "Imagination", "ReplayBuffer", "RunningNorm", "ShacTrainer", "SuperMimicCfg",
    "WorldModel", "actor_step", "flatten_lstm", "lr_factor", "mlp", "set_lr", "standard_spec", "train_world_model",
    "unflatten_lstm",
]
