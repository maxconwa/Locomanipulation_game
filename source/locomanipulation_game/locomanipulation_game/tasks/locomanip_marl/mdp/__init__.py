"""MDP terms for the two-agent task, in one `mdp` namespace: Isaac Lab's built-ins, the locomotion ones from
isaaclab_tasks, the legs' ALMI terms (legs_rewards.py), then the task's own."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .legs_rewards import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
