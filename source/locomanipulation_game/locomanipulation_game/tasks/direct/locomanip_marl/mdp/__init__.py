"""MDP terms for the two-agent task.

Everything the manager-based rounds use, plus the agent-aware terms here.
"""

from locomanipulation_game.tasks.manager_based.locomanipulation_game.mdp import *  # noqa: F401, F403

from .actions import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
