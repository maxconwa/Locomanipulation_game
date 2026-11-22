"""Import the supermimic package on its own: locomanipulation_game/__init__.py registers the Isaac tasks (it imports
isaaclab), so the tests put the package's parent folder on sys.path and import `supermimic` directly, on the CPU."""

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "source",
                                           "locomanipulation_game", "locomanipulation_game"))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
import torch; torch.set_num_threads(4)
