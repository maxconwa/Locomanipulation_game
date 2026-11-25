"""The modules an entry point imports before AppLauncher must not import Isaac Lab, warp or omni: a pip warp loaded
before Kit's breaks its extensions (``module 'warp.types' has no attribute 'array'``)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULES = [
    "pretraining.scripts.train_commands",
    "pretraining.scripts.train",
    "pretraining.scripts.fit_prior",
    "pretraining.scripts.crosscheck",
    "pretraining.scripts.play",
    "pretraining.scripts.pulse_baseline",
    "pretraining.scripts.export_decoders",
    "pretraining.scripts.pd_stability",
    "pretraining.scripts.wrist_frames",
    "pretraining.scripts.rollout",
    "pretraining.marl.command_runner",
    "pretraining.marl.main_obs",
    "pretraining.marl.marl_source",
    "pretraining.marl.pulse",
    "evals.isaac.latent",
]


@pytest.mark.parametrize("module", MODULES)
def test_no_isaac_before_the_app(module):
    code = (f"import sys, importlib; importlib.import_module({module!r}); "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('warp', 'isaaclab', 'omni')); print(bad); "
            "sys.exit(1 if bad else 0)")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
