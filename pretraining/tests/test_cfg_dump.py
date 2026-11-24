"""The resolved-cfg dump guard (scripts/dump_task_cfgs.py): a dump is deterministic and a compare names the change."""

import copy
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dump_task_cfgs.py"


def _module():
    spec = importlib.util.spec_from_file_location("dump_task_cfgs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.sim
def test_dump_is_deterministic_and_compare_flags_a_change(sim_app):
    dumper = _module()
    first, second = dumper.dump(dumper.DEFAULT_TASKS), dumper.dump(dumper.DEFAULT_TASKS)
    assert dumper.compare(first, second) == []
    changed = copy.deepcopy(second)
    changed["LocoManip-Marl-Direct-v0"]["lambda_legs"] += 0.25
    assert dumper.compare(first, changed) == ["LocoManip-Marl-Direct-v0.lambda_legs"]
