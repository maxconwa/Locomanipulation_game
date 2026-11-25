"""Tests marked ``sim`` share one headless Isaac Sim app per session; the others never start it."""

import sys

import pytest

_APP = []  # the session's Isaac Sim app, once a sim test asks for it


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: needs the Isaac Sim app (the sim_app fixture)")
    # Kit must load its own warp before anything imports isaaclab.utils (which pulls in the pip warp): some test
    # modules do so at import time (main_obs loads joint_noise.py), so a ``-m sim`` session starts the app before
    # collection rather than in the fixture.
    markexpr = config.getoption("markexpr", "") or ""
    if "sim" in markexpr and "not sim" not in markexpr:
        _start_app()


def pytest_unconfigure(config):
    # Closing the app ends the process without flushing, so it waits until pytest has written its report.
    if _APP:
        sys.stdout.flush()
        sys.stderr.flush()
        _APP[0].close()


def _start_app():
    from isaaclab.app import AppLauncher

    if not _APP:
        # Kit reads sys.argv; give it the launcher's own (none) rather than pytest's (-m, -k, -q ...).
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            _APP.append(AppLauncher(headless=True).app)
        finally:
            sys.argv = argv
    return _APP[0]


@pytest.fixture(scope="session")
def sim_app():
    return _start_app()
