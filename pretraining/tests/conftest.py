"""Tests marked ``sim`` share one headless Isaac Sim app per session; the others never start it."""

import sys

import pytest

_APP = []  # the session's Isaac Sim app, once a sim test asks for it


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: needs the Isaac Sim app (the sim_app fixture)")


def pytest_unconfigure(config):
    # Closing the app ends the process without flushing, so it waits until pytest has written its report.
    if _APP:
        sys.stdout.flush()
        sys.stderr.flush()
        _APP[0].close()


@pytest.fixture(scope="session")
def sim_app():
    from isaaclab.app import AppLauncher

    if not _APP:
        _APP.append(AppLauncher(headless=True).app)
    return _APP[0]
