"""Smoke test — verifies the package installs and exposes __version__."""

import waypoint


def test_version_is_string() -> None:
    assert isinstance(waypoint.__version__, str)
    assert len(waypoint.__version__) > 0
