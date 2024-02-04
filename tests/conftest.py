"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests.

"""

import os
import pytest


@pytest.fixture
def video_file() -> str:
    return os.path.join(
        os.getcwd(),
        "lightning-pose/data/mirror-mouse-example/videos/test_vid.mp4",
    )
