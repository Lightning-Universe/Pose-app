"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests.

"""

import numpy as np
import os
import pytest
import shutil

from lightning_pose_app import (
    LIGHTNING_POSE_DIR,
    LABELED_DATA_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_TMP_DIR,
)

ROOT = os.path.dirname(os.path.dirname(__file__))


def make_tmp_project() -> tuple:

    proj_dir = "data/mirror-mouse-example"

    proj_dir_abs = os.path.join(ROOT, proj_dir)
    if os.path.isdir(proj_dir_abs):
        print(f"{proj_dir_abs} already exists!")
        return proj_dir, proj_dir_abs

    # copy full example data directory over
    src = os.path.join(ROOT, LIGHTNING_POSE_DIR, proj_dir)
    shutil.copytree(src, proj_dir_abs)

    # copy and rename the video for further tests
    tmp_vid_dir = os.path.join(proj_dir_abs, VIDEOS_TMP_DIR)
    os.makedirs(tmp_vid_dir, exist_ok=True)
    src = os.path.join(proj_dir_abs, VIDEOS_DIR, "test_vid.mp4")
    dst = os.path.join(tmp_vid_dir, "test_vid_copy.mp4")
    shutil.copyfile(src, dst)

    # make csv file for label studio
    n_frames = len(os.listdir(os.path.join(proj_dir_abs, LABELED_DATA_DIR)))
    idxs_selected = np.arange(n_frames)
    n_digits = 2
    extension = "png"
    frames_to_label = np.sort(np.array([
        "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected]))
    np.savetxt(
        os.path.join(proj_dir_abs, LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME),
        frames_to_label,
        delimiter=",",
        fmt="%s"
    )

    return proj_dir, proj_dir_abs


@pytest.fixture
def tmp_proj_dir() -> str:

    proj_dir, proj_dir_abs = make_tmp_project()

    # return to tests
    yield proj_dir

    # cleanup after all tests have run (no more calls to yield)
    shutil.rmtree(proj_dir_abs)


@pytest.fixture
def video_file() -> str:
    return os.path.join(
        ROOT, LIGHTNING_POSE_DIR, "data/mirror-mouse-example/videos/test_vid.mp4",
    )


@pytest.fixture
def root_dir() -> str:
    return ROOT
