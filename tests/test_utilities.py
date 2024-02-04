import cv2
import numpy as np
import os
import pandas as pd

from lightning_pose_app.utilities import check_codec_format


def test_args_to_dict():

    from lightning_pose_app.utilities import args_to_dict

    string = "A=1 B=2"
    args_dict = args_to_dict(string)
    assert len(args_dict) == 2
    assert args_dict["A"] == "1"
    assert args_dict["B"] == "2"


def test_reencode_video(video_file, tmpdir):
    from lightning_pose_app.utilities import reencode_video
    video_file_new = os.path.join(str(tmpdir), 'test.mp4')
    reencode_video(video_file, video_file_new)
    assert check_codec_format(video_file_new)


def test_check_codec_format(video_file):
    assert check_codec_format(video_file)


def test_copy_and_reformat_video(video_file, tmpdir):

    from lightning_pose_app.utilities import copy_and_reformat_video

    # check when dst_dir exists
    video_file_new_1 = copy_and_reformat_video(video_file, str(tmpdir), remove_old=False)
    assert os.path.exists(video_file)
    assert check_codec_format(video_file_new_1)

    # check when dst_dir does not exist
    dst_dir = str(os.path.join(tmpdir, 'subdir'))
    video_file_new_2 = copy_and_reformat_video(video_file, dst_dir, remove_old=False)
    assert os.path.exists(video_file)
    assert check_codec_format(video_file_new_2)


def test_copy_and_reformat_video_directory(video_file, tmpdir):
    from lightning_pose_app.utilities import copy_and_reformat_video_directory
    src_dir = os.path.dirname(video_file)
    dst_dir = str(tmpdir)
    copy_and_reformat_video_directory(src_dir, dst_dir)
    assert os.path.exists(video_file)
    files = os.listdir(dst_dir)
    for file in files:
        assert check_codec_format(os.path.join(dst_dir, file))


def test_get_frames_from_idxs(video_file):
    from lightning_pose_app.utilities import get_frames_from_idxs
    cap = cv2.VideoCapture(video_file)
    n_frames = 3
    frames = get_frames_from_idxs(cap, np.arange(n_frames))
    cap.release()
    assert frames.shape == (n_frames, 1, 406, 396)
    assert frames.dtype == np.uint8


def test_make_video_snippet(video_file, tmpdir):

    from lightning_pose_app.utilities import make_video_snippet

    # get video info
    cap = cv2.VideoCapture(video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # make fake predictions and save to tmpdir
    keypoints = ['paw1', 'paw2']
    n_keypoints = len(keypoints)
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["tracker"], keypoints, xyl_labels], names=["scorer", "bodyparts", "coords"],
    )
    preds = np.random.rand(n_frames, n_keypoints * 3)  # x, y, likelihood
    df = pd.DataFrame(preds, columns=pdindex)
    preds_file = os.path.join(str(tmpdir), 'preds.csv')
    df.to_csv(preds_file)

    # CHECK 1: requested clip is shorter than actual video
    clip_length = 1
    snippet_file = make_video_snippet(
        video_file=video_file,
        preds_file=preds_file,
        clip_length=clip_length,
    )
    cap = cv2.VideoCapture(snippet_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames_1 == int(fps * clip_length)

    # CHECK 2: requested clip is longer than actual video (return original video)
    clip_length = 100
    snippet_file = make_video_snippet(
        video_file=video_file,
        preds_file=preds_file,
        clip_length=clip_length,
    )
    cap = cv2.VideoCapture(snippet_file)
    n_frames_2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames_2 == n_frames


def test_abspath():

    from lightning_pose_app.utilities import abspath

    path1 = 'test/directory'
    abspath1 = abspath(path1)
    assert abspath1 == os.path.abspath(path1)

    path2 = '/test/directory'
    abspath2 = abspath(path2)
    assert abspath2 == os.path.abspath(path2[1:])
