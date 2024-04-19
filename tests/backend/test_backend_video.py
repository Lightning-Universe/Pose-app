import os

import cv2
import numpy as np

from lightning_pose_app.backend.video import check_codec_format


def test_check_codec_format(video_file):
    assert check_codec_format(video_file)


def test_reencode_video(video_file, tmpdir):
    from lightning_pose_app.backend.video import reencode_video
    video_file_new = os.path.join(str(tmpdir), 'test.mp4')
    reencode_video(video_file, video_file_new)
    assert check_codec_format(video_file_new)


def test_copy_and_reformat_video(video_file, tmpdir):

    from lightning_pose_app.backend.video import copy_and_reformat_video

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
    from lightning_pose_app.backend.video import copy_and_reformat_video_directory
    src_dir = os.path.dirname(video_file)
    dst_dir = str(tmpdir)
    copy_and_reformat_video_directory(src_dir, dst_dir)
    assert os.path.exists(video_file)
    files = os.listdir(dst_dir)
    for file in files:
        assert check_codec_format(os.path.join(dst_dir, file))


def test_get_frames_from_idxs(video_file):
    from lightning_pose_app.backend.video import get_frames_from_idxs
    cap = cv2.VideoCapture(video_file)
    n_frames = 3
    frames = get_frames_from_idxs(cap, np.arange(n_frames))
    cap.release()
    assert frames.shape == (n_frames, 1, 406, 396)
    assert frames.dtype == np.uint8


def test_make_video_snippet(video_file, video_file_pred_df, tmpdir):

    from lightning_pose_app.backend.video import make_video_snippet

    n_frames = video_file_pred_df.shape[0]

    # save out dummy video prediction dataframe to csv file
    preds_file = os.path.join(str(tmpdir), 'preds.csv')
    video_file_pred_df.to_csv(preds_file)

    # CHECK 1: requested clip is shorter than actual video
    clip_length = 1
    snippet_file, _, _ = make_video_snippet(
        video_file=video_file,
        preds_file=preds_file,
        clip_length=clip_length,
    )
    cap = cv2.VideoCapture(snippet_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames_1 == int(fps * clip_length)
    os.remove(snippet_file)

    # CHECK 2: requested clip is longer than actual video (return original video)
    clip_length = 100
    snippet_file, _, _ = make_video_snippet(
        video_file=video_file,
        preds_file=preds_file,
        clip_length=clip_length,
    )
    cap = cv2.VideoCapture(snippet_file)
    n_frames_2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames_2 == n_frames
    os.remove(snippet_file)


def test_compute_motion_energy_from_predection_df(video_file_pred_df):

    from lightning_pose_app.backend.video import compute_motion_energy_from_predection_df

    likelihood_thresh = 0
    me = compute_motion_energy_from_predection_df(video_file_pred_df, likelihood_thresh)
    assert video_file_pred_df.shape[0] == len(me)
    assert np.isnan(me).sum() == 0

    df = video_file_pred_df.copy()
    mask = df.columns.get_level_values('coords').isin(['likelihood'])
    loc_until_row = 3
    sum_of_nan = loc_until_row + 1  # dataframe indexing includes this row
    df.loc[:, mask] = 1
    df.loc[:loc_until_row, mask] = 0
    likelihood_thresh = 0.5
    me = compute_motion_energy_from_predection_df(df, likelihood_thresh)
    assert df.shape[0] == len(me)
    assert np.isnan(me).sum() == sum_of_nan
