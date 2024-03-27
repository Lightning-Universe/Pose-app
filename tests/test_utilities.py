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


def test_check_codec_format(video_file):
    assert check_codec_format(video_file)


def test_reencode_video(video_file, tmpdir):
    from lightning_pose_app.utilities import reencode_video
    video_file_new = os.path.join(str(tmpdir), 'test.mp4')
    reencode_video(video_file, video_file_new)
    assert check_codec_format(video_file_new)


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


def test_get_frame_number():

    from lightning_pose_app.utilities import get_frame_number

    file = "img000346.png"
    out = get_frame_number(file)
    assert out == (346, "img", "png")

    file = "frame3.jpg"
    out = get_frame_number(file)
    assert out == (3, "frame", "jpg")

    file = "im000.jpeg"
    out = get_frame_number(file)
    assert out == (0, "im", "jpeg")


def test_is_context_dataset(tmp_proj_dir):

    from lightning_pose_app import LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME
    from lightning_pose_app.utilities import is_context_dataset

    labeled_data_dir = os.path.abspath(tmp_proj_dir)

    # test should fail since each frame has an entry in the csv file
    assert not is_context_dataset(
        labeled_data_dir=labeled_data_dir,
        selected_frames_filename=SELECTED_FRAMES_FILENAME,
    )

    # remove final two entries to provide context
    csv_file = os.path.join(labeled_data_dir, LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME)
    img_files = np.genfromtxt(csv_file, delimiter=',', dtype=str)
    new_csv_file = csv_file.replace(".csv", ".tmp.csv")
    np.savetxt(new_csv_file, img_files[:-2], delimiter=",", fmt="%s")

    # test should pass since each frame has context (frame 00 will auto use 00 for negative frames)
    assert is_context_dataset(
        labeled_data_dir=labeled_data_dir,
        selected_frames_filename=os.path.basename(new_csv_file),
    )

    # test should fail if labeled frame directory does not exist
    assert not is_context_dataset(
        labeled_data_dir=os.path.join(labeled_data_dir, "nonexistent_directory"),
        selected_frames_filename=SELECTED_FRAMES_FILENAME,
    )


def test_compute_resize_dims():

    from lightning_pose_app.utilities import compute_resize_dims

    # 128 is minimum for resizing, since heatmaps will be smaller by a factor of 2
    assert compute_resize_dims(24) == 128
    assert compute_resize_dims(127) == 128

    # should round down to nearest power of 2 until 512, then rounds down to 384
    assert compute_resize_dims(129) == 128
    assert compute_resize_dims(250) == 128
    assert compute_resize_dims(380) == 256
    assert compute_resize_dims(385) == 256
    assert compute_resize_dims(512) == 384
    assert compute_resize_dims(1025) == 384
    assert compute_resize_dims(2049) == 384


def test_compute_batch_sizes():

    from lightning_pose_app.utilities import compute_batch_sizes

    train, dali_base, dali_cxt = compute_batch_sizes(400, 400)
    assert train == 16
    assert dali_base == 16
    assert dali_cxt == 16

    train, dali_base, dali_cxt = compute_batch_sizes(512, 512)
    assert train == 8
    assert dali_base == 8
    assert dali_cxt == 8

    train, dali_base, dali_cxt = compute_batch_sizes(1024, 1024)
    assert train == 6
    assert dali_base == 8
    assert dali_cxt == 8


def test_update_config():

    from lightning_pose_app.utilities import update_config

    # two fields deep
    cfg = {"data": {"csv_file": "path0", "data_dir": "path1"}}
    new = {"data": {"csv_file": "path2"}}
    cfg_new = update_config(cfg, new)
    assert cfg_new["data"]["csv_file"] == new["data"]["csv_file"]
    assert cfg_new["data"]["data_dir"] == cfg["data"]["data_dir"]

    # three fields deep
    cfg = {
        "eval": {
            "conf_thresh": 0.9,
            "fiftyone": {"port": 5151},
        },
    }
    new = {
        "eval": {
            "fiftyone": {"port": 1000},
        },
    }
    cfg_new = update_config(cfg, new)
    assert cfg_new["eval"]["fiftyone"]["port"] == new["eval"]["fiftyone"]["port"]
    assert cfg_new["eval"]["conf_thresh"] == cfg["eval"]["conf_thresh"]

    # four fields deep
    cfg = {
        "dali": {
            "general": {"seed": 123},
            "base": {"train": {"seq_len": 10}},
        },
    }
    new = {
        "dali": {
            "base": {"train": {"seq_len": 20}},
        },
    }
    cfg_new = update_config(cfg, new)
    assert cfg_new["dali"]["base"]["train"]["seq_len"] == new["dali"]["base"]["train"]["seq_len"]
    assert cfg_new["dali"]["general"]["seed"] == cfg_new["dali"]["general"]["seed"]


def test_abspath():

    from lightning_pose_app.utilities import abspath

    path1 = 'test/directory'
    abspath1 = abspath(path1)
    assert abspath1 == os.path.abspath(path1)

    path2 = '/test/directory'
    abspath2 = abspath(path2)
    assert abspath2 == os.path.abspath(path2[1:])
