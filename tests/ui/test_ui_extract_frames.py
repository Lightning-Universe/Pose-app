from lightning.app import CloudCompute
import numpy as np
import os
import pandas as pd
import shutil

from lightning_pose_app import (
    LABELED_DATA_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_TMP_DIR,
)
from lightning_pose_app.backend.extract_frames import (
    export_frames,
    find_contextual_frames,
)
    

def test_extract_frames_work(
    video_file, video_file_pred_df, video_file_pca_singleview_df, tmpdir,
):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app.ui.extract_frames import ExtractFramesWork

    work = ExtractFramesWork(
        cloud_compute=CloudCompute("default"),
    )

    # -----------------
    # extract frames 0
    # -----------------
    # use "random" method
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    n_frames_per_video = 10
    work._extract_frames(
        method="random",
        video_file=video_file,
        proj_dir=proj_dir,
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) > n_frames_per_video
    assert os.path.exists(os.path.join(video_dir, SELECTED_FRAMES_FILENAME))
    assert work.work_is_done

    # -----------------
    # extract frames 1
    # -----------------
    # use "active" method
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-2')
    model_dir = os.path.join(proj_dir, MODELS_DIR, 'dd-mm-yy/hh-mm-ss')
    # save predictions
    path = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_file_pred_df.to_csv(path)
    # save metrics
    path = os.path.join(
        model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name + "_pca_singleview_error.csv"
    )
    video_file_pca_singleview_df.to_csv(path)
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    n_frames_per_video = 10
    work._extract_frames(
        method="active",
        video_file=video_file,
        proj_dir=proj_dir,
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
        model_dir=os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR),
        thresh_metric_z=0.5,  # important! otherwise this setup doesn't pick up any outliers
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) > n_frames_per_video
    assert os.path.exists(os.path.join(video_dir, SELECTED_FRAMES_FILENAME))
    assert work.work_is_done

    # -----------------
    # unzip frames 0
    # -----------------
    # extract some frames
    save_dir_1 = os.path.join(str(tmpdir), 'labeled-frames-1')
    idxs = np.array([5, 10, 15, 20])
    export_frames(
        video_file=video_file,
        save_dir=save_dir_1,
        frame_idxs=idxs,
        context_frames=2,
    )
    # zip up a subset of the frames
    n_frames_to_zip = 3
    frame_files = os.listdir(save_dir_1)
    new_vid_name = "TEST_VID_ZIPPED_FRAMES"
    dst = os.path.join(tmpdir, new_vid_name)
    os.makedirs(dst, exist_ok=True)
    files = []
    for f in range(n_frames_to_zip):
        src = os.path.join(save_dir_1, frame_files[f])
        shutil.copyfile(src, os.path.join(dst, frame_files[f]))
        files.append(frame_files[f])
    # make a csv file to accompany frames
    np.savetxt(
        os.path.join(dst, SELECTED_FRAMES_FILENAME),
        np.sort(files),
        delimiter=",",
        fmt="%s",
    )
    # zip it all up
    new_video_name = new_vid_name + "_NEW0"
    new_video_path = os.path.join(tmpdir, new_video_name)
    zipped_file = new_video_path + ".zip"
    shutil.make_archive(new_video_path, "zip", dst)

    # test unzip frames
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, new_video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    work.work_is_done = False
    work._unzip_frames(
        video_file=zipped_file,
        proj_dir=proj_dir,
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) == (n_frames_to_zip + 1)
    idx_file_abs = os.path.join(video_dir, SELECTED_FRAMES_FILENAME)
    assert os.path.exists(idx_file_abs)
    df = pd.read_csv(idx_file_abs, header=None)
    assert df.shape[0] == n_frames_to_zip
    assert work.work_is_done

    # -----------------
    # unzip frames 1
    # -----------------
    # make sure unzipping handles intermediate folder
    new_video_name = new_vid_name + "_NEW1"
    src = new_video_path
    dst = os.path.join(tmpdir, new_video_name, "intermediate_subdir")
    # os.makedirs(os.path.dirname(dst), exist_ok=True)  # need to create for path purposes
    shutil.copytree(src, dst)
    new_video_path = os.path.join(tmpdir, new_video_name)
    zipped_file = new_video_path + ".zip"
    shutil.make_archive(new_video_path, "zip", new_video_path)

    # test unzip frames
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, new_video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    work.work_is_done = False
    work._unzip_frames(
        video_file=zipped_file,
        proj_dir=proj_dir,
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) == (n_frames_to_zip + 1)
    idx_file_abs = os.path.join(video_dir, SELECTED_FRAMES_FILENAME)
    assert os.path.exists(idx_file_abs)
    df = pd.read_csv(idx_file_abs, header=None)
    assert df.shape[0] == n_frames_to_zip
    assert work.work_is_done

    # -----------------
    # cleanup
    # -----------------
    del work


def test_extract_frames_ui(root_dir, tmp_proj_dir):

    from lightning_pose_app.ui.extract_frames import ExtractFramesUI

    video_name = "test_vid_copy"
    video_file_ = video_name + ".mp4"
    video_file = os.path.join(tmp_proj_dir, VIDEOS_TMP_DIR, video_file_)

    flow = ExtractFramesUI()

    # set attributes
    flow.proj_dir = tmp_proj_dir
    flow.st_extract_status[video_file] = "initialized"

    # -------------------
    # extract frames
    # -------------------
    n_frames_per_video = 10
    flow.run(
        action="extract_frames",
        video_files=[video_file],
        n_frames_per_video=n_frames_per_video,
        testing=True,
    )

    # make sure flow attributes are properly cleaned up
    assert flow.st_extract_status[video_file] == "complete"
    assert len(flow.works_dict) == 0
    assert flow.work_is_done_extract_frames

    # make sure frames were extracted
    proj_dir_abs = os.path.join(root_dir, tmp_proj_dir)
    frame_dir_abs = os.path.join(proj_dir_abs, LABELED_DATA_DIR, video_name)
    idx_file_abs = os.path.join(frame_dir_abs, SELECTED_FRAMES_FILENAME)
    assert os.path.isfile(os.path.join(proj_dir_abs, VIDEOS_DIR, video_file_))
    assert os.path.isdir(frame_dir_abs)
    assert os.path.isfile(idx_file_abs)

    df = pd.read_csv(idx_file_abs, header=None)
    assert df.shape[0] == n_frames_per_video

    # -------------------
    # test find_contextual_frames
    # -------------------
     


    # -------------------
    # unzip frames
    # -------------------
    # TODO

    # -----------------
    # cleanup
    # -----------------
    del flow
