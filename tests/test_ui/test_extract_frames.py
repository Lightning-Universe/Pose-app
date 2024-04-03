from lightning.app import CloudCompute
import numpy as np
import os
import pandas as pd
import shutil


from lightning_pose_app import (
    LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME, MODEL_VIDEO_PREDS_INFER_DIR
)
from lightning_pose_app import MODELS_DIR, VIDEOS_TMP_DIR, VIDEOS_DIR


def test_extract_frames_work(
    video_file, video_file_pred_df, video_file_pca_singleview_df , tmpdir
):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app.ui.extract_frames import ExtractFramesWork

    work = ExtractFramesWork(
        cloud_compute=CloudCompute("default"),
    )

    # -----------------
    # read frame function
    # -----------------
    resize_dims = 8
    frames = work._read_nth_frames(video_file, n=10, resize_dims=resize_dims)
    assert frames.shape == (100, resize_dims, resize_dims)

    # -----------------
    # select indices
    # -----------------
    n_clusters = 5
    idxs = work._select_frame_idxs(
        video_file, resize_dims=resize_dims, n_clusters=n_clusters, frame_skip=1,
    )
    assert len(idxs) == n_clusters

    # -----------------
    # select indices w/ model
    # -----------------
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-0')
    model_dir = os.path.join(proj_dir, MODELS_DIR, 'dd-mm-yy/hh-mm-ss')
    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    # save predictions
    path = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_file_pred_df.to_csv(path)
    # save metrics
    path = os.path.join(
        model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name + "_pca_singleview_error.csv"
    )
    video_file_pca_singleview_df.to_csv(path)
    # select frames
    n_frames_per_video = 7
    idxs = work._select_frame_idxs_using_model(
        video_file=video_file,
        proj_dir=proj_dir,
        model_dir=os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR),
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
        thresh_metric_z=0.5,  # important! otherwise this setup doesn't pick up any outliers
    )
    assert len(idxs) == n_frames_per_video

    # -----------------
    # export frames
    # -----------------
    save_dir_0 = os.path.join(str(tmpdir), 'labeled-frames-0')
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_0,
        frame_idxs=idxs,
        context_frames=0,  # no context
    )
    assert len(os.listdir(save_dir_0)) == len(idxs)

    save_dir_1 = os.path.join(str(tmpdir), 'labeled-frames-1')
    idxs = np.array([5, 10, 15, 20])
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_1,
        frame_idxs=idxs,
        context_frames=2,  # 2-frame context
    )
    assert len(os.listdir(save_dir_1)) == 5 * len(idxs)

    save_dir_2 = os.path.join(str(tmpdir), 'labeled-frames-2')
    idxs = np.array([10])  # try with single frame
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_2,
        frame_idxs=idxs,
        context_frames=2,  # 2-frame context
    )
    assert len(os.listdir(save_dir_2)) == 5 * len(idxs)

    # -----------------
    # extract frames
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
    assert work.work_is_done_extract_frames

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
    assert work.work_is_done_extract_frames

    # -----------------
    # unzip frames
    # -----------------
    # zip up a subset of the frames extracted from the previous test
    n_frames_to_zip = 5
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
    new_video_name = new_vid_name + "_NEW"
    new_video_path = os.path.join(tmpdir, new_video_name)
    zipped_file = new_video_path + ".zip"
    shutil.make_archive(new_video_path, "zip", dst)

    # test unzip frames
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, new_video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    work.work_is_done_extract_frames = False
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
    assert work.work_is_done_extract_frames

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
    # unzip frames
    # -------------------
    # TODO

    # -----------------
    # cleanup
    # -----------------
    del flow


def test_identify_outliers():

    from lightning_pose_app.ui.extract_frames import identify_outliers

    likelihood_data = {
        'frame': [1, 2, 3, 4, 5],
        'paw1': [1.0, 1.0, 1.0, 1.0, 1.0],
        'paw2': [1.0, 1.0, 1.0, 1.0, 1.0],
        'paw3': [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    pca_singleview_data = {
        'frame': [1, 2, 3, 4, 5],
        'paw1': [1.0, 1.0, 1.0, 10, 1.0],
        'paw2': [1.0, 1.0, 1.0, 10, 1.0],
        'paw3': [1.0, 1.0, 1.0, 10, 1.0],
    }
    likelihood_mock = pd.DataFrame(likelihood_data).set_index('frame')
    pca_singleview_mock = pd.DataFrame(pca_singleview_data).set_index('frame')
    pca_multiview_mock = pd.DataFrame(pca_singleview_data).set_index('frame')
    temp_norm_mock = pd.DataFrame(pca_singleview_data).set_index('frame')

    mock_metrics = {
        'likelihood': None,
        'pca_singleview': None,
        'pca_multiview': None,
        'temporal_norm': None,
    }
    mock_metrics['likelihood'] = likelihood_mock
    mock_metrics['pca_singleview'] = pca_singleview_mock
    mock_metrics['pca_multiview'] = pca_multiview_mock
    mock_metrics['temporal_norm'] = temp_norm_mock

    outlier_total = identify_outliers(mock_metrics, likelihood_thresh=1, thresh_metric_z=1)

    assert len(outlier_total) == likelihood_mock.shape[0]
    # check that the max index is on the fourth position = 3
    assert np.argmax(outlier_total) == 3
    # check that outlier score sums up to n_keypoints(=3) X (n_metrics iteams - 1)
    assert outlier_total[3] == likelihood_mock.shape[1] * (len(mock_metrics) - 1)


def test_select_max_frame_per_cluster():
    from lightning_pose_app.ui.extract_frames import select_max_frame_per_cluster

    df = pd.DataFrame({
        "frames index": [1, 2, 3, 4, 5, 6],
        "error score": [10, 10, 15, 15, 10, 10],
        "cluster_labels": [1, 1, 2, 1, 2, 2]
    })

    list_of_frames = select_max_frame_per_cluster(df)
    assert list_of_frames[0] == 3
    assert list_of_frames[1] == 4


# create a mock predictions files specially fits to test select_frames_using_metrics function
def generate_mock_preds_for_AL_testing(n_frames, keypoints, high_motion_frames=None):
    n_frames_per_group = 10
    # Initialize the DataFrame to hold the data
    columns = pd.MultiIndex.from_product(
        [keypoints, ['x', 'y', 'likelihood']],
        names=['keypoint', 'coords']
    )
    df = pd.DataFrame(index=range(n_frames), columns=columns).fillna(0.0)

    # Assign the same x, y values across all keypoints for each group of 10 frames
    for group_start in range(1, n_frames, n_frames_per_group):
        # Generate a single set of x, y values for the current group
        xy_values = np.random.rand(1, 2) * 0.1  # Low x, y values for simplicity

        # Assign these x, y values across all keypoints for all 10 frames in the group
        for keypoint in keypoints:
            df.loc[group_start:group_start + 9, (keypoint, 'x')] = xy_values[0, 0]
            df.loc[group_start:group_start + 9, (keypoint, 'y')] = xy_values[0, 1]
            # Set a constant likelihood of 1.0 for simplicity, but this can be varied if needed
            df.loc[group_start:group_start + 9, (keypoint, 'likelihood')] = 1.0
    return df


def mock_error_metrix_df(n_frames, keypoints):
    n_frames_per_group = 10
    # Initialize predictions with random values
    preds = np.random.rand(n_frames, len(keypoints))
    # Adjust predictions to ensure the first frame of each group has the highest score
    for group_start in range(0, n_frames, n_frames_per_group):
        # Find the maximum value in the group
        max_value_in_group = np.max(preds[group_start:group_start + 10])
        # Set the first frame of each group to have a slightly higher value than the max found,
        # ensuring it has the highest error score
        preds[group_start] = max_value_in_group + np.random.rand(1, len(keypoints)) * 0.1 + 0.9
    df = pd.DataFrame(preds, columns=keypoints)
    return df


# Should return array of [1,11,21,31,41]
def test_select_frames_using_metrics():

    from lightning_pose_app.ui.extract_frames import select_frames_using_metrics

    keypoints = ['paw1', 'paw2', 'paw3']
    n_frames = 50
    n_frames_per_video = 5

    # Generate mock data
    pca_singleview_mock = mock_error_metrix_df(n_frames, keypoints)
    pca_multiview_mock = mock_error_metrix_df(n_frames, keypoints)
    temp_norm_mock = mock_error_metrix_df(n_frames, keypoints)
    preds_mock = generate_mock_preds_for_AL_testing(n_frames, keypoints, high_motion_frames=None)

    metrics = {
        'likelihood': None,
        'pca_singleview': None,
        'pca_multiview': None,
        'temporal_norm': None,
    }
    metrics['pca_singleview'] = pca_singleview_mock
    metrics['pca_multiview'] = pca_multiview_mock
    metrics['temporal_norm'] = temp_norm_mock

    idxs_selected = select_frames_using_metrics(
        preds_mock,
        metrics,
        n_frames_per_video,
        likelihood_thresh=0,
        thresh_metric_z=1
    )

    assert len(idxs_selected) == n_frames_per_video
    assert idxs_selected[0] == 1
