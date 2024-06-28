import numpy as np
import os
import pandas as pd

from lightning_pose_app import (
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODELS_DIR,
)


def test_read_nth_frames(video_file):

    from lightning_pose_app.backend.extract_frames import read_nth_frames

    resize_dims = 8
    frames = read_nth_frames(video_file=video_file, n=10, resize_dims=resize_dims)
    assert frames.shape == (100, resize_dims, resize_dims)


def test_select_idxs_kmeans(video_file):

    from lightning_pose_app.backend.extract_frames import select_frame_idxs_kmeans

    resize_dims = 8
    n_clusters = 5
    idxs = select_frame_idxs_kmeans(
        video_file=video_file,
        resize_dims=resize_dims,
        n_frames_to_select=n_clusters,
        frame_skip=1,
    )
    assert len(idxs) == n_clusters


def test_identify_outliers():

    from lightning_pose_app.backend.extract_frames import identify_outliers

    likelihood_data = {
        "frame": [1, 2, 3, 4, 5],
        "paw1": [1.0, 1.0, 1.0, 1.0, 1.0],
        "paw2": [1.0, 1.0, 1.0, 1.0, 1.0],
        "paw3": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    pca_singleview_data = {
        "frame": [1, 2, 3, 4, 5],
        "paw1": [1.0, 1.0, 1.0, 10, 1.0],
        "paw2": [1.0, 1.0, 1.0, 10, 1.0],
        "paw3": [1.0, 1.0, 1.0, 10, 1.0],
    }
    likelihood_mock = pd.DataFrame(likelihood_data).set_index("frame")
    pca_singleview_mock = pd.DataFrame(pca_singleview_data).set_index("frame")
    pca_multiview_mock = pd.DataFrame(pca_singleview_data).set_index("frame")
    temp_norm_mock = pd.DataFrame(pca_singleview_data).set_index("frame")

    mock_metrics = {
        "likelihood": likelihood_mock,
        "pca_singleview": pca_singleview_mock,
        "pca_multiview": pca_multiview_mock,
        "temporal_norm": temp_norm_mock,
    }

    outlier_total = identify_outliers(mock_metrics, likelihood_thresh=1, thresh_metric_z=1)

    assert len(outlier_total) == likelihood_mock.shape[0]
    # check that the max index is on the fourth position = 3
    assert np.argmax(outlier_total) == 3
    # check that outlier score sums up to n_keypoints(=3) X (n_metrics iteams - 1)
    assert outlier_total[3] == likelihood_mock.shape[1] * (len(mock_metrics) - 1)


def test_select_max_frame_per_cluster():

    from lightning_pose_app.backend.extract_frames import select_max_frame_per_cluster

    df = pd.DataFrame({
        "frames index": [1, 2, 3, 4, 5, 6],
        "error score": [10, 10, 15, 15, 10, 10],
        "cluster_labels": [1, 1, 2, 1, 2, 2]
    })

    list_of_frames = select_max_frame_per_cluster(df)
    assert list_of_frames[0] == 3
    assert list_of_frames[1] == 4


def generate_mock_preds_for_al_testing(n_frames, keypoints, high_motion_frames=None):
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


def test_select_frames_using_metrics():

    from lightning_pose_app.backend.extract_frames import select_frames_using_metrics

    keypoints = ['paw1', 'paw2', 'paw3']
    n_frames = 50
    n_frames_per_video = 5

    # Generate mock data
    np.random.seed(0)
    pca_singleview_mock = mock_error_metrix_df(n_frames, keypoints)
    pca_multiview_mock = mock_error_metrix_df(n_frames, keypoints)
    temp_norm_mock = mock_error_metrix_df(n_frames, keypoints)
    preds_mock = generate_mock_preds_for_al_testing(n_frames, keypoints, high_motion_frames=None)

    metrics = {
        'likelihood': None,
        'pca_singleview': pca_singleview_mock,
        'pca_multiview': pca_multiview_mock,
        'temporal_norm': temp_norm_mock,
    }

    # Should return array of [1,11,21,31,41]
    idxs_selected = select_frames_using_metrics(
        preds_mock,
        metrics,
        n_frames_per_video,
        likelihood_thresh=0.0,
        thresh_metric_z=1.0,
    )

    assert len(idxs_selected) == n_frames_per_video
    assert idxs_selected[0] == 1


def test_select_frames_model(
    video_file, video_file_pred_df, video_file_pca_singleview_df, tmpdir,
):

    from lightning_pose_app.backend.extract_frames import select_frame_idxs_model

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
    n_frames_to_select = 7
    idxs = select_frame_idxs_model(
        video_file=video_file,
        model_dir=os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR),
        n_frames_to_select=n_frames_to_select,
        frame_range=[0, 1],
        thresh_metric_z=0.5,  # important! otherwise this setup doesn't pick up any outliers
    )
    assert len(idxs) == n_frames_to_select


def test_export_frames(video_file, tmpdir):

    from lightning_pose_app.backend.extract_frames import export_frames

    # multiple frames, no context
    save_dir_0 = os.path.join(str(tmpdir), 'labeled-frames-0')
    idxs = np.array([0, 2, 4, 6, 8, 10])
    export_frames(
        video_file=video_file,
        save_dir=save_dir_0,
        frame_idxs=idxs,
        context_frames=0,
    )
    assert len(os.listdir(save_dir_0)) == len(idxs)

    # multiple frames, 2-frame context
    save_dir_1 = os.path.join(str(tmpdir), 'labeled-frames-1')
    idxs = np.array([5, 10, 15, 20])
    export_frames(
        video_file=video_file,
        save_dir=save_dir_1,
        frame_idxs=idxs,
        context_frames=2,
    )
    assert len(os.listdir(save_dir_1)) == 5 * len(idxs)

    # single frame, 2-frame context
    save_dir_2 = os.path.join(str(tmpdir), 'labeled-frames-2')
    idxs = np.array([10])
    export_frames(
        video_file=video_file,
        save_dir=save_dir_2,
        frame_idxs=idxs,
        context_frames=2,
    )
    assert len(os.listdir(save_dir_2)) == 5 * len(idxs)


def test_find_contextual_frames():

    from lightning_pose_app.backend.extract_frames import find_contextual_frames

    test_cases = [
        {
            "input": [1, 4, 7, 2, 3, 9, 130],
            "expected_output": [1, 2, 3, 4, 7, 9, 130],
            "expected_is_context": False
        },
        {
            "input": [1, 2, 3, 4, 5, 11, 12, 13, 14],
            "expected_output": [1, 2, 3, 4, 5, 11, 12, 13, 14],
            "expected_is_context": False
        },
        {
            "input": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124,
            ],
            "expected_output": [3, 4, 5, 6, 7, 8, 102, 122],
            "expected_is_context": True
        },
        {
            "input": [11, 12, 13, 14, 15, 16, 17],
            "expected_output": [13, 14, 15],
            "expected_is_context": True
        }
    ]

    for case in test_cases:
        result, is_context = find_contextual_frames(case["input"])
        assert result == case["expected_output"], f"Failed for input: {case['input']}"
        assert is_context == case["expected_is_context"], f"Failed for input: {case['input']}"
