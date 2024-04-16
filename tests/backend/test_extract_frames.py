import numpy as np
import pandas as pd


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
        video_file=video_file, resize_dims=resize_dims, n_clusters=n_clusters, frame_skip=1,
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


def test_select_frames_using_metrics():

    from lightning_pose_app.backend.extract_frames import select_frames_using_metrics

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
        'pca_singleview': pca_singleview_mock,
        'pca_multiview': pca_multiview_mock,
        'temporal_norm': temp_norm_mock,
    }

    # Should return array of [1,11,21,31,41]
    idxs_selected = select_frames_using_metrics(
        preds_mock,
        metrics,
        n_frames_per_video,
        likelihood_thresh=0,
        thresh_metric_z=1
    )

    assert len(idxs_selected) == n_frames_per_video
    assert idxs_selected[0] == 1
