import logging

import numpy as np
import pandas as pd
from scipy.stats import zscore

from lightning_pose_app.backend.video import compute_motion_energy_from_predection_df
from lightning_pose_app.utilities import run_kmeans

_logger = logging.getLogger('APP.BACKEND.EXTRACT_FRAMES')


def identify_outliers(
    metrics: dict,
    likelihood_thresh: float,
    thresh_metric_z: float,
) -> np.ndarray:
    # Initialize dictionary to store outlier flags for each metric
    outliers = {m: None for m in metrics.keys()}
    # Determine outliers for each metric
    for metric, val in metrics.items():
        if metric == "likelihood":
            outliers[metric] = val < likelihood_thresh
        else:
            # Apply z-score and threshold to determine outliers
            outliers[metric] = val.apply(zscore).abs() > thresh_metric_z
    # Combine outlier flags from all metrics into a single 3D array
    outliers_all = np.concatenate(
        [d.to_numpy()[:, :, None] for _, d in outliers.items()],
        axis=-1,
    )  # Shape: (n_frames, n_keypoints, n_metrics)
    # Sum outlier flags to identify total outliers for each frame
    outliers_total = np.sum(outliers_all, axis=(1, 2))
    return outliers_total


def select_max_frame_per_cluster(df: pd.DataFrame) -> list:
    # Copy the DataFrame to avoid modifying the original
    df_copy = df.copy()
    # Group by 'cluster_labels' and find the index of the max 'error score' in each group
    idxs_max_error = df_copy.groupby('cluster_labels')['error score'].idxmax()
    # Select the rows that correspond to the max 'error score' in each cluster
    final_selection = df_copy.loc[idxs_max_error].sort_values(
        by="frames index"
    )["frames index"].tolist()
    return final_selection


def select_frames_using_metrics(
    preds: pd.DataFrame,
    metrics: dict,
    n_frames_per_video: int,
    likelihood_thresh: float,
    thresh_metric_z: float,
) -> list:

    # Store likelihood scores in metrics dictionary
    mask = preds.columns.get_level_values('coords').isin(['likelihood'])
    metrics["likelihood"] = preds.loc[:, mask]

    # only look at frames with high motion energy
    me = compute_motion_energy_from_predection_df(preds, likelihood_thresh)
    me_prctile = 50 if preds.shape[0] < 1e5 else 75  # take fewer frames if there are many
    # Select index of high ME frames
    idxs_high_me = np.where(me > np.percentile(me, me_prctile))[0]
    for metric, val in metrics.items():
        metrics[metric] = val.loc[idxs_high_me]
    # identify outliers using various metrics
    outliers_total = identify_outliers(metrics, likelihood_thresh, thresh_metric_z)
    # grab the frames with the largest number of outliers
    frames_sample_multiplier = 10 if preds.shape[0] < 1e5 else 40
    frames_to_grab = min(n_frames_per_video * frames_sample_multiplier, preds.shape[0])
    outlier_frames = pd.DataFrame({
        "frames index": idxs_high_me,
        "error score": outliers_total,
    }).sort_values(by="error score", ascending=False).head(frames_to_grab)
    # Select frames with an error score greater than 0
    outlier_frames_nozero = outlier_frames[outlier_frames["error score"] > 0]
    # Prepare data for clustering
    outlier_idx = outlier_frames_nozero["frames index"].values
    mask = preds.columns.get_level_values('coords').isin(['x', 'y'])
    data_to_cluster = preds.loc[outlier_idx, mask]
    # drop all columns with NA in all cells and rows with NA in any cell
    data_to_cluster = data_to_cluster.dropna(axis=1, how='all').dropna(axis=0, how='any')
    # Run KMeans clustering
    cluster_labels, _ = run_kmeans(data_to_cluster.to_numpy(), n_frames_per_video)
    clustered_data = pd.DataFrame({
        "frames index": data_to_cluster.index,
        "cluster_labels": cluster_labels,
    })
    clustered_data_errors = clustered_data.merge(outlier_frames_nozero, on="frames index")
    # Select the frame with the maximum error score in each cluster for the final selection
    idxs_selected = select_max_frame_per_cluster(clustered_data_errors)
    return idxs_selected
