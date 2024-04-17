"""Functions for selecting frames to label from videos."""

import logging
import os

import cv2
import numpy as np
import pandas as pd
from lightning.app import LightningWork
from scipy.stats import zscore
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Optional

from lightning_pose_app.backend.video import (
    compute_motion_energy_from_predection_df,
    get_frames_from_idxs,
)
from lightning_pose_app.utilities import run_kmeans

_logger = logging.getLogger('APP.BACKEND.EXTRACT_FRAMES')


def read_nth_frames(
    video_file: str,
    n: int = 1,
    resize_dims: int = 64,
    work: Optional[LightningWork] = None,  # for online progress updates
) -> np.ndarray:

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        _logger.error(f"Error opening video file {video_file}")

    frames = []
    frame_counter = 0
    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    with tqdm(total=int(frame_total)) as pbar:
        while cap.isOpened():
            # Read the next frame
            ret, frame = cap.read()
            if ret:
                # If the frame was successfully read, then process it
                if frame_counter % n == 0:
                    frame_resize = cv2.resize(frame, (resize_dims, resize_dims))
                    frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
                    frames.append(frame_gray.astype(np.float16))
                frame_counter += 1
                progress = frame_counter / frame_total * 100.0
                # periodically update progress of worker if available
                if work is not None:
                    if round(progress, 4) - work.progress >= work.progress_delta:
                        if progress > 100:
                            work.progress = 100.0
                        else:
                            work.progress = round(progress, 4)
                pbar.update(1)
            else:
                # If we couldn't read a frame, we've probably reached the end
                break

    # When everything is done, release the video capture object
    cap.release()

    return np.array(frames)


def select_frame_idxs_kmeans(
    video_file: str,
    resize_dims: int = 64,
    n_frames_to_select: int = 20,
    frame_skip: int = 1,
    frame_range: list = [0, 1],
    work: Optional[LightningWork] = None,  # for online progress updates
) -> np.ndarray:

    # check inputs
    if frame_skip != 1:
        raise NotImplementedError
    assert frame_range[0] >= 0
    assert frame_range[1] <= 1

    # read all frames, reshape, chop off unwanted portions of beginning/end
    frames = read_nth_frames(
        video_file=video_file,
        n=frame_skip,
        resize_dims=resize_dims,
        work=work,
    )
    frame_count = frames.shape[0]
    beg_frame = int(float(frame_range[0]) * frame_count)
    end_frame = int(float(frame_range[1]) * frame_count) - 2  # leave room for context
    assert (end_frame - beg_frame) >= n_frames_to_select, "valid video segment too short!"
    batches = np.reshape(frames, (frames.shape[0], -1))[beg_frame:end_frame]

    # take temporal diffs
    _logger.info('computing motion energy...')
    me = np.concatenate([
        np.zeros((1, batches.shape[1])),
        np.diff(batches, axis=0)
    ])
    # take absolute values and sum over all pixels to get motion energy
    me = np.sum(np.abs(me), axis=1)

    # find high me frames, defined as those with me larger than nth percentile me
    prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
    idxs_high_me = np.where(me > np.percentile(me, prctile))[0]

    # compute pca over high me frames
    _logger.info('performing pca over high motion energy frames...')
    pca_obj = PCA(n_components=np.min([batches[idxs_high_me].shape[0], 32]))
    embedding = pca_obj.fit_transform(X=batches[idxs_high_me])
    del batches  # free up memory

    # cluster low-d pca embeddings
    _logger.info('performing kmeans clustering...')
    _, centers = run_kmeans(X=embedding, n_clusters=n_frames_to_select)
    # centers is initially of shape (n_clusters, n_pcs); reformat
    centers = centers.T[None, :]

    # find high me frame that is closest to each cluster center
    # embedding is shape (n_frames, n_pcs)
    # centers is shape (1, n_pcs, n_clusters)
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    # dists is shape (n_frames, n_clusters)
    idxs_prototypes_ = np.argmin(dists, axis=0)
    # now index into high me frames to get overall indices, add offset
    idxs_prototypes = idxs_high_me[idxs_prototypes_] + beg_frame

    return idxs_prototypes


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
    n_frames_to_select: int,
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
    frames_to_grab = min(n_frames_to_select * frames_sample_multiplier, preds.shape[0])
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
    cluster_labels, _ = run_kmeans(data_to_cluster.to_numpy(), n_frames_to_select)
    clustered_data = pd.DataFrame({
        "frames index": data_to_cluster.index,
        "cluster_labels": cluster_labels,
    })
    clustered_data_errors = clustered_data.merge(outlier_frames_nozero, on="frames index")
    # Select the frame with the maximum error score in each cluster for the final selection
    idxs_selected = select_max_frame_per_cluster(clustered_data_errors)
    return idxs_selected


def select_frame_idxs_model(
    video_file: str,
    model_dir: str,
    n_frames_to_select: int,
    frame_range: list = [0, 1],
    likelihood_thresh: float = 0.0,
    thresh_metric_z: float = 3.0,
) -> np.ndarray:

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    pred_file = os.path.join(model_dir, video_name + ".csv")
    preds = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)

    metrics = {
        'likelihood': None,
        'pca_singleview': None,
        'pca_multiview': None,
        'temporal_norm': None,
    }

    keys_to_remove = []
    for key in metrics.keys():
        if key == "pca_singleview":
            filename = os.path.join(model_dir, video_name + "_pca_singleview_error.csv")
        elif key == "pca_multiview":
            filename = os.path.join(model_dir, video_name + "_pca_multiview_error.csv")
        elif key == "temporal_norm":
            filename = os.path.join(model_dir, video_name + "_temporal_norm.csv")
        else:
            filename = 'None'

        if os.path.exists(filename):
            # update dict with dataframe
            metrics[key] = pd.read_csv(filename, index_col=0)
        else:
            # make sure we remove this key later
            keys_to_remove.append(key)

    # remove unused keys from metrics
    for key in keys_to_remove:
        metrics.pop(key)

    idxs_selected = select_frames_using_metrics(
        preds=preds,
        metrics=metrics,
        n_frames_to_select=n_frames_to_select,
        likelihood_thresh=likelihood_thresh,
        thresh_metric_z=thresh_metric_z,
    )

    return np.array(idxs_selected)


def export_frames(
    video_file: str,
    save_dir: str,
    frame_idxs: np.ndarray,
    format: str = "png",
    n_digits: int = 8,
    context_frames: int = 0,
) -> None:
    """

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    save_dir: absolute path to directory in which selected frames are saved
    frame_idxs: indices of frames to grab
    format: only "png" currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save

    """

    cap = cv2.VideoCapture(video_file)

    # expand frame_idxs to include context frames
    if context_frames > 0:
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)

    # load frames from video
    frames = get_frames_from_idxs(cap, frame_idxs)

    # save out frames
    os.makedirs(save_dir, exist_ok=True)
    for frame, idx in zip(frames, frame_idxs):
        cv2.imwrite(
            filename=os.path.join(save_dir, "img%s.%s" % (str(idx).zfill(n_digits), format)),
            img=frame[0],
        )
