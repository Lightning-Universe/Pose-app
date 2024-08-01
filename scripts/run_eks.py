import argparse
import os
import yaml

import numpy as np
import pandas as pd
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.utils import convert_lp_dlc, make_output_dataframe, populate_output_dataframe
from omegaconf import DictConfig
from typing import Optional

from lightning_pose_app import MODEL_VIDEO_PREDS_INFER_DIR
from lightning_pose_app.backend.train_infer import inference_with_metrics, make_labeled_video


def run_eks(
    save_dir: str,
    model_dirs: list,
    video_file: str,
    make_labeled_video_full: bool = True,
    make_labeled_video_clip: bool = False,
    keypoints_to_smooth: Optional[list] = None,
    smooth_param: Optional[float] = None,
):

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            raise NotADirectoryError(f"{model_dir} does not exist")

    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"{video_file} is not a valid video file")
    video_name = os.path.basename(video_file)

    # -----------------------------------------
    # load predictions from each model
    # -----------------------------------------
    csv_files = []
    for model_dir in model_dirs:
        pred_file = os.path.join(
            model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name.replace(".mp4", ".csv")
        )
        csv_files.append(pred_file)

    dfs = []
    for file_path in csv_files:
        try:
            preds_df = pd.read_csv(file_path, index_col=0, header=[0, 1, 2])
            keypoint_names = [c[1] for c in preds_df.columns[::3]]
            model_name = preds_df.columns[0][0]
            preds_df_fmt = convert_lp_dlc(preds_df, keypoint_names, model_name=model_name)
            dfs.append(preds_df_fmt)
        except Exception as e:
            print(f"Failed to load DataFrame from {file_path}: {e}")

    keypoints_to_smooth = keypoints_to_smooth or keypoint_names

    # -----------------------------------------
    # run eks
    # -----------------------------------------

    # make empty dataframe for eks outputs
    df_eks = make_output_dataframe(preds_df)  # make from unformatted dataframe

    # convert list of DataFrames to a 3D NumPy array
    data_arrays = [df.to_numpy() for df in dfs]
    markers_3d_array = np.stack(data_arrays, axis=0)

    # map keypoint names to keys in dfs and crop markers_3d_array
    keypoint_is = {}
    keys = []
    for i, col in enumerate(dfs[0].columns):
        keypoint_is[col] = i
    for part in keypoints_to_smooth:
        keys.append(keypoint_is[part + '_x'])
        keys.append(keypoint_is[part + '_y'])
        keys.append(keypoint_is[part + '_likelihood'])
    key_cols = np.array(keys)
    markers_3d_array = markers_3d_array[:, :, key_cols]

    # call the smoother function
    df_dicts, s_finals = ensemble_kalman_smoother_singlecam(
        markers_3d_array=markers_3d_array,
        bodypart_list=keypoints_to_smooth,
        smooth_param=smooth_param,  # default (None) is to compute automatically
        s_frames=[(0, min(10000, markers_3d_array.shape[0]))],  # optimize on first 10k frames
    )

    # put results into new dataframe
    for k, keypoint_name in enumerate(keypoints_to_smooth):
        keypoint_df = df_dicts[k][keypoint_name + '_df']
        df_eks = populate_output_dataframe(
            keypoint_df,
            keypoint_name,
            df_eks,
        )

    # -----------------------------------------
    # save eks outputs
    # -----------------------------------------
    preds_file = os.path.join(
        save_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name.replace(".mp4", ".csv")
    )
    os.makedirs(os.path.dirname(preds_file), exist_ok=True)
    df_eks.to_csv(preds_file)

    # -----------------------------------------
    # post-eks tasks
    # -----------------------------------------

    # compute metrics on eks
    data_module = None  # None for now; this means PCA metrics are not computed
    first_model_cfg_file = os.path.join(model_dirs[0], "config.yaml")
    cfg = DictConfig(yaml.safe_load(open(first_model_cfg_file, "r")))
    preds_df = inference_with_metrics(
        video_file=video_file,
        cfg=cfg,
        preds_file=preds_file,
        ckpt_file=None,
        data_module=data_module,
        trainer=None,
        metrics=True,
    )

    if make_labeled_video_full:
        make_labeled_video(
            video_file=video_file,
            preds_df=preds_df,
            save_file=preds_file.replace(".csv", ".labeled.mp4"),
            confidence_thresh=cfg.eval.confidence_thresh_for_vid,
            work=None,
        )

    if make_labeled_video_clip:
        print("Cannot currently make video clips for EKS")
        # The short clip made for each individual model can actually differ depending on model
        # predictions, so we cannot ensemble these as if they are all predictions from the same
        # clip
        #
        # # hack; rerun this function using the video clip from the first ensemble member
        # run_eks(
        #     save_dir=save_dir,
        #     model_dirs=model_dirs,
        #     video_file=os.path.join(csv_files[0].replace(".csv", ".short.mp4")),
        #     make_labeled_video_full=True,
        #     make_labeled_video_clip=False,
        #     keypoints_to_smooth=keypoints_to_smooth,
        #     smooth_param=smooth_param,
        # )

    return None


# parse and check command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir_list", type=str)
parser.add_argument("--video_file", type=str)
parser.add_argument("--compute_metrics", type=bool, default=True)
parser.add_argument("--label_video_full", type=bool, default=False)
parser.add_argument("--label_video_snippet", type=bool, default=True)
parser.add_argument("--eks_save_dir", type=str)

args = parser.parse_args()

run_eks(
    save_dir=args.eks_save_dir,
    model_dirs=args.model_dir_list.split(":"),
    video_file=args.video_file,
    make_labeled_video_full=args.label_video_full,
    make_labeled_video_clip=args.label_video_snippet,
    keypoints_to_smooth=None,  # default to smoothing all
    smooth_param=None,  # default to finding optimal smoothing param
)
