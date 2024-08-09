import argparse
import os

import numpy as np
import yaml
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.scripts import get_data_module, get_dataset, get_imgaug_transform
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig

from lightning_pose_app import MODEL_VIDEO_PREDS_INFER_DIR
from lightning_pose_app.backend.train_infer import inference_with_metrics, make_labeled_video
from lightning_pose_app.backend.video import (
    check_codec_format,
    copy_and_reformat_video,
    make_video_snippet,
)

# parse and check command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--video_file", type=str)
parser.add_argument("--compute_metrics", type=str, default="True")
parser.add_argument("--label_video_full", type=str, default="False")
parser.add_argument("--label_video_snippet", type=str, default="True")

args = parser.parse_args()

# argparse doesn't handle "type=bool" args the way you would expect
if (args.compute_metrics == "True") or (args.compute_metrics == "true"):
    compute_metrics = True
else:
    compute_metrics = False

if (args.label_video_full == "True") or (args.label_video_full == "true"):
    make_labeled_video_full = True
else:
    make_labeled_video_full = False

if (args.label_video_snippet == "True") or (args.label_video_snippet == "true"):
    make_labeled_video_clip = True
else:
    make_labeled_video_clip = False

model_dir = args.model_dir
if not os.path.exists(model_dir):
    raise NotADirectoryError(f"{model_dir} does not exist")

video_file = args.video_file
if not os.path.isfile(video_file):
    raise FileNotFoundError(f"{video_file} is not a valid video file")

# copy and reformat video if it is not the correct codec (h.264)
if not check_codec_format(video_file):
    reformatted_dir = os.path.dirname(video_file)
    os.makedirs(reformatted_dir, exist_ok=True)
    video_file = copy_and_reformat_video(
        video_file=args.video_file,
        dst_dir=reformatted_dir,
    )

# load config
config_file = os.path.join(model_dir, ".hydra/config.yaml")
if not os.path.exists(config_file):
    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError("Could not find config.yaml; has your model finished training?")
cfg = DictConfig(yaml.safe_load(open(config_file, "r")))

# define where predictions will be saved
pred_dir = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR)
preds_file = os.path.join(pred_dir, os.path.basename(video_file).replace(".mp4", ".csv"))

# create a data module that will be used to compute several metrics
if compute_metrics:
    # don't augment images
    cfg.training.imgaug = "default"
    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=cfg.data.data_dir, imgaug_transform=imgaug_transform)
    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=cfg.data.video_dir)
    data_module.setup()
else:
    # don't build data module
    data_module = None

# compute predictions
ckpt_file = ckpt_path_from_base_path(base_path=model_dir, model_name=cfg.model.model_name)
preds_df = inference_with_metrics(
    video_file=video_file,
    cfg=cfg,
    preds_file=preds_file,
    ckpt_file=ckpt_file,
    data_module=data_module,
    metrics=compute_metrics,
)

# output labeled videos
if make_labeled_video_full:
    make_labeled_video(
        video_file=video_file,
        preds_df=preds_df,
        save_file=preds_file.replace(".csv", ".labeled.mp4"),
        confidence_thresh=cfg.eval.confidence_thresh_for_vid,
    )

if make_labeled_video_clip:
    # make short labeled snippet for manual inspection
    video_file_snippet, snippet_start_idx, snippet_start_sec = make_video_snippet(
        video_file=video_file,
        preds_file=preds_file,
        clip_length=30,  # seconds
    )
    # run inference on short clip
    preds_file_snippet = video_file_snippet.replace(".mp4", ".csv")
    preds_df_snippet = inference_with_metrics(
        video_file=video_file_snippet,
        cfg=cfg,
        preds_file=preds_file_snippet,
        ckpt_file=ckpt_file,
        data_module=data_module,
        metrics=compute_metrics,
    )
    # create labeled video
    make_labeled_video(
        video_file=video_file_snippet,
        preds_df=preds_df_snippet,
        save_file=preds_file_snippet.replace(".csv", ".labeled.mp4"),
        confidence_thresh=cfg.eval.confidence_thresh_for_vid,
        video_start_time=snippet_start_sec,
    )
