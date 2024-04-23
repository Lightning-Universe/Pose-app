"""Functions for training and inference."""

import datetime
import gc
import logging
import os
import shutil

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.app import LightningWork
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import (
    predict_dataset,
    predict_single_video,
)
from lightning_pose.utils.scripts import (
    calculate_train_batches,
    compute_metrics,
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from lightning_pose_app import MODEL_VIDEO_PREDS_TRAIN_DIR

_logger = logging.getLogger('APP.BACKEND.TRAIN_INFER')


class TrainerProgress(pl.callbacks.Callback):

    def __init__(self, work, update_train=True, update_inference=False):
        self.work = work
        self.update_train = update_train
        self.update_inference = update_inference
        self.progress_delta = 0.5

    def _update_progress(self, progress):
        if self.work.progress is None:
            if progress > self.progress_delta:
                self.work.progress = round(progress, 4)
        elif round(progress, 4) - self.work.progress >= self.progress_delta:
            if progress > 100:
                self.work.progress = 100.0
            else:
                self.work.progress = round(progress, 4)

    @pl.utilities.rank_zero_only
    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        if self.update_train:
            progress = 100 * (trainer.current_epoch + 1) / float(trainer.max_epochs)
            self._update_progress(progress)

    @pl.utilities.rank_zero_only
    def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        if self.update_inference:
            progress = \
                100 * (batch_idx + 1) / float(trainer.predict_loop.max_batches[dataloader_idx])
            self._update_progress(progress)


def train(
    cfg: DictConfig,
    results_dir: str,
    work: Optional[LightningWork] = None,  # for online progress updates
) -> None:

    # mimic hydra, change dir into results dir
    pwd = os.getcwd()
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    # ----------------------------------------------------------------------------------
    # set up data/model objects
    # ----------------------------------------------------------------------------------

    pretty_print_cfg(cfg)

    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)

    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

    # ----------------------------------------------------------------------------------
    # set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    callbacks = get_callbacks(cfg, early_stopping=False)
    # add callback to log progress
    if work:
        callbacks.append(TrainerProgress(work))

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg, dataset)

    # set up trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
    )

    # train model!
    trainer.fit(model=model, datamodule=data_module)

    # save config file
    cfg_file_local = os.path.join(results_dir, "config.yaml")
    with open(cfg_file_local, "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    # ----------------------------------------------------------------------------------
    # post-training analysis: labeled frames
    # ----------------------------------------------------------------------------------
    hydra_output_directory = os.getcwd()
    _logger.info("Hydra output directory: {}".format(hydra_output_directory))
    # get best ckpt
    best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)
    # check if best_ckpt is a file
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

    # make unaugmented data_loader if necessary
    if cfg.training.imgaug != "default":
        cfg_pred = cfg.copy()
        cfg_pred.training.imgaug = "default"
        imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
        dataset_pred = get_dataset(
            cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred)
        data_module_pred = get_data_module(
            cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
        data_module_pred.setup()
    else:
        data_module_pred = data_module

    # predict on all labeled frames (train/val/test)
    pretty_print_str("Predicting train/val/test images...")
    # compute and save frame-wise predictions
    preds_file = os.path.join(hydra_output_directory, "predictions.csv")
    predict_dataset(
        cfg=cfg,
        trainer=trainer,
        model=model,
        data_module=data_module_pred,
        ckpt_file=best_ckpt,
        preds_file=preds_file,
    )
    # compute and save various metrics
    try:
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
    except Exception as e:
        _logger.error(f"Error computing metrics\n{e}")

    # ----------------------------------------------------------------------------------
    # post-training analysis: unlabeled videos
    # ----------------------------------------------------------------------------------
    if cfg.eval.predict_vids_after_training:

        # make new trainer for inference so we can properly log inference progress
        if work:
            callbacks = TrainerProgress(work, update_inference=True)
        else:
            callbacks = []
        trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=callbacks)

        pretty_print_str("Predicting videos...")
        if cfg.eval.test_videos_directory is None:
            filenames = []
        else:
            filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
            pretty_print_str(
                f"Found {len(filenames)} videos to predict on "
                f"(in cfg.eval.test_videos_directory)"
            )

        for v, video_file in enumerate(filenames):
            assert os.path.isfile(video_file)
            pretty_print_str(f"Predicting video: {video_file}...")
            # get save name for prediction csv file
            video_pred_dir = os.path.join(hydra_output_directory, MODEL_VIDEO_PREDS_TRAIN_DIR)
            video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
            prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
            # get save name labeled video csv
            if cfg.eval.save_vids_after_training:
                labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
                labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + ".labeled.mp4")
            else:
                labeled_mp4_file = None
            # predict on video
            if work:
                work.status_ = f"inference on video {v + 1} / {len(filenames)}"
            preds_df = inference_with_metrics(
                video_file=video_file,
                ckpt_file=best_ckpt,
                cfg=cfg,
                preds_file=prediction_csv_file,
                data_module=data_module_pred,
                trainer=trainer,
            )
            # create labeled video
            if labeled_mp4_file:
                if work:
                    work.status_ = "creating labeled video"
                make_labeled_video(
                    video_file=video_file,
                    preds_df=preds_df,
                    save_file=labeled_mp4_file,
                    confidence_thresh=cfg.eval.confidence_thresh_for_vid,
                    work=work,
                )

    # ----------------------------------------------------------------------------------
    # clean up
    # ----------------------------------------------------------------------------------
    # remove lightning logs
    shutil.rmtree(os.path.join(results_dir, "lightning_logs"), ignore_errors=True)

    # change directory back
    os.chdir(pwd)

    # clean up memory
    del imgaug_transform
    del dataset
    del data_module
    del data_module_pred
    del loss_factories
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def inference_with_metrics(
    video_file: str,
    cfg: DictConfig,
    preds_file: str,
    ckpt_file: Optional[str] = None,
    data_module: Optional[callable] = None,
    trainer: Optional[pl.Trainer] = None,
) -> pd.DataFrame:

    # update video size in config
    video_clip = VideoFileClip(video_file)
    cfg.data.image_orig_dims.width = video_clip.w
    cfg.data.image_orig_dims.height = video_clip.h

    # compute predictions if they don't already exist
    if not os.path.exists(preds_file):
        preds_df = predict_single_video(
            video_file=video_file,
            ckpt_file=ckpt_file,
            cfg_file=cfg,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
        )
    else:
        preds_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # compute and save various metrics
    try:
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module)
    except Exception as e:
        _logger.error(f"Error predicting on {video_file}:\n{e}")

    video_clip.close()
    del video_clip
    gc.collect()

    return preds_df


def make_labeled_video(
    video_file: str,
    preds_df: pd.DataFrame,
    save_file: str,
    video_start_time: float = 0.0,
    confidence_thresh: float = 0.9,
    dotsize: int = 4,
    colormap: str = "cool",
    fps: Optional[float] = None,
    work: Optional[LightningWork] = None,  # for online progress updates
) -> None:
    """Produce video with predictions overlaid on frames.

    This is modified version of lightning_pose.utils.predictions.create_labeled_video that takes
    a LightningWork as an additional input in order to update a progress bar in the app.

    Parameters
    ----------
    video_file: absolute path to raw mp4 file
    preds_df: dataframe with predictions
    save_file: absolute path to labeled mp4 file
    video_start_time: time (in seconds) of video start (for clips)
    confidence_thresh: drop all markers with confidence below this threshold
    dotsize: size of marker dot on labeled video
    colormap: matplotlib color map for markers
    fps: None to default to fps of original video
    work: optional LightningWork whose progress attribute will be updated during video creation

    """

    if os.path.exists(save_file):
        return

    video_clip = VideoFileClip(video_file)

    # split predictions into markers and confidences
    keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
    xs_arr = keypoints_arr[:, :, 0]
    ys_arr = keypoints_arr[:, :, 1]
    n_frames, n_keypoints = xs_arr.shape

    # make masking array (do not render markers below the confidence threshold)
    mask_array = keypoints_arr[:, :, 2] > confidence_thresh
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if mask_array is None:
        mask_array = ~np.isnan(xs_arr)

    # set colormap for each color
    colors = make_cmap(n_keypoints, cmap=colormap)

    # extract info from clip
    nx, ny = video_clip.size
    dur = video_clip.duration - video_clip.start
    fps_og = video_clip.fps

    # upsample clip if low resolution; need to do this for dots and text to look nice
    if nx <= 100 or ny <= 100:
        upsample_factor = 2.5
    elif nx <= 192 or ny <= 192:
        upsample_factor = 2
    else:
        upsample_factor = 1

    if upsample_factor > 1:
        video_clip = video_clip.resize((upsample_factor * nx, upsample_factor * ny))
        nx, ny = video_clip.size

    print(f"Duration of video [s]: {np.round(dur, 2)}, recorded at {np.round(fps_og, 2)} fps!")

    def seconds_to_hms(seconds):
        # Convert seconds to a timedelta object
        td = datetime.timedelta(seconds=seconds)

        # Extract hours, minutes, and seconds from the timedelta object
        hours = td // datetime.timedelta(hours=1)
        minutes = (td // datetime.timedelta(minutes=1)) % 60
        seconds = td % datetime.timedelta(minutes=1)

        # Format the hours, minutes, and seconds into a string
        hms_str = f"{hours:02}:{minutes:02}:{seconds.seconds:02}"

        return hms_str

    # add marker to each frame t, where t is in sec
    def add_marker_and_timestamps(get_frame, t):
        image = get_frame(t * 1.0)
        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * 1.0 * fps_og))
        # ----------------
        # update progress
        # ----------------
        if work:
            progress = 100.0 * t / dur
            progress_delta = 0.5
            if work.progress is None:
                if progress > progress_delta:
                    work.progress = round(progress, 4)
            elif round(progress, 4) - work.progress >= progress_delta:
                if progress > 100:
                    work.progress = 100.0
                else:
                    work.progress = round(progress, 4)
        # ----------------
        # markers
        # ----------------
        for bpindex in range(n_keypoints):
            if index >= n_frames:
                print("Skipped frame {}, marker {}".format(index, bpindex))
                continue
            if mask_array[index, bpindex]:
                xc = min(int(upsample_factor * xs_arr[index, bpindex]), nx - 1)
                yc = min(int(upsample_factor * ys_arr[index, bpindex]), ny - 1)
                frame = cv2.circle(
                    frame,
                    center=(xc, yc),
                    radius=dotsize,
                    color=colors[bpindex].tolist(),
                    thickness=-1,
                )
        # ----------------
        # timestamps
        # ----------------
        seconds_from_start = t + video_start_time
        time_from_start = seconds_to_hms(seconds_from_start)
        idx_from_start = int(np.round(seconds_from_start * 1.0 * fps_og))
        text = f"t={time_from_start}, frame={idx_from_start}"
        # define text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        # calculate the size of the text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # calculate the position of the text in the upper-left corner
        offset = 6
        text_x = offset  # offset from the left
        text_y = text_size[1] + offset  # offset from the bottom
        # make black rectangle with a small padding of offset / 2 pixels
        cv2.rectangle(
            frame,
            (text_x - int(offset / 2), text_y + int(offset / 2)),
            (text_x + text_size[0] + int(offset / 2), text_y - text_size[1] - int(offset / 2)),
            (0, 0, 0),  # rectangle color
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # font color
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        return frame

    clip_marked = video_clip.fl(add_marker_and_timestamps)
    clip_marked.write_videofile(save_file, codec="libx264", fps=fps or fps_og or 20.0)

    # clean up memory
    clip_marked.close()
    del clip_marked
    video_clip.close()
    del video_clip
    gc.collect()


def make_cmap(number_colors: int, cmap: str = "cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors
