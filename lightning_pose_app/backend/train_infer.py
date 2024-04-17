"""Functions for training and inference."""

import gc
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from lightning.app import LightningWork
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import (
    create_labeled_video,
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

from lightning_pose_app import (
    MODEL_VIDEO_PREDS_TRAIN_DIR,
)

_logger = logging.getLogger('APP.BACKEND.TRAIN_INFER')


class TrainerProgress(Callback):

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

    @rank_zero_only
    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        if self.update_train:
            progress = 100 * (trainer.current_epoch + 1) / float(trainer.max_epochs)
            self._update_progress(progress)

    @rank_zero_only
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
                labeled_mp4_file = os.path.join(
                    labeled_vid_dir, video_pred_name + "_labeled.mp4")
            else:
                labeled_mp4_file = None
            # predict on video
            self._inference_with_metrics_and_labeled_video(
                video_file=video_file,
                ckpt_file=best_ckpt,
                cfg=cfg,
                preds_file=prediction_csv_file,
                data_module=data_module_pred,
                trainer=trainer,
                make_labeled_video=True if labeled_mp4_file is not None else False,
                labeled_video_file=labeled_mp4_file,
                status_str=f"inference on video {v + 1} / {len(filenames)}",
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
    ckpt_file: str,
    cfg: DictConfig,
    preds_file: str,
    data_module: callable,
    trainer: pl.Trainer,
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

    return preds_df


def make_labeled_video(
    video_file: str,
    preds_df: pd.DataFrame,
    save_file: str,
    video_start_time: float = 0.0,
    confidence_thresh: float = 0.9,
):

    video_clip = VideoFileClip(video_file)

    keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
    xs_arr = keypoints_arr[:, :, 0]
    ys_arr = keypoints_arr[:, :, 1]
    mask_array = keypoints_arr[:, :, 2] > confidence_thresh
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # video generation
    create_labeled_video(
        clip=video_clip,
        xs_arr=xs_arr,
        ys_arr=ys_arr,
        mask_array=mask_array,
        filename=save_file,
        start_time=video_start_time,
    )
