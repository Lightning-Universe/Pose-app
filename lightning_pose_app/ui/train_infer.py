"""UI for training models."""

import logging
import os
import shutil
from datetime import datetime
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from lightning.app import CloudCompute, LightningFlow, LightningWork
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.state import AppState
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from streamlit_autorefresh import st_autorefresh

from lightning_pose_app import (
    LABELED_DATA_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODEL_VIDEO_PREDS_TRAIN_DIR,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_INFER_DIR,
    VIDEOS_TMP_DIR,
)
from lightning_pose_app.build_configs import LitPoseBuildConfig
from lightning_pose_app.utilities import (
    StreamlitFrontend,
    abspath,
    copy_and_reformat_video,
    is_context_dataset,
    make_video_snippet,
    update_config,
)

_logger = logging.getLogger('APP.TRAIN_INFER')

st.set_page_config(layout="wide")

# options for handling video labeling
VIDEO_LABEL_NONE = "Do not run inference on videos after training"
VIDEO_LABEL_INFER = "Run inference on videos"
VIDEO_LABEL_INFER_LABEL = "Run inference on videos and make labeled movie"


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


class LitPose(LightningWork):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pwd = os.getcwd()
        self.progress = 0.0
        self.status_ = "initialized"

        self.work_is_done_training = False
        self.work_is_done_inference = False
        self.count = 0

    def _train(
        self,
        config_file: str,
        config_overrides: dict,
        results_dir: str,
    ) -> None:

        import gc

        import torch
        from lightning_pose.utils import pretty_print_cfg, pretty_print_str
        from lightning_pose.utils.io import (
            check_video_paths,
            return_absolute_data_paths,
            return_absolute_path,
        )
        from lightning_pose.utils.predictions import predict_dataset
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
        from omegaconf import DictConfig, OmegaConf

        self.work_is_done_training = False

        # ----------------------------------------------------------------------------------
        # set up config
        # ----------------------------------------------------------------------------------

        # load config
        cfg = DictConfig(yaml.safe_load(open(abspath(config_file), "r")))

        # update config with user-provided overrides
        cfg = update_config(cfg, config_overrides)

        # mimic hydra, change dir into results dir
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
        callbacks = get_callbacks(cfg)
        # add callback to log progress
        callbacks.append(TrainerProgress(self))

        # calculate number of batches for both labeled and unlabeled data per epoch
        limit_train_batches = calculate_train_batches(cfg, dataset)

        # set up trainer
        self.status_ = "training"
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
            callbacks = TrainerProgress(self, update_inference=True)
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
        # save config file
        cfg_file_local = os.path.join(results_dir, "config.yaml")
        with open(cfg_file_local, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)

        # remove lightning logs
        shutil.rmtree(os.path.join(results_dir, "lightning_logs"), ignore_errors=True)

        os.chdir(self.pwd)

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

        self.work_is_done_training = True

    def _run_inference(
        self,
        model_dir: str,
        video_file: str,
        make_labeled_video_full: bool,
        make_labeled_video_clip: bool,
    ):

        from lightning_pose.utils.io import ckpt_path_from_base_path
        from lightning_pose.utils.scripts import get_data_module, get_dataset, get_imgaug_transform

        print(f"========= launching inference\nvideo: {video_file}\nmodel: {model_dir}\n=========")

        # set flag for parent app
        self.work_is_done_inference = False

        # ----------------------------------------------------------------------------------
        # set up paths
        # ----------------------------------------------------------------------------------

        # check: does file exist?
        if not os.path.exists(video_file):
            video_file_abs = abspath(video_file)
        else:
            video_file_abs = video_file
        video_file_exists = os.path.exists(video_file_abs)
        _logger.info(f"video file exists? {video_file_exists}")
        if not video_file_exists:
            _logger.info("skipping inference")
            return

        # load config
        config_file = os.path.join(model_dir, "config.yaml")
        cfg = DictConfig(yaml.safe_load(open(abspath(config_file), "r")))
        cfg.training.imgaug = "default"  # don't do imgaug

        # define paths
        data_dir_abs = cfg.data.data_dir
        video_dir_abs = cfg.data.video_dir
        cfg.data.csv_file = os.path.join(data_dir_abs, cfg.data.csv_file)

        pred_dir = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR)
        preds_file = os.path.join(
            abspath(pred_dir), os.path.basename(video_file_abs).replace(".mp4", ".csv"))

        # ----------------------------------------------------------------------------------
        # set up data/model/trainer objects
        # ----------------------------------------------------------------------------------

        # imgaug transform
        imgaug_transform = get_imgaug_transform(cfg=cfg)

        # dataset
        dataset = get_dataset(cfg=cfg, data_dir=data_dir_abs, imgaug_transform=imgaug_transform)

        # datamodule; breaks up dataset into train/val/test
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir_abs)
        data_module.setup()

        ckpt_file = ckpt_path_from_base_path(
            base_path=abspath(model_dir), model_name=cfg.model.model_name
        )

        # add callback to log progress
        callbacks = TrainerProgress(self, update_inference=True)

        # set up trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=callbacks)

        # ----------------------------------------------------------------------------------
        # run inference on full video; compute metrics; export labeled video
        # ----------------------------------------------------------------------------------
        self._inference_with_metrics_and_labeled_video(
            video_file=video_file_abs,
            ckpt_file=ckpt_file,
            cfg=cfg,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
            make_labeled_video=make_labeled_video_full,
        )

        # ----------------------------------------------------------------------------------
        # run inference on video clip; compute metrics; export labeled video
        # ----------------------------------------------------------------------------------
        if make_labeled_video_clip:
            self.progress = 0.0  # reset progress so it will be updated during snippet inference
            self.status_ = "creating labeled video"
            video_file_abs_short, clip_start_idx, clip_start_sec = make_video_snippet(
                video_file=video_file_abs,
                preds_file=preds_file,
            )
            preds_file_short = preds_file.replace(".csv", ".short.csv")
            self._inference_with_metrics_and_labeled_video(
                video_file=video_file_abs_short,
                ckpt_file=ckpt_file,
                cfg=cfg,
                preds_file=preds_file_short,
                data_module=data_module,
                trainer=trainer,
                make_labeled_video=make_labeled_video_clip,
                video_start_time=clip_start_sec,
            )

        # set flag for parent app
        self.work_is_done_inference = True

    def _inference_with_metrics_and_labeled_video(
        self,
        video_file: str,
        ckpt_file: str,
        cfg: DictConfig,
        preds_file: str,
        data_module: callable,
        trainer: pl.Trainer,
        make_labeled_video: bool = False,
        labeled_video_file: Optional[str] = None,
        video_start_time: float = 0.0,
        status_str: str = "video inference",
    ) -> None:

        from lightning_pose.utils.predictions import create_labeled_video, predict_single_video
        from lightning_pose.utils.scripts import compute_metrics
        from moviepy.editor import VideoFileClip

        # update video size in config
        video_clip = VideoFileClip(video_file)
        cfg.data.image_orig_dims.width = video_clip.w
        cfg.data.image_orig_dims.height = video_clip.h

        # compute predictions if they don't already exist
        self.progress = 0.0
        self.status_ = status_str
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

        # export labeled video
        if make_labeled_video:
            self.status_ = "creating labeled video"
            keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
            xs_arr = keypoints_arr[:, :, 0]
            ys_arr = keypoints_arr[:, :, 1]
            mask_array = keypoints_arr[:, :, 2] > cfg.eval.confidence_thresh_for_vid
            filename = labeled_video_file or preds_file.replace(".csv", ".labeled.mp4")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            # video generation
            create_labeled_video(
                clip=video_clip,
                xs_arr=xs_arr,
                ys_arr=ys_arr,
                mask_array=mask_array,
                filename=filename,
                start_time=video_start_time,
            )

    @staticmethod
    def _make_fiftyone_dataset(config_file, results_dir, config_overrides=None, **kwargs):

        from lightning_pose.utils.fiftyone import FiftyOneImagePlotter, check_dataset
        from omegaconf import DictConfig

        # load config (absolute path)
        cfg = DictConfig(yaml.safe_load(open(abspath(config_file), "r")))

        # update config with user-provided overrides (this is mostly for unit testing)
        for key1, val1 in config_overrides.items():
            for key2, val2 in val1.items():
                cfg[key1][key2] = val2

        # edit config
        cfg.data.data_dir = os.path.join(os.getcwd(), cfg.data.data_dir)
        model_dir = "/".join(results_dir.split("/")[-2:])
        # get project name from config file, assuming first part is model_config_
        proj_name = os.path.basename(config_file).split(".")[0][13:]
        cfg.eval.fiftyone.dataset_name = f"{proj_name}_{model_dir}"
        cfg.eval.fiftyone.model_display_names = [model_dir.split("_")[-1]]
        cfg.eval.hydra_paths = [results_dir]

        # build dataset
        fo_plotting_instance = FiftyOneImagePlotter(cfg=cfg)
        dataset = fo_plotting_instance.create_dataset()
        # create metadata and print if there are problems
        check_dataset(dataset)
        # print the name of the dataset
        fo_plotting_instance.dataset_info_print()

    def run(self, action=None, **kwargs):
        if action == "train":
            self._train(**kwargs)
            self._make_fiftyone_dataset(**kwargs)
        elif action == "run_inference":
            proj_dir = '/'.join(kwargs["model_dir"].split('/')[:3])
            new_vid_file = copy_and_reformat_video(
                video_file=abspath(kwargs["video_file"]),
                dst_dir=abspath(os.path.join(proj_dir, VIDEOS_INFER_DIR)),
                remove_old=kwargs.pop("remove_old", True),
            )
            # save relative rather than absolute path
            kwargs["video_file"] = '/'.join(new_vid_file.split('/')[-4:])
            self._run_inference(**kwargs)


class TrainUI(LightningFlow):
    """UI to interact with training and inference."""

    def __init__(self, *args, allow_context=True, max_epochs_default=300, **kwargs):

        super().__init__(*args, **kwargs)

        # updated externally by parent app
        self.trained_models = []
        self.proj_dir = None
        self.config_dict = None
        self.n_labeled_frames = None
        self.n_total_frames = None

        # ------------------------
        # Training
        # ------------------------
        # for now models will be trained sequentially with a single work rather than in parallel
        self.work = LitPose(
            cloud_compute=CloudCompute("gpu"),
            cloud_build_config=LitPoseBuildConfig(),
        )
        self.allow_context = allow_context  # this will be updated if/when project is loaded
        self.max_epochs_default = max_epochs_default

        # flag; used internally and externally
        self.run_script_train = False
        # track number of times user hits buttons; used internally and externally
        self.submit_count_train = 0

        # output from the UI (all will be dicts with keys=models, except st_max_epochs)
        self.st_train_status = {}  # 'none' | 'initialized' | 'active' | 'complete'
        self.st_losses = {}
        self.st_datetimes = {}
        self.st_train_label_opt = None  # what to do with video evaluation
        self.st_max_epochs = None

        # ------------------------
        # Inference
        # ------------------------
        # works will be allocated once videos are uploaded
        self.works_dict = Dict()
        self.work_is_done_inference = False

        # flag; used internally and externally
        self.run_script_infer = False
        # track number of times user hits buttons; used internally and externally
        self.submit_count_infer = 0

        # output from the UI
        self.st_infer_status = {}  # 'initialized' | 'active' | 'complete'
        self.st_inference_model = None
        self.st_infer_label_opt = None  # what to do with video evaluation
        self.st_inference_videos = []
        self.st_label_short = True
        self.st_label_full = False

    def _train(self, config_filename=None, video_dirname=VIDEOS_DIR):

        if config_filename is None:
            _logger.error("config_filename must be specified for training models")

        # set config overrides
        base_dir = os.path.join(os.getcwd(), self.proj_dir[1:])

        if self.st_train_label_opt == VIDEO_LABEL_NONE:
            predict_vids = False
            save_vids = False
        elif self.st_train_label_opt == VIDEO_LABEL_INFER:
            predict_vids = True
            save_vids = False
        else:
            predict_vids = True
            save_vids = True

        config_overrides = {
            "data": {
                "data_dir": base_dir,
                "video_dir": os.path.join(base_dir, video_dirname),
            },
            "eval": {
                "test_videos_directory": os.path.join(base_dir, video_dirname),
                "predict_vids_after_training": predict_vids,
                "save_vids_after_training": save_vids,
            },
            "model": {  # update these below if necessary
                "model_type": "heatmap",
            },
            "training": {
                "imgaug": "dlc",
                "max_epochs": self.st_max_epochs,
            }
        }

        # train models
        for m in ["super", "semisuper", "super ctx", "semisuper ctx"]:
            status = self.st_train_status[m]
            if status == "initialized" or status == "active":
                self.st_train_status[m] = "active"
                config_overrides["model"]["losses_to_use"] = self.st_losses[m]
                if m.find("ctx") > -1:
                    config_overrides["model"]["model_type"] = "heatmap_mhcrnn"
                self.work.run(
                    action="train",
                    config_file=os.path.join(self.proj_dir, config_filename),
                    config_overrides=config_overrides,
                    results_dir=os.path.join(base_dir, MODELS_DIR, self.st_datetimes[m])
                )
                self.st_train_status[m] = "complete"
                self.work.progress = 0.0  # reset for next model

        self.submit_count_train += 1

    def _run_inference(self, model_dir=None, video_files=None, testing=False):

        self.work_is_done_inference = False

        if not model_dir:
            model_dir = os.path.join(self.proj_dir, MODELS_DIR, self.st_inference_model)
        if not video_files:
            video_files = self.st_inference_videos

        # launch works (sequentially for now)
        for video_file in video_files:
            video_key = video_file.replace(".", "_")  # keys cannot contain "."
            if video_key not in self.works_dict.keys():
                self.works_dict[video_key] = LitPose(
                    cloud_compute=CloudCompute("gpu"),
                    parallel=is_running_in_cloud(),
                )
            status = self.st_infer_status[video_file]
            if status == "initialized" or status == "active":
                self.st_infer_status[video_file] = "active"
                # run inference (automatically reformats video for DALI)
                self.works_dict[video_key].run(
                    action="run_inference",
                    model_dir=model_dir,
                    video_file="/" + video_file,
                    make_labeled_video_full=self.st_label_full,
                    make_labeled_video_clip=self.st_label_short,
                    remove_old=not testing,  # remove bad format file by default
                )
                self.st_infer_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done_inference:
                    # kill work
                    _logger.info(f"killing work from video {video_key}")
                    if not testing:  # cannot run stop() from tests for some reason
                        self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_inference = True

    def _determine_dataset_type(self, **kwargs):
        """Check if labeled data directory contains context frames."""
        self.allow_context = is_context_dataset(
            labeled_data_dir=os.path.join(abspath(self.proj_dir), LABELED_DATA_DIR),
            selected_frames_filename=SELECTED_FRAMES_FILENAME,
        )

    def run(self, action, **kwargs):
        if action == "train":
            self._train(**kwargs)
        elif action == "run_inference":
            self._run_inference(**kwargs)
        elif action == "determine_dataset_type":
            self._determine_dataset_type(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    # make a train tab and an inference tab
    train_tab, infer_tab = st.columns(2, gap="large")

    # add shadows around each column
    # box-shadow args: h-offset v-offset blur spread color
    st.markdown("""
        <style type="text/css">
        div[data-testid="column"] {
            box-shadow: 3px 3px 10px -1px rgb(0 0 0 / 20%);
            border-radius: 5px;
            padding: 2% 3% 3% 3%;
        }
        </style>
    """, unsafe_allow_html=True)

    # constantly refresh so that:
    # - labeled frames are updated
    # - training progress is updated
    if (state.n_labeled_frames != state.n_total_frames) \
            or state.run_script_train or state.run_script_infer:
        st_autorefresh(interval=2000, key="refresh_train_ui")

    # add a sidebar to show the labeling progress
    # Calculate percentage of frames labeled
    if state.n_total_frames == 0 or state.n_total_frames is None:
        labeling_progress = 0.0
    else:
        labeling_progress = state.n_labeled_frames / state.n_total_frames
    st.sidebar.markdown('### Labeling Progress')
    st.sidebar.progress(labeling_progress)
    st.sidebar.write(f"You have labeled {state.n_labeled_frames}/{state.n_total_frames} frames")

    st.sidebar.markdown("""### Existing models""")
    st.sidebar.selectbox("Browse", sorted(state.trained_models, reverse=True))
    st.sidebar.write("Proceed to next tabs to analyze previously trained models")

    with train_tab:

        st.header("Train Networks")

        st.markdown(
            """
            #### Training options
            """
        )
        # expander = st.expander("Change Defaults")
        expander = st.expander(
            "Expand to adjust maximum training epochs and select unsupervised losses")
        # max epochs
        st_max_epochs = expander.text_input(
            "Set the max training epochs (all models)", value=state.max_epochs_default)

        # unsupervised losses (semi-supervised only; only expose relevant losses)
        expander.write("Select losses for semi-supervised model")
        pcamv = state.config_dict["data"].get("mirrored_column_matches", [])
        if len(pcamv) > 0:
            st_loss_pcamv = expander.checkbox("PCA Multiview", value=True)
        else:
            st_loss_pcamv = False
        pcasv = state.config_dict["data"].get("columns_for_singleview_pca", [])
        if len(pcasv) > 0:
            st_loss_pcasv = expander.checkbox("Pose PCA", value=True)
        else:
            st_loss_pcasv = False
        st_loss_temp = expander.checkbox("Temporal", value=True)

        st.markdown(
            """
            #### Video handling options""",
            help="Choose if you want to automatically run inference on the videos uploaded for "
                 "labeling. **Warning** : Video traces will not be available in the "
                 "Video Diagnostics tab if you choose “Do not run inference”"
        )
        st_train_label_opt = st.radio(
            "",
            options=[VIDEO_LABEL_NONE, VIDEO_LABEL_INFER, VIDEO_LABEL_INFER_LABEL],
            label_visibility="collapsed",
            index=1,  # default to inference but no labeled movie
        )

        st.markdown(
            """
            #### Select models to train
            """
        )
        st_train_super = st.checkbox("Supervised", value=True)
        st_train_semisuper = st.checkbox("Semi-supervised", value=True)
        if state.allow_context:
            st_train_super_ctx = st.checkbox("Supervised context", value=True)
            st_train_semisuper_ctx = st.checkbox("Semi-supervised context", value=True)
        else:
            st_train_super_ctx = False
            st_train_semisuper_ctx = False

        st_submit_button_train = st.button("Train models", disabled=state.run_script_train)

        # give user training updates
        if state.run_script_train:
            for m in ["super", "semisuper", "super ctx", "semisuper ctx"]:
                if m in state.st_train_status.keys() and state.st_train_status[m] != "none":
                    status = state.st_train_status[m]
                    status_ = None  # more detailed status info from work
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        p = state.work.progress
                        status_ = state.work.status_
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)
                    st.progress(
                        p / 100.0, f"{m} progress ({status_ or status}: {int(p)}\% complete)"
                    )

        if st_submit_button_train:
            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) \
                    and (st_train_semisuper or st_train_semisuper_ctx):
                st.warning("Must select at least one semi-supervised loss if training that model")
                st_submit_button_train = False

        if state.submit_count_train > 0 \
                and not state.run_script_train \
                and not st_submit_button_train:
            proceed_str = "Training complete; see diagnostics in the following tabs."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

        if st_submit_button_train:
            # save streamlit options to flow object
            state.submit_count_train += 1
            state.st_max_epochs = int(st_max_epochs)
            state.st_train_label_opt = st_train_label_opt
            state.st_train_status = {
                "super": "initialized" if st_train_super else "none",
                "semisuper": "initialized" if st_train_semisuper else "none",
                "super ctx": "initialized" if st_train_super_ctx else "none",
                "semisuper ctx": "initialized" if st_train_semisuper_ctx else "none",
            }

            # construct semi-supervised loss list
            semi_losses = []
            if st_loss_pcamv:
                semi_losses.append("pca_multiview")
            if st_loss_pcasv:
                semi_losses.append("pca_singleview")
            if st_loss_temp:
                semi_losses.append("temporal")
            state.st_losses = {
                "super": [],
                "semisuper": semi_losses,
                "super ctx": [],
                "semisuper ctx": semi_losses,
            }

            # set model times
            st_datetimes = {}
            dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
            # force different datetimes
            for i in range(4):
                if i == 0:  # supervised model
                    st_datetimes["super"] = dtime[:-2] + "00_super"
                if i == 1:  # semi-supervised model
                    st_datetimes["semisuper"] = dtime[:-2] + "01_semisuper"
                if i == 2:  # supervised context model
                    st_datetimes["super ctx"] = dtime[:-2] + "02_super-ctx"
                if i == 3:  # semi-supervised context model
                    st_datetimes["semisuper ctx"] = dtime[:-2] + "03_semisuper-ctx"

            # NOTE: cannot set these dicts entry-by-entry in the above loop, o/w don't get set?
            state.st_datetimes = st_datetimes
            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")

    with infer_tab:

        st.header(
            body="Predict on New Videos",
            help="Select your preferred inference model, then drag and drop your video file(s). "
                 "Monitor the upload progress bar and click **Run inference** once uploads are "
                 "complete. "
                 "After completion, labeled videos are created if requested "
                 "(see 'Video handling options' section below). "
                 "Once inference concludes for all videos, the "
                 "'waiting for existing inference to finish' warning will disappear."
        )

        model_dir = st.selectbox(
            "Choose model to run inference", sorted(state.trained_models, reverse=True))

        # upload video files
        video_dir = os.path.join(state.proj_dir[1:], VIDEOS_TMP_DIR)
        os.makedirs(video_dir, exist_ok=True)

        # initialize the file uploader
        uploaded_files = st.file_uploader("Select video files", accept_multiple_files=True)

        # for each of the uploaded files
        st_videos = []
        for uploaded_file in uploaded_files:
            # read it
            bytes_data = uploaded_file.read()
            # name it
            filename = uploaded_file.name.replace(" ", "_")
            filepath = os.path.join(video_dir, filename)
            st_videos.append(filepath)
            if not state.run_script_infer:
                # write the content of the file to the path, but not while processing
                with open(filepath, "wb") as f:
                    f.write(bytes_data)

        # allow user to select labeled video option
        st.markdown(
            """
            #### Video handling options""",
            help="Select checkboxes to automatically save a labeled video (short clip or full "
                 "video or both) after inference is complete. The short clip contains the 30 "
                 "second period of highest motion energy in the predictions."
        )
        st_label_short = st.checkbox(
            "Save labeled video (30 second clip)", value=state.st_label_short,
        )
        st_label_full = st.checkbox(
            "Save labeled video (full video)", value=state.st_label_full,
        )

        st_submit_button_infer = st.button(
            "Run inference",
            disabled=len(st_videos) == 0 or state.run_script_infer,
        )
        if state.run_script_infer:
            keys = [k for k, _ in state.works_dict.items()]  # cannot directly call keys()?
            for vid, status in state.st_infer_status.items():
                status_ = None  # more detailed status provided by work
                if status == "initialized":
                    p = 0.0
                elif status == "active":
                    vid_ = vid.replace(".", "_")
                    if vid_ in keys:
                        try:
                            p = state.works_dict[vid_].progress
                            status_ = state.works_dict[vid_].status_
                        except:
                            p = 100.0  # if work is deleted while accessing
                    else:
                        p = 100.0
                elif status == "complete":
                    p = 100.0
                else:
                    st.text(status)
                st.progress(
                    p / 100.0, f"{vid} progress ({status_ or status}: {int(p)}\% complete)")
            st.warning("waiting for existing inference to finish")

        # Lightning way of returning the parameters
        if st_submit_button_infer:

            state.submit_count_infer += 1

            state.st_inference_model = model_dir
            state.st_inference_videos = st_videos
            state.st_infer_status = {s: "initialized" for s in st_videos}
            state.st_label_short = st_label_short
            state.st_label_full = st_label_full
            st.text("Request submitted!")
            state.run_script_infer = True  # must the last to prevent race condition

            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_infer_ui_submitted")
