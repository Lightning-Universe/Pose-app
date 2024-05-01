"""UI for training models."""

import gc
import logging
import os
from datetime import datetime
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from eks.utils import convert_lp_dlc, make_output_dataframe, populate_output_dataframe
from eks.singleview_smoother import ensemble_kalman_smoother_single_view
from lightning.app import CloudCompute, LightningFlow, LightningWork
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.state import AppState
from omegaconf import DictConfig
from streamlit_autorefresh import st_autorefresh

from lightning_pose_app import (
    __version__,
    ENSEMBLE_MEMBER_FILENAME,
    LABELED_DATA_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_INFER_DIR,
    VIDEOS_TMP_DIR,
)
from lightning_pose_app.backend.train_infer import (
    TrainerProgress,
    inference_with_metrics,
    make_labeled_video,
    train,
)
from lightning_pose_app.backend.video import copy_and_reformat_video, make_video_snippet
from lightning_pose_app.build_configs import LitPoseBuildConfig
from lightning_pose_app.utilities import (
    StreamlitFrontend,
    abspath,
    is_context_dataset,
    update_config,
)

_logger = logging.getLogger('APP.TRAIN_INFER')

st.set_page_config(layout="wide")

# options for handling video labeling
VIDEO_LABEL_NONE = "Do not run inference on videos after training"
VIDEO_LABEL_INFER = "Run inference on videos"
VIDEO_LABEL_INFER_LABEL = "Run inference on videos and make labeled movie"

VIDEO_SELECT_NEW = "Upload new video(s)"
VIDEO_SELECT_UPLOADED = "Select previously uploaded video(s)"

MIN_TRAIN_FRAMES = 20


class LitPose(LightningWork):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # record progress of computationally-intensive steps (like training and inference)
        self.progress = 0.0

        # methods update status for fine-grained feedback during processing
        self.status_ = "initialized"

        # flag to communicate state of work to parent flow
        self.work_is_done = False

    def _train(
        self,
        config_file: str,
        config_overrides: dict,
        results_dir: str,
    ) -> None:

        # set flag for parent app
        self.work_is_done = False

        # load config
        cfg = DictConfig(yaml.safe_load(open(abspath(config_file), "r")))

        # update config with user-provided overrides
        cfg = update_config(cfg, config_overrides)

        self.status_ = "training"
        train(
            cfg=cfg,
            results_dir=results_dir,
            work=self,
        )

        # set flag for parent app
        self.work_is_done = True

    def _run_inference(
        self,
        model_dir: str,
        video_file: str,
        make_labeled_video_full: bool = False,
        make_labeled_video_clip: bool = False,
    ):

        from lightning_pose.utils.io import ckpt_path_from_base_path
        from lightning_pose.utils.scripts import get_data_module, get_dataset, get_imgaug_transform

        _logger.info(
            f"\n========= launching inference\nvideo: {video_file}\nmodel: {model_dir}\n========="
        )

        # set flag for parent app
        self.work_is_done = False

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
        self.status_ = "video inference"
        self.progress = 0.0
        preds_df = inference_with_metrics(
            video_file=video_file_abs,
            ckpt_file=ckpt_file,
            cfg=cfg,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
        )
        if make_labeled_video_full:
            self.status_ = "creating labeled video"
            self.progress = 0.0
            make_labeled_video(
                video_file=video_file_abs,
                preds_df=preds_df,
                save_file=preds_file.replace(".csv", ".labeled.mp4"),
                confidence_thresh=cfg.eval.confidence_thresh_for_vid,
                work=self,
            )

        # ----------------------------------------------------------------------------------
        # run inference on video clip; compute metrics; export labeled video
        # ----------------------------------------------------------------------------------
        if make_labeled_video_clip:
            self.status_ = "creating short clip"
            self.progress = 50.0  # will not dynamically update, but show user something is happing
            video_file_abs_short, clip_start_idx, clip_start_sec = make_video_snippet(
                video_file=video_file_abs,
                preds_file=preds_file,
            )
            self.status_ = "video inference (short clip)"
            self.progress = 0.0
            preds_file_short = preds_file.replace(".csv", ".short.csv")
            preds_df = inference_with_metrics(
                video_file=video_file_abs_short,
                ckpt_file=ckpt_file,
                cfg=cfg,
                preds_file=preds_file_short,
                data_module=data_module,
                trainer=trainer,
            )
            self.status_ = "creating labeled video (short clip)"
            self.progress = 0.0
            make_labeled_video(
                video_file=video_file_abs_short,
                preds_df=preds_df,
                save_file=preds_file_short.replace(".csv", ".labeled.mp4"),
                video_start_time=clip_start_sec,
                confidence_thresh=cfg.eval.confidence_thresh_for_vid,
                work=self,
            )

        # clean up memory
        del imgaug_transform
        del dataset
        del data_module
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        # set flag for parent app
        self.work_is_done = True

    def _run_eks(
        self,
        ensemble_dir: str,
        model_dirs: list,
        video_file: str,
        make_labeled_video_full: bool = False,
        make_labeled_video_clip: bool = False,
        keypoints_to_smooth: Optional[list] = None,
        smooth_param: Optional[float] = None,
        **kwargs
    ) -> None:

        # set flag for parent app
        self.work_is_done = False

        # -----------------------------------------
        # handle inputs
        # -----------------------------------------
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

        video_name = os.path.basename(video_file)

        # -----------------------------------------
        # load predictions from each model
        # -----------------------------------------
        csv_files = []
        for model_dir in model_dirs:
            pred_file = abspath(os.path.join(
                model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name.replace(".mp4", ".csv")
            ))
            csv_files.append(pred_file)

        dfs = []
        for file_path in csv_files:
            try:
                preds_df = pd.read_csv(file_path, index_col=0, header=[0, 1, 2])
                _logger.info(f"Successfully loaded DataFrame from {file_path}")
                keypoint_names = [c[1] for c in preds_df.columns[::3]]
                model_name = preds_df.columns[0][0]
                preds_df_fmt = convert_lp_dlc(preds_df, keypoint_names, model_name=model_name)
                dfs.append(preds_df_fmt)
            except Exception as e:
                _logger.exception(f"Failed to load DataFrame from {file_path}: {e}")

        keypoints_to_smooth = keypoints_to_smooth or keypoint_names

        # -----------------------------------------
        # run eks
        # -----------------------------------------
        self.status_ = "running eks"
        self.progress = 0.0

        # make empty dataframe for eks outputs
        df_eks = make_output_dataframe(preds_df)  # make from unformatted dataframe

        # loop over keypoints; apply eks to each individually
        for k, keypoint_name in enumerate(keypoints_to_smooth):
            # run eks
            keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
                markers_list=dfs,
                keypoint_ensemble=keypoint_name,
                smooth_param=smooth_param,  # default (None) is to compute automatically
            )
            keypoint_df = keypoint_df_dict[keypoint_name + '_df']  # make cleaner 2

            # put results into new dataframe
            df_eks = populate_output_dataframe(
                keypoint_df, 
                keypoint_name, 
                df_eks,
            )

            self.progress = (k + 1.0) / len(keypoints_to_smooth) * 100.0

        # -----------------------------------------
        # save eks outputs
        # -----------------------------------------
        preds_file = abspath(os.path.join(
            ensemble_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name.replace(".mp4", ".csv")
        ))
        os.makedirs(os.path.dirname(preds_file), exist_ok=True)
        df_eks.to_csv(preds_file)

        # -----------------------------------------
        # post-eks tasks
        # -----------------------------------------

        # compute metrics on eks
        data_module = None  # None for now; this means PCA metrics are not computed
        first_model_cfg_file = abspath(os.path.join(model_dirs[0], "config.yaml"))
        cfg = DictConfig(yaml.safe_load(open(first_model_cfg_file, "r")))
        self.status_ = "computing metrics"
        self.progress = 0.0
        preds_df = inference_with_metrics(
            video_file=video_file_abs,
            cfg=cfg,
            preds_file=preds_file,
            ckpt_file=None,
            data_module=data_module,
            trainer=None,
            metrics=True,
        )

        if make_labeled_video_full:
            self.status_ = "creating labeled video"
            self.progress = 0.0
            make_labeled_video(
                video_file=video_file_abs,
                preds_df=preds_df,
                save_file=preds_file.replace(".csv", ".labeled.mp4"),
                confidence_thresh=cfg.eval.confidence_thresh_for_vid,
                work=self,
            )

        if make_labeled_video_clip:
            # hack; rerun this function using the video clip from the first ensemble member
            self._run_eks(
                ensemble_dir=ensemble_dir,
                model_dirs=model_dirs,
                video_file=os.path.join(csv_files[0].replace(".csv", ".short.mp4")),
                make_labeled_video_full=True,
                make_labeled_video_clip=False,
                keypoints_to_smooth=keypoints_to_smooth,
                smooth_param=smooth_param,
            )

        # set flag for parent app
        self.work_is_done = True

    @staticmethod
    def _make_fiftyone_dataset(
        config_file: str,
        results_dir: str,
        config_overrides: Optional[dict] = None,
        **kwargs,
    ) -> None:

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

    def run(self, action: str, **kwargs) -> None:
        if action == "train":
            # don't fit model if training has already launched
            if os.path.exists(kwargs["results_dir"]):
                return
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
        elif action == "run_eks":
            proj_dir = '/'.join(kwargs["model_dir"].split('/')[:3])
            new_vid_file = copy_and_reformat_video(
                video_file=abspath(kwargs["video_file"]),
                dst_dir=abspath(os.path.join(proj_dir, VIDEOS_INFER_DIR)),
                remove_old=kwargs.pop("remove_old", True),
            )
            # save relative rather than absolute path
            kwargs["video_file"] = '/'.join(new_vid_file.split('/')[-4:])
            self._run_eks(**kwargs)


class TrainUI(LightningFlow):
    """UI to interact with training and inference."""

    def __init__(
        self,
        *args,
        allow_context: bool = True,
        max_epochs_default: int = 300,
        rng_seed_data_pt_default: int = 0,
        **kwargs
    ) -> None:

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
        self.rng_seed_data_pt_default = rng_seed_data_pt_default
        # flag; used internally and externally
        self.run_script_train = False
        # track number of times user hits buttons; used internally and externally
        self.submit_count_train = 0

        # output from the UI
        self.st_train_status = {}  # 'none' | 'initialized' | 'active' | 'complete'
        self.st_losses = []  # for both semi-super and semi-super context models
        self.st_train_label_opt = None  # what to do with video evaluation
        self.st_max_epochs = None
        self.st_rng_seed_data_pt = [rng_seed_data_pt_default]
        self.st_train_flag = {
            "super": False,
            "semisuper": False,
            "super-ctx": False,
            "semisuper-ctx": False,
        }
        self.st_datetime = None

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

        # ------------------------
        # Ensembling
        # ------------------------
        self.work_is_done_eks = False

    def _train(
        self,
        config_filename: Optional[str] = None,
        video_dirname: str = VIDEOS_DIR,
        testing: bool = False,
        **kwargs
    ) -> None:

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

        try:
            from lightning_pose import __version__ as lightning_pose_version
        except ImportError:
            lightning_pose_version = "1.2.3"

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
                "lightning_pose_version": lightning_pose_version,
                "lightning_pose_app_version": __version__,
            },
            "training": {
                "imgaug": "dlc",
                "check_val_every_n_epoch": 10,
                "log_every_n_steps": 5,
                "max_epochs": self.st_max_epochs,
            }
        }

        # update logging for edge cases where we want to train with very little data
        if self.n_labeled_frames <= 20:
            config_overrides["training"]["log_every_n_steps"] = 1
            config_overrides["training"]["train_batch_size"] = 1
        if self.st_max_epochs < 10:
            config_overrides["training"]["check_val_every_n_epoch"] = 1
            config_overrides["training"]["log_every_n_steps"] = 1

        # populate status dict
        # this dict controls execution logic as well as progress bar updates in the UI
        # cannot do this in the other nested for loops otherwise not all entries are initialized
        # at the same time, even if called before launching the works
        for m, (model_type, train_flag) in enumerate(self.st_train_flag.items()):
            for rng in self.st_rng_seed_data_pt:
                if testing:
                    model_type += "_PYTEST"
                model_name = f"{self.st_datetime[:-2]}{m:02}_{model_type}-{rng}"
                if model_name not in self.st_train_status.keys():
                    self.st_train_status[model_name] = "initialized" if train_flag else "none"

        # train models
        for model_name, status in self.st_train_status.items():
            if status == "initialized" or status == "active":
                self.st_train_status[model_name] = "active"

                # update config
                config_overrides["training"]["rng_seed_data_pt"] = int(model_name.split("-")[-1])
                if model_name.find("semi") > -1:
                    config_overrides["model"]["losses_to_use"] = self.st_losses
                else:
                    config_overrides["model"]["losses_to_use"] = []
                if model_name.find("ctx") > -1:
                    config_overrides["model"]["model_type"] = "heatmap_mhcrnn"

                # start training
                self.work.run(
                    action="train",
                    config_file=os.path.join(self.proj_dir, config_filename),
                    config_overrides=config_overrides,
                    results_dir=os.path.join(base_dir, MODELS_DIR, model_name),
                )

                # post-training cleanup
                self.st_train_status[model_name] = "complete"
                self.work.progress = 0.0  # reset for next model

        self.submit_count_train += 1

    def _launch_works(
        self,
        action: str,
        model_dirs: list,
        video_files: list,
        work_kwargs: dict,
        testing: bool = False,
        **kwargs
    ) -> None:

        # launch works (sequentially for now)
        for model_dir in model_dirs:
            for video_file in video_files:
                # combine model and video; keys cannot contain "."
                worker_key = f"{video_file.replace('.', '_')}---{model_dir}"
                if worker_key not in self.works_dict.keys():
                    self.works_dict[worker_key] = LitPose(
                        cloud_compute=CloudCompute("gpu"),
                        parallel=is_running_in_cloud(),
                    )
                status = self.st_infer_status[worker_key]
                if status == "initialized" or status == "active":
                    self.st_infer_status[worker_key] = "active"
                    # launch single worker for this model/video combo
                    if testing:
                        remove_old = False
                    else:
                        remove_old = VIDEOS_TMP_DIR in video_file  # only remove tmp files
                    self.works_dict[worker_key].run(
                        action=action,
                        model_dir=model_dir,  # used by inference, ignored by eks
                        video_file="/" + video_file,
                        remove_old=remove_old,
                        **work_kwargs,
                    )
                    self.st_infer_status[worker_key] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for worker_key in list(self.works_dict):
                if (worker_key in self.works_dict.keys()) \
                        and self.works_dict[worker_key].work_is_done:
                    # kill work
                    _logger.info(f"killing worker {worker_key}")
                    if not testing:  # cannot run stop() from tests for some reason
                        self.works_dict[worker_key].stop()
                    del self.works_dict[worker_key]

    def _run_inference(
        self,
        model_dirs: Optional[list] = None,
        video_files: Optional[list] = None,
        testing: bool = False,
        **kwargs
    ) -> None:

        self.work_is_done_inference = False

        if not model_dirs:
            model_dirs = [os.path.join(self.proj_dir, MODELS_DIR, self.st_inference_model)]
        if not video_files:
            video_files = self.st_inference_videos

        # populate status dict
        # this dict controls execution logic as well as progress bar updates in the UI
        # cannot do this in the other nested for loops otherwise not all entries are initialized
        # at the same time, even if called before launching the works
        for model_dir in model_dirs:
            for video_file in video_files:
                # combine model and video; keys cannot contain "."
                worker_key = f"{video_file.replace('.', '_')}---{model_dir}"
                if worker_key not in self.st_infer_status.keys():
                    self.st_infer_status[worker_key] = "initialized"

        work_kwargs = {
            "make_labeled_video_full": self.st_label_full,
            "make_labeled_video_clip": self.st_label_short,
        }
        self._launch_works(
            action="run_inference",
            model_dirs=model_dirs,
            video_files=video_files,
            testing=testing,
            work_kwargs=work_kwargs,
        )

        # set flag for parent app
        self.work_is_done_inference = True

    def _run_eks(
        self,
        ensemble_dir: str,
        model_dirs: list,
        video_files: Optional[list] = None,
        testing: bool = False,
        **kwargs
    ) -> None:

        self.work_is_done_eks = False

        if not video_files:
            video_files = self.st_inference_videos

        # populate status dict
        # this dict controls execution logic as well as progress bar updates in the UI
        # cannot do this in the other nested for loops otherwise not all entries are initialized
        # at the same time, even if called before launching the works
        for model_dir in [ensemble_dir]:
            for video_file in video_files:
                # combine model and video; keys cannot contain "."
                worker_key = f"{video_file.replace('.', '_')}---{model_dir}"
                if worker_key not in self.st_infer_status.keys():
                    self.st_infer_status[worker_key] = "initialized"

        work_kwargs = {
            "ensemble_dir": ensemble_dir,
            "model_dirs": model_dirs,
            "make_labeled_video_full": self.st_label_full,
            "make_labeled_video_clip": self.st_label_short,
            "smooth_param": kwargs.get("smooth_param", None)
        }
        self._launch_works(
            action="run_eks",
            model_dirs=[ensemble_dir],  # just loop over ensemble dir for each video
            video_files=video_files,
            testing=testing,
            work_kwargs=work_kwargs,
        )

        self.work_is_done_eks = True

    def _determine_dataset_type(self, **kwargs):
        """Check if labeled data directory contains context frames."""
        self.allow_context = is_context_dataset(
            labeled_data_dir=os.path.join(abspath(self.proj_dir), LABELED_DATA_DIR),
            selected_frames_filename=SELECTED_FRAMES_FILENAME,
        )

    def run(self, action: str, **kwargs) -> None:
        if action == "train":
            self._train(**kwargs)
        elif action == "run_inference":

            # check to see if we have a single model or an ensemble
            default_model_dir = os.path.join(self.proj_dir, MODELS_DIR, self.st_inference_model)
            model_dir = kwargs.pop("model_dir", default_model_dir)

            if ENSEMBLE_MEMBER_FILENAME not in os.listdir(abspath(model_dir)):
                # single model
                self._run_inference(model_dirs=[model_dir], **kwargs)
            else:

                ensemble_list_file = os.path.join(model_dir, ENSEMBLE_MEMBER_FILENAME)
                with open(abspath(ensemble_list_file), "r") as file:
                    model_dirs = [line.strip() for line in file.readlines()]

                # IMPORTANT: since '_run_inference' and '_run_eks' are not the actual 'run' method
                # they will not be cached.
                # Therefore these 'if not' statements prevent these functions from being run more
                # than once.
                # It is important that the Streamlit function set these flags to False upon the
                # button click to ensure a 'True' value is not stored from a previous run.

                # run inference with ensemble
                if not self.work_is_done_inference:
                    self._run_inference(model_dirs=model_dirs, **kwargs)

                # run eks on ensemble output
                if not self.work_is_done_eks:
                    self._run_eks(ensemble_dir=model_dir, model_dirs=model_dirs, **kwargs)

        elif action == "determine_dataset_type":
            self._determine_dataset_type(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    train_tab, right_column = st.columns([1, 1])
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

        st.header("Train networks")

        st.markdown(
            """
            #### Training options
            """
        )
        expander = st.expander("Expand to adjust training parameters")
        # max epochs
        st_max_epochs = expander.text_input(
            "Set the max training epochs (all models)", value=state.max_epochs_default)

        st_rng_seed_data_pt = expander.text_input(
            "Set the seed(s) for all models (int or comma-separated string)",
            value=state.rng_seed_data_pt_default,
            help="By setting a seed or a list of seeds, you enable reproducible model training, "
                 "ensuring consistent results across different runs. Users can specify a single "
                 "integer for individual models or a list to train multiple networks "
                 "(e.g. 1,5,6,7) thereby enhancing flexibility and control over the training "
                 "process."
        )
        if isinstance(st_rng_seed_data_pt, int):
            st_rng_seed_data_pt = [st_rng_seed_data_pt]
        elif isinstance(st_rng_seed_data_pt, str):
            st_rng_seed_data_pt_ = []
            for rng in st_rng_seed_data_pt.split(","):
                try:
                    st_rng_seed_data_pt_.append(int(rng))
                except:
                    continue
            st_rng_seed_data_pt = np.unique(st_rng_seed_data_pt_).tolist()
        elif isinstance(st_rng_seed_data_pt, list):
            pass
        else:
            st.text("RNG seed must be a single integer or a list of comma-separated integers")

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

        # as of 04/2024 we are deprecating this option to streamline video handling
        # video handling will now be solely carried out by the inference tab
        # st.markdown(
        #     """
        #     #### Video handling options""",
        #     help="Choose if you want to automatically run inference on the videos uploaded for "
        #          "labeling. **Warning** : Video traces will not be available in the "
        #          "Video Diagnostics tab if you choose “Do not run inference”"
        # )
        # st_train_label_opt = st.radio(
        #     "",
        #     options=[VIDEO_LABEL_NONE, VIDEO_LABEL_INFER, VIDEO_LABEL_INFER_LABEL],
        #     label_visibility="collapsed",
        #     index=1,  # default to inference but no labeled movie
        # )
        st_train_label_opt = VIDEO_LABEL_NONE

        st.markdown(
            """
            #### Select models to train
            """
        )
        st_train_flag = {
            "super": st.checkbox("Supervised", value=True),
            "semisuper": st.checkbox("Semi-supervised", value=True),
            "super-ctx": st.checkbox("Supervised context", value=True)
            if state.allow_context else False,
            "semisuper-ctx": st.checkbox("Semi-supervised context", value=True)
            if state.allow_context else False,
        }

        st_submit_button_train = st.button(
            "Train models", 
            disabled=(
                state.n_labeled_frames < MIN_TRAIN_FRAMES
                or state.run_script_train
                or state.run_script_infer
            ),
        )

        # give user training updates
        if state.run_script_train:
            for m, status in state.st_train_status.items():
                if status != "none":
                    status_ = None  # define more detailed status info from work
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        p = state.work.progress
                        status_ = state.work.status_
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)

                    rng = m.split("-")[-1]
                    st.progress(
                        p / 100.0,
                        f"model: {m} (rng={rng})\n\n"
                        f"{status_ or status}: {int(p)}\% complete"
                    )

        if st_submit_button_train:
            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) \
                    and (st_train_flag["semisuper"] or st_train_flag["semisuper-ctx"]):
                st.warning("Must select at least one semi-supervised loss if training that model")
                st_submit_button_train = False

        if state.n_labeled_frames < MIN_TRAIN_FRAMES:
            st.warning(f"Must label at least {MIN_TRAIN_FRAMES} frames before training")
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
            state.st_rng_seed_data_pt = st_rng_seed_data_pt
            state.st_train_label_opt = st_train_label_opt
            state.st_train_flag = st_train_flag
            state.st_train_status = {}  # reset; the Flow will update this
            dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
            state.st_datetime = dtime

            # construct semi-supervised loss list
            semi_losses = []
            if st_loss_pcamv:
                semi_losses.append("pca_multiview")
            if st_loss_pcasv:
                semi_losses.append("pca_singleview")
            if st_loss_temp:
                semi_losses.append("temporal")
            state.st_losses = semi_losses

            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")

    with right_column:

        inference_container = st.container()
        with inference_container:
            st.header(
                body="Predict on new videos",
                help="Select your preferred inference model, then drag and drop your video "
                     "file(s). Monitor the upload progress bar and click **Run inference** "
                     "once uploads are complete. "
                     "After completion, labeled videos are created if requested "
                     "(see 'Video handling options' section below). "
                     "Once inference concludes for all videos, the "
                     "waiting for existing inference to finish' warning will disappear."
            )

            model_dir = st.selectbox(
                "Select model to run inference", sorted(state.trained_models, reverse=True))

            # upload video files
            video_dir = os.path.join(state.proj_dir[1:], VIDEOS_TMP_DIR)
            os.makedirs(video_dir, exist_ok=True)

            # allow user to select video through uploading or already-uploaded video
            video_select_option = st.radio(
                "Video selection",
                options=[
                    VIDEO_SELECT_NEW,
                    VIDEO_SELECT_UPLOADED,
                ],
            )

            if video_select_option == VIDEO_SELECT_NEW:

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

            elif video_select_option == VIDEO_SELECT_UPLOADED:

                uploaded_video_dir_train = os.path.join(state.proj_dir[1:], VIDEOS_DIR)
                list_train = []
                if os.path.isdir(uploaded_video_dir_train):
                    list_train = [
                        os.path.join(uploaded_video_dir_train, vid)
                        for vid in os.listdir(uploaded_video_dir_train)
                    ]

                uploaded_video_dir_infer = os.path.join(state.proj_dir[1:], VIDEOS_INFER_DIR)
                list_infer = []
                if os.path.isdir(uploaded_video_dir_infer):
                    list_infer = [
                        os.path.join(uploaded_video_dir_infer, vid)
                        for vid in os.listdir(uploaded_video_dir_infer)
                    ]

                st_videos = st.multiselect(
                    "Select video files",
                    list_train + list_infer,
                    help="Videos in the 'videos_infer' directory have been previously uploaded "
                         "for inference. "
                         "Videos in the 'videos' directory have been previously uploaded for "
                         "frame extraction.",
                    format_func=lambda x: "/".join(x.split("/")[-2:]),
                )

            # allow user to select labeled video option
            st.markdown(
                """
                #### Video labeling options""",
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
                disabled=(
                    len(st_videos) == 0
                    or state.run_script_train
                    or state.run_script_infer
                ),
            )
            if state.run_script_infer:
                # cannot directly call keys()?
                worker_keys = [k for k, _ in state.works_dict.items()]
                for worker_key, status in state.st_infer_status.items():
                    status_ = None  # more detailed status provided by work
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        if worker_key in worker_keys:
                            try:
                                p = state.works_dict[worker_key].progress
                                status_ = state.works_dict[worker_key].status_
                            except Exception:
                                p = 100.0  # if work is deleted while accessing
                        else:
                            p = 100.0
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)

                    vid_, model_ = worker_key.split("---")
                    model_ = "/".join(model_.split("/")[-2:])
                    vid_ = "/".join(vid_.split("/")[-2:])
                    st.progress(
                        p / 100.0,
                        f"model: {model_}\n\nvideo: {vid_}\n\n"
                        f"{status_ or status}: {int(p)}\% complete")

                st.warning("waiting for existing inference to finish")

            # Lightning way of returning the parameters
            if st_submit_button_infer:

                state.submit_count_infer += 1

                state.st_inference_model = model_dir
                state.st_inference_videos = st_videos
                state.st_infer_status = {}  # reset; the Flow will update this
                state.st_label_short = st_label_short
                state.st_label_full = st_label_full
                state.work_is_done_inference = False  # alert flow inference needs performing
                state.work_is_done_eks = False  # alert flow eks needs performing
                st.text("Request submitted!")
                state.run_script_infer = True  # must the last to prevent race condition

                # force rerun to show "waiting for existing..." message
                st_autorefresh(interval=2000, key="refresh_infer_ui_submitted")

        st.markdown("----")

        eks_tab = st.container()
        with eks_tab:

            st.header("Create an ensemble of models")
            selected_models = st.multiselect(
               "Select models for ensembling",
               sorted(state.trained_models, reverse=True),
               help="Select which models you want to create an new ensemble model",
            )
            eks_model_name = st.text_input(
                label="Add ensemble name",
                value="eks",
                help="Provide a name that will be appended to the data/time information."
            )
            eks_model_name = eks_model_name.replace(" ", "_")

            st_submit_button_eks = st.button(
                "Create ensemble",
                key="eks_unique_key_button",
                disabled=(
                    len(selected_models) < 2
                    or state.run_script_train
                    or state.run_script_infer
                ),
            )

            if st_submit_button_eks:

                model_abs_paths = [
                   os.path.join(state.proj_dir, MODELS_DIR, model_name)
                   for model_name in selected_models
                ]

                dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
                eks_dir = os.path.join(state.proj_dir[1:], MODELS_DIR, f"{dtime}_{eks_model_name}")

                ensemble_file = create_ensemble_directory(
                    ensemble_dir=eks_dir,
                    model_dirs=model_abs_paths,
                )

                if os.path.exists(ensemble_file):
                    st.text(f"Ensemble created!")

                st_autorefresh(interval=2000, key="refresh_eks_ui_submitted")


def create_ensemble_directory(ensemble_dir: str, model_dirs: list):

    # create a folder for the ensemble
    os.makedirs(ensemble_dir, exist_ok=True)

    # save model paths in a text file
    text_file_path = os.path.join(ensemble_dir, ENSEMBLE_MEMBER_FILENAME)
    with open(text_file_path, 'w') as file:
        file.writelines(f"{path}\n" for path in model_dirs)

    return text_file_path
