"""UI for training models."""

from datetime import datetime
from lightning import CloudCompute, LightningFlow
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.state import AppState
from lightning.app.storage import FileSystem
from lightning.app.structures import Dict
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import lightning.pytorch as pl
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time
import yaml

from lightning_pose_app import LABELED_DATA_DIR, VIDEOS_DIR, VIDEOS_TMP_DIR, VIDEOS_INFER_DIR
from lightning_pose_app import MODELS_DIR, COLLECTED_DATA_FILENAME
from lightning_pose_app.build_configs import LitPoseBuildConfig
from lightning_pose_app.utilities import StreamlitFrontend, WorkWithFileSystem
from lightning_pose_app.utilities import reencode_video, check_codec_format

st.set_page_config(layout="wide")


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


class LitPose(WorkWithFileSystem):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, name="train_infer", **kwargs)

        self.pwd = os.getcwd()
        self.progress = 0.0

        self.work_is_done_training = False
        self.work_is_done_inference = False
        self.count = 0

    def _reformat_video(self, video_file, **kwargs):

        # get new names (ensure mp4 file extension, no tmp directory)
        ext = os.path.splitext(os.path.basename(video_file))[1]
        video_file_mp4_ext = video_file.replace(f"{ext}", ".mp4")
        video_file_new = video_file_mp4_ext.replace(VIDEOS_TMP_DIR, VIDEOS_INFER_DIR)
        video_file_abs_new = self.abspath(video_file_new)

        # check 0: do we even need to reformat?
        if self._drive.isfile(video_file_new):
            return video_file_new

        # pull videos from FileSystem
        self.get_from_drive([video_file])
        video_file_abs = self.abspath(video_file)

        # check 1: does file exist?
        video_file_exists = os.path.exists(video_file_abs)
        if not video_file_exists:
            print(f"{video_file_abs} does not exist! skipping")
            return None

        # check 2: is file in the correct format for DALI?
        video_file_correct_codec = check_codec_format(video_file_abs)

        # reencode/rename
        if not video_file_correct_codec:
            print("re-encoding video to be compatable with Lightning Pose video reader")
            reencode_video(video_file_abs, video_file_abs_new)
            # remove old video from local files
            os.remove(video_file_abs)
        else:
            # make dir to write into
            os.makedirs(os.path.dirname(video_file_abs_new), exist_ok=True)
            # rename
            os.rename(video_file_abs, video_file_abs_new)

        # remove old video(s) from FileSystem
        if self._drive.isfile(video_file):
            self._drive.rm(video_file)
        if self._drive.isfile(video_file_mp4_ext):
            self._drive.rm(video_file_mp4_ext)

        # push possibly reformated, renamed videos to FileSystem
        self.put_to_drive([video_file_new])

        return video_file_new

    def _train(self, inputs, outputs, cfg_overrides, results_dir):

        from omegaconf import DictConfig, OmegaConf
        from lightning_pose.utils import pretty_print_str, pretty_print_cfg
        from lightning_pose.utils.io import (
            check_video_paths,
            return_absolute_data_paths,
            return_absolute_path,
        )
        from lightning_pose.utils.predictions import predict_dataset
        from lightning_pose.utils.scripts import (
            export_predictions_and_labeled_video,
            get_data_module,
            get_dataset,
            get_imgaug_transform,
            get_loss_factories,
            get_model,
            get_callbacks,
            calculate_train_batches,
            compute_metrics,
        )

        self.work_is_done_training = False

        # ----------------------------------------------------------------------------------
        # Pull data from FileSystem
        # ----------------------------------------------------------------------------------

        # pull config, frames, labels, and videos (relative paths)
        self.get_from_drive(inputs)

        # load config (absolute path)
        for i in inputs:
            if i.endswith(".yaml"):
                config_file = i
        cfg = DictConfig(yaml.safe_load(open(self.abspath(config_file), "r")))

        # update config with user-provided overrides
        for key1, val1 in cfg_overrides.items():
            for key2, val2 in val1.items():
                cfg[key1][key2] = val2

        # mimic hydra, change dir into results dir
        os.makedirs(results_dir, exist_ok=True)
        os.chdir(results_dir)

        # ----------------------------------------------------------------------------------
        # Set up data/model objects
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
        # Set up and run training
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
        # Post-training analysis
        # ----------------------------------------------------------------------------------
        hydra_output_directory = os.getcwd()
        print("Hydra output directory: {}".format(hydra_output_directory))
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
            print(f"Error computing metrics\n{e}")

        # predict folder of videos
        if cfg.eval.predict_vids_after_training:
            pretty_print_str("Predicting videos...")
            if cfg.eval.test_videos_directory is None:
                filenames = []
            else:
                filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
                pretty_print_str(
                    f"Found {len(filenames)} videos to predict on "
                    f"(in cfg.eval.test_videos_directory)"
                )
            for video_file in filenames:
                assert os.path.isfile(video_file)
                pretty_print_str(f"Predicting video: {video_file}...")
                # get save name for prediction csv file
                video_pred_dir = os.path.join(hydra_output_directory, "video_preds")
                video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
                prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
                # get save name labeled video csv
                if cfg.eval.save_vids_after_training:
                    labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
                    labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + "_labeled.mp4")
                else:
                    labeled_mp4_file = None
                # predict on video
                export_predictions_and_labeled_video(
                    video_file=video_file,
                    cfg=cfg,
                    ckpt_file=best_ckpt,
                    prediction_csv_file=prediction_csv_file,
                    labeled_mp4_file=labeled_mp4_file,
                    trainer=trainer,
                    model=model,
                    data_module=data_module_pred,
                    save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
                )
                # compute and save various metrics
                try:
                    compute_metrics(
                        cfg=cfg, preds_file=prediction_csv_file, data_module=data_module_pred
                    )
                except Exception as e:
                    print(f"Error predicting on video {video_file}:\n{e}")
                    continue

        # ----------------------------------------------------------------------------------
        # Push results to FileSystem, clean up
        # ----------------------------------------------------------------------------------
        # save config file
        cfg_file_local = os.path.join(results_dir, "config.yaml")
        with open(cfg_file_local, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)

        os.chdir(self.pwd)
        self.put_to_drive(outputs)  # IMPORTANT! must come after changing directories
        self.work_is_done_training = True

    def _run_inference(self, model_dir, video_file):

        from omegaconf import DictConfig
        from lightning_pose.utils.io import ckpt_path_from_base_path
        from lightning_pose.utils.predictions import predict_single_video
        from lightning_pose.utils.scripts import (
            get_data_module,
            get_dataset,
            get_imgaug_transform,
            compute_metrics,
        )

        print(
            f"============ launching inference\nvideo: {video_file}\nmodel: {model_dir}\n"
            f"============"
        )

        # set flag for parent app
        self.work_is_done_inference = False

        # ----------------------------------------------------------------------------------
        # Pull data from FileSystem
        # ----------------------------------------------------------------------------------

        # pull video from FileSystem
        self.get_from_drive([video_file])

        # check: does file exist?
        video_file_abs = self.abspath(video_file)
        video_file_exists = os.path.exists(video_file_abs)
        print(f"video file exists? {video_file_exists}")
        if not video_file_exists:
            print("skipping inference")
            return

        # pull model from FileSystem
        self.get_from_drive([model_dir])

        # load config (absolute path)
        config_file = os.path.join(model_dir, "config.yaml")
        cfg = DictConfig(yaml.safe_load(open(self.abspath(config_file), "r")))
        cfg.training.imgaug = "default"  # don't do imgaug

        # define paths
        data_dir_abs = cfg.data.data_dir
        video_dir_abs = cfg.data.video_dir
        cfg.data.csv_file = os.path.join(data_dir_abs, cfg.data.csv_file)

        pred_dir = os.path.join(model_dir, "video_preds_infer")
        preds_file = os.path.join(
            self.abspath(pred_dir), os.path.basename(video_file_abs).replace(".mp4", ".csv"))

        # ----------------------------------------------------------------------------------
        # Set up data/model objects
        # ----------------------------------------------------------------------------------

        # imgaug transform
        imgaug_transform = get_imgaug_transform(cfg=cfg)

        # dataset
        dataset = get_dataset(cfg=cfg, data_dir=data_dir_abs, imgaug_transform=imgaug_transform)

        # datamodule; breaks up dataset into train/val/test
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir_abs)
        data_module.setup()

        ckpt_file = ckpt_path_from_base_path(
            base_path=self.abspath(model_dir), model_name=cfg.model.model_name
        )

        # ----------------------------------------------------------------------------------
        # Set up and run inference
        # ----------------------------------------------------------------------------------

        # add callback to log progress
        callbacks = TrainerProgress(self, update_inference=True)

        # set up trainer
        trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=callbacks)

        # compute predictions
        predict_single_video(
            video_file=video_file_abs,
            ckpt_file=ckpt_file,
            cfg_file=cfg,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
        )

        # compute and save various metrics
        try:
            compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module)
        except Exception as e:
            print(f"Error predicting on {video_file}:\n{e}")

        # ----------------------------------------------------------------------------------
        # Push results to FileSystem, clean up
        # ----------------------------------------------------------------------------------
        self.put_to_drive([pred_dir])

        # set flag for parent app
        self.work_is_done_inference = True

    def run(self, action=None, **kwargs):
        if action == "train":
            self._train(**kwargs)
        elif action == "run_inference":
            new_vid_file = self._reformat_video(**kwargs)
            kwargs["video_file"] = new_vid_file
            self._run_inference(**kwargs)


class TrainUI(LightningFlow):
    """UI to interact with training and inference."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # shared storage system
        self._drive = FileSystem()

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

        # flag; used internally and externally
        self.run_script_train = False
        # track number of times user hits buttons; used internally and externally
        self.submit_count_train = 0

        # output from the UI (all will be dicts with keys=models, except st_max_epochs)
        self.st_train_status = {}  # 'none' | 'initialized' | 'active' | 'complete'
        self.st_losses = {}
        self.st_datetimes = {}
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
        self.st_inference_videos = []

    def _push_video(self, video_file):
        if video_file[0] == "/":
            src = os.path.join(os.getcwd(), video_file[1:])
            dst = video_file
        else:
            src = os.path.join(os.getcwd(), video_file)
            dst = "/" + video_file
        if not self._drive.isfile(dst) and os.path.exists(src):
            # only put to FileSystem under two conditions:
            # 1. file exists locally; if it doesn't, maybe it has already been deleted for a reason
            # 2. file does not already exist on FileSystem; avoids excessive file transfers
            print(f"TRAIN_INFER UI try put {dst}")
            self._drive.put(src, dst)
            print(f"TRAIN_INFER UI success put {dst}")

    def _train(
        self,
        config_filename=None,
        video_dirname=VIDEOS_DIR,
        labeled_data_dirname=LABELED_DATA_DIR,
        csv_filename=COLLECTED_DATA_FILENAME,
    ):

        if config_filename is None:
            print("ERROR: config_filename must be specified for training models")

        # check to see if we're in demo mode or not
        base_dir = os.path.join(os.getcwd(), self.proj_dir[1:])
        model_dir = os.path.join(self.proj_dir, MODELS_DIR)
        cfg_overrides = {
            "data": {
                "data_dir": base_dir,
                "video_dir": os.path.join(base_dir, video_dirname),
            },
            "eval": {
                "test_videos_directory": os.path.join(base_dir, video_dirname),
                "predict_vids_after_training": True,
            },
            "training": {
                "imgaug": "dlc",
                "max_epochs": self.st_max_epochs,
            }
        }

        # list files needed from FileSystem
        inputs = [
            os.path.join(self.proj_dir, config_filename),
            os.path.join(self.proj_dir, labeled_data_dirname),
            os.path.join(self.proj_dir, video_dirname),
            os.path.join(self.proj_dir, csv_filename),
        ]

        # train models
        for m in ["super", "semisuper"]:
            status = self.st_train_status[m]
            if status == "initialized" or status == "active":
                self.st_train_status[m] = "active"
                outputs = [os.path.join(model_dir, self.st_datetimes[m])]
                cfg_overrides["model"] = {"losses_to_use": self.st_losses[m]}
                self.work.run(
                    action="train", inputs=inputs, outputs=outputs, cfg_overrides=cfg_overrides,
                    results_dir=os.path.join(base_dir, MODELS_DIR, self.st_datetimes[m])
                )
                self.st_train_status[m] = "complete"
                self.work.progress = 0.0  # reset for next model

        self.submit_count_train += 1

    def _run_inference(self, model_dir=None, video_files=None):

        self.work_is_done_inference = False

        if not model_dir:
            model_dir = os.path.join(self.proj_dir, MODELS_DIR, self.st_inference_model)
        if not video_files:
            video_files = self.st_inference_videos

        # launch works:
        # - sequential if local
        # - parallel if on cloud
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
                # move video from ui machine to shared FileSystem
                self._push_video(video_file=video_file)
                # run inference (automatically reformats video for DALI)
                self.works_dict[video_key].run(
                    action="run_inference",
                    model_dir=model_dir,
                    video_file="/" + video_file,
                )
                self.st_infer_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done_inference:
                    # kill work
                    print(f"killing work from video {video_key}")
                    self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_inference = True

    def run(self, action, **kwargs):
        if action == "push_video":
            self._push_video(**kwargs)
        elif action == "train":
            self._train(**kwargs)
        elif action == "run_inference":
            self._run_inference(**kwargs)

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
    if state.n_total_frames == 0:
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
            #### Defaults
            """
        )
        expander = st.expander("Change Defaults")

        # max epochs
        st_max_epochs = expander.text_input(
            "Max training epochs (supervised and semi-supervised)",
            value=300,
        )

        # unsupervised losses (semi-supervised only; only expose relevant losses)
        expander.write("Select losses for semi-supervised model")
        pcamv = state.config_dict["data"]["mirrored_column_matches"]
        if len(pcamv) > 0:
            st_loss_pcamv = expander.checkbox("PCA Multiview", value=True)
        else:
            st_loss_pcamv = False
        pcasv = state.config_dict["data"]["columns_for_singleview_pca"]
        if len(pcasv) > 0:
            st_loss_pcasv = expander.checkbox("PCA Singleview", value=True)
        else:
            st_loss_pcasv = False
        st_loss_temp = expander.checkbox("Temporal", value=True)

        st.markdown(
            """
            #### Select models to train
            """
        )
        st_train_super = st.checkbox("Supervised", value=True)
        st_train_semisuper = st.checkbox("Semi-supervised", value=True)

        st_submit_button_train = st.button("Train models", disabled=state.run_script_train)

        # give user training updates
        if state.run_script_train:
            for m in ["super", "semisuper"]:
                if m in state.st_train_status.keys() and state.st_train_status[m] != "none":
                    status = state.st_train_status[m]
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        p = state.work.progress
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)
                    st.progress(p / 100.0, f"{m} progress ({status}: {int(p)}\% complete)")

        if st_submit_button_train:
            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) and st_train_semisuper:
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
            state.st_train_status = {
                "super": "initialized" if st_train_super else "none", 
                "semisuper": "initialized" if st_train_semisuper else "none",
            }

            # construct semi-supervised loss list
            semi_losses = []
            if st_loss_pcamv:
                semi_losses.append("pca_multiview")
            if st_loss_pcasv:
                semi_losses.append("pca_singleview")
            if st_loss_temp:
                semi_losses.append("temporal")
            state.st_losses = {"super": [], "semisuper": semi_losses}

            # set model times
            st_datetimes = {}
            for i in range(2):
                dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
                if i == 0:  # supervised model
                    st_datetimes["super"] = dtime
                    time.sleep(1)  # allow date/time to update
                if i == 1:  # semi-supervised model
                    st_datetimes["semisuper"] = dtime

            # NOTE: cannot set these dicts entry-by-entry in the above loop, o/w don't get set?
            state.st_datetimes = st_datetimes
            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")

    with infer_tab:

        st.header("Predict on New Videos")

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

        st_submit_button_infer = st.button(
            "Run inference",
            disabled=len(st_videos) == 0 or state.run_script_infer,
        )
        if state.run_script_infer:
            keys = [k for k, _ in state.works_dict.items()]  # cannot directly call keys()?
            for vid, status in state.st_infer_status.items():
                if status == "initialized":
                    p = 0.0
                elif status == "active":
                    vid_ = vid.replace(".", "_")
                    if vid_ in keys:
                        try:
                            p = state.works_dict[vid_].progress
                        except:
                            p = 100.0  # if work is deleted while accessing
                    else:
                        p = 100.0  # state.work.progress
                elif status == "complete":
                    p = 100.0
                else:
                    st.text(status)
                st.progress(p / 100.0, f"{vid} progress ({status}: {int(p)}\% complete)")
            st.warning("waiting for existing inference to finish")

        # Lightning way of returning the parameters
        if st_submit_button_infer:

            state.submit_count_infer += 1

            state.st_inference_model = model_dir
            state.st_inference_videos = st_videos
            state.st_infer_status = {s: 'initialized' for s in st_videos}
            st.text("Request submitted!")
            state.run_script_infer = True  # must the last to prevent race condition

            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_infer_ui_submitted")
