"""UI for training models."""

from datetime import datetime
from lightning import CloudCompute, LightningFlow, LightningWork
from lightning.app.utilities.state import AppState
from lightning.app.storage import FileSystem
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import lightning.pytorch as pl
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import subprocess
import time
import yaml

from lightning_pose_app.build_configs import LitPoseBuildConfig
from lightning_pose_app.utilities import StreamlitFrontend

st.set_page_config(layout="wide")


class TrainingProgress(Callback):

    def __init__(self, work):
        self.work = work
        self.progress_delta = 0.5

    @rank_zero_only
    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        progress = 100 * (trainer.current_epoch + 1) / float(trainer.max_epochs)
        if self.work.progress is None:
            if progress > self.progress_delta:
                self.work.progress = round(progress, 4)
        elif round(progress, 4) - self.work.progress >= self.progress_delta:
            if progress > 100:
                self.work.progress = 100.0
            else:
                self.work.progress = round(progress, 4)


class LitPose(LightningWork):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pwd = os.getcwd()
        self.progress = 0.0

        self._drive = FileSystem()

        self.work_is_done_training = True
        self.work_is_done_inference = True
        self.count = 0

    def get_from_drive(self, inputs):
        for i in inputs:
            print(f"TRAIN drive get {i}")
            try:  # file may not be ready
                src = i  # shared
                dst = self.abspath(i)  # local
                self._drive.get(src, dst, overwrite=True)
                print(f"drive data saved at {dst}")
            except Exception as e:
                print(e)
                print(f"did not load {i} from drive")
                pass

    def put_to_drive(self, outputs):
        for o in outputs:
            print(f"TRAIN drive try put {o}")
            src = self.abspath(o)  # local
            dst = o  # shared
            # make sure dir ends with / so that put works correctly
            if os.path.isdir(src):
                src = os.path.join(src, "")
                dst = os.path.join(dst, "")
            # check to make sure file exists locally
            if not os.path.exists(src):
                continue
            self._drive.put(src, dst)
            print(f"TRAIN drive success put {dst}")

    @staticmethod
    def abspath(path):
        if path[0] == "/":
            path_ = path[1:]
        else:
            path_ = path
        return os.path.abspath(path_)

    def _reformat_videos(self, video_files=None, **kwargs):

        # pull videos from Drive; these will come from "root." component
        self.get_from_drive(video_files)

        video_files_new = []
        for video_file in video_files:

            video_file_abs = self.abspath(video_file)

            # check 1: does file exist?
            video_file_exists = os.path.exists(video_file_abs)
            if not video_file_exists:
                continue

            # check 2: is file in the correct format for DALI?
            video_file_correct_codec = check_codec_format(video_file_abs)
            ext = os.path.splitext(os.path.basename(video_file))[1]
            video_file_new = video_file.replace(f"_tmp{ext}", ".mp4")
            video_file_abs_new = os.path.join(os.getcwd(), video_file_new)
            if not video_file_correct_codec:
                print("re-encoding video to be compatable with Lightning Pose video reader")
                reencode_video(video_file_abs, video_file_abs_new)
                # remove local version of old video
                # cannot remove Drive version of old video, created by other Work
                os.remove(video_file_abs)
                # record
                video_files_new.append(video_file_new)
            else:
                # rename
                os.rename(video_file_abs, video_file_abs_new)
                # record
                video_files_new.append(video_file_new)

            # push possibly reformated, renamed videos to Drive
            self.put_to_drive(video_files_new)

        return video_files_new

    def _train(self, inputs, outputs, cfg_overrides, results_dir):

        from omegaconf import DictConfig
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
        # Pull data from drive
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

        if (
                ("temporal" in cfg.model.losses_to_use)
                and model.do_context
                and not data_module.unlabeled_dataloader.context_sequences_successive
        ):
            raise ValueError(
                f"Temporal loss is not compatible with non-successive context sequences. "
                f"Please change cfg.dali.context.train.consecutive_sequences=True."
            )

        # ----------------------------------------------------------------------------------
        # Set up and run training
        # ----------------------------------------------------------------------------------

        # logger
        logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

        # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
        callbacks = get_callbacks(cfg)
        # add callback to log progress
        callbacks.append(TrainingProgress(self))

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
                    labeled_mp4_file = os.path.join(labeled_vid_dir,
                                                    video_pred_name + "_labeled.mp4")
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
        # Push results to drive, clean up
        # ----------------------------------------------------------------------------------
        os.chdir(self.pwd)
        self.put_to_drive(outputs)  # IMPORTANT! must come after changing directories
        self.work_is_done_training = True

    def _run_inference(self, model, video):
        import time
        self.work_is_done_inference = False
        print(f"launching inference for video {video} using model {model}")
        time.sleep(5)
        self.work_is_done_inference = True

    def run(self, action=None, **kwargs):
        if action == "train":
            self._train(**kwargs)
        elif action == "run_inference":
            self._run_inference(**kwargs)


class TrainUI(LightningFlow):
    """UI to enter training parameters for demo data."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._drive = FileSystem()

        self.work = LitPose(
            cloud_compute=CloudCompute("gpu"),
            cloud_build_config=LitPoseBuildConfig(),
        )

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script_train = False
        self.run_script_infer = False

        # for controlling messages to user
        self.submit_count_train = 0

        # for controlling when models are broadcast to other flows/workers
        self.count = 0

        # save parameters for later run
        self.proj_dir = None

        self.n_labeled_frames = None  # set externally
        self.n_total_frames = None  # set externally

        # updated externally by top-level flow
        self.trained_models = []
        self.progress = 0

        # output from the UI (train; all will be dicts with keys=models, except st_max_epochs)
        self.st_max_epochs = None
        self.st_train_status = {}  # 'none' | 'initialized' | 'active' | 'complete'
        self.st_losses = {}
        self.st_datetimes = {}

        # output from the UI (infer)
        self.st_inference_model = None
        self.st_inference_videos = None

    def run(self, action, **kwargs):
        if action == "push_video":
            video_file = kwargs["video_file"]
            if video_file[0] == "/":
                src = os.path.join(os.getcwd(), video_file[1:])
                dst = video_file
            else:
                src = os.path.join(os.getcwd(), video_file)
                dst = "/" + video_file
            self._drive.put(src, dst)

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
            or state.run_script_train:
        st_autorefresh(interval=2000, key="refresh_train_ui")

    # add a sidebar to show the labeling progress
    # Calculate percentage of frames labeled
    labeling_progress = state.n_labeled_frames / state.n_total_frames
    st.sidebar.markdown('### Labeling Progress')
    st.sidebar.progress(labeling_progress)
    st.sidebar.write(f"You have labeled {state.n_labeled_frames} out of {state.n_total_frames} frames.")

    st.sidebar.markdown("""### Existing models""")
    st.sidebar.selectbox("Browse", sorted(state.trained_models, reverse=True))
    st.sidebar.write("Proceed to next tabs to analyze your previously trained models.")

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
            value=100,
        )

        # unsupervised losses (semi-supervised only)
        expander.write("Select losses for semi-supervised model")
        st_loss_pcamv = expander.checkbox("PCA Multiview", value=True)
        st_loss_pcasv = expander.checkbox("PCA Singleview", value=True)
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
        video_dir = os.path.join(state.proj_dir[1:], "videos")
        os.makedirs(video_dir, exist_ok=True)

        # initialize the file uploader
        uploaded_files = st.file_uploader("Choose video files", accept_multiple_files=True)

        # for each of the uploaded files
        st_videos = []
        for uploaded_file in uploaded_files:
            # read it
            bytes_data = uploaded_file.read()
            # name it
            filename = uploaded_file.name.replace(" ", "_")
            filepath = os.path.join(video_dir, filename)
            st_videos.append(filepath)
            # write the content of the file to the path
            with open(filepath, "wb") as f:
                f.write(bytes_data)

        st_submit_button_infer = st.button(
            "Run inference",
            disabled=len(st_videos) == 0 or state.run_script_infer,
        )
        if state.run_script_infer:
            st.warning("waiting for existing inference to finish")

        # Lightning way of returning the parameters
        if st_submit_button_infer:
            state.st_inference_model = model_dir
            state.st_inference_videos = st_videos
            st.text("Request submitted!")
            state.run_script_infer = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_infer_ui_submitted")


def reencode_video(input_file: str, output_file: str) -> None:
    """reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new video

    """
    # check input file exists
    assert os.path.isfile(input_file), "input video does not exist."
    # check directory for saving outputs exists
    assert os.path.isdir(
        os.path.dirname(output_file)), \
        f"saving folder {os.path.dirname(output_file)} does not exist."
    ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
    subprocess.run(ffmpeg_cmd, shell=True)


def check_codec_format(input_file: str):
    """Run FFprobe command to get video codec and pixel format."""

    ffmpeg_cmd = f'ffmpeg -i {input_file}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
    output_str = output_str.stderr

    # search for correct codec (h264) and pixel format (yuv420p)
    if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
        # print('Video uses H.264 codec')
        is_codec = True
    else:
        # print('Video does not use H.264 codec')
        is_codec = False
    return is_codec
