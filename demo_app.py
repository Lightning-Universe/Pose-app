"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app demo_app.py

"""

from lightning.app import CloudCompute, LightningApp, LightningFlow
import logging
import os
import shutil
import sys
import yaml

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.ui.project import ProjectUI
from lightning_pose_app.ui.streamlit import StreamlitAppLightningPose
from lightning_pose_app.ui.streamlit_video_viewer import StreamlitVideoViewer
from lightning_pose_app.ui.train_infer import TrainUI
from lightning_pose_app import LIGHTNING_POSE_DIR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('APP')


class LitPoseApp(LightningFlow):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # paths
        # -----------------------------
        self.data_dir = "/data"  # relative to Pose-app root
        self.proj_name = "demo"

        # load default config and pass to project manager
        config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # project manager (flow)
        self.project_ui = ProjectUI(
            data_dir=self.data_dir,
            default_config_dict=default_config_dict,
        )

        # training tab (flow + work)
        self.train_ui = TrainUI(allow_context=False, max_epochs_default=150)
        self.train_ui.n_labeled_frames = 90  # hard-code these values for now
        self.train_ui.n_total_frames = 90

        # fiftyone tab (work)
        self.fiftyone = LitBashWork(
            cloud_compute=CloudCompute("default"),
        )

        # streamlit tabs (flow + work)
        self.streamlit_frame = StreamlitAppLightningPose(app_type="frame")
        self.streamlit_video = StreamlitAppLightningPose(app_type="video")
        self.streamlit_video_player = StreamlitVideoViewer()

        # tensorboard tab (work)
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
        )

        # -----------------------------
        # copy toy data to project
        # -----------------------------
        # here we copy the toy dataset config file, frames, and labels that come packaged with the
        # lightning-pose repo and move it to a new directory that is consistent with the project
        # structure the app expects
        # later we will write that newly copied data to the FileSystem so other Works have access

        # copy config file
        toy_config_file_src = os.path.join(
            LIGHTNING_POSE_DIR, "scripts/configs/config_mirror-mouse-example.yaml")
        toy_config_file_dst = os.path.join(
            os.getcwd(), self.data_dir[1:], self.proj_name, "model_config_demo.yaml")
        self._copy_file(toy_config_file_src, toy_config_file_dst)

        # frames, videos, and labels
        toy_data_src = os.path.join(LIGHTNING_POSE_DIR, "data/mirror-mouse-example")
        toy_data_dst = os.path.join(os.getcwd(), self.data_dir[1:], self.proj_name)
        self._copy_dir(toy_data_src, toy_data_dst)

        self.demo_data_transferred = False

    @staticmethod
    def _copy_file(src_path, dst_path):
        """Copy a file from the source path to the destination path."""
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.isfile(dst_path):
                shutil.copy(src_path, dst_path)
                _logger.debug(f"File copied from {src_path} to {dst_path}")
            else:
                _logger.debug(f"Did not copy {src_path} to {dst_path}; {dst_path} already exists")
        except IOError as e:
            _logger.warning(f"Unable to copy file. {e}")

    def _copy_dir(self, src_path, dst_path):
        """Copy a directory from the source path to the destination path."""
        try:
            os.makedirs(dst_path, exist_ok=True)
            src_files_or_dirs = os.listdir(src_path)
            for src_file_or_dir in src_files_or_dirs:
                if os.path.isfile(os.path.join(src_path, src_file_or_dir)):
                    self._copy_file(
                        os.path.join(src_path, src_file_or_dir),
                        os.path.join(dst_path, src_file_or_dir),
                    )
                else:
                    src_dir = os.path.join(src_path, src_file_or_dir)
                    dst_dir = os.path.join(dst_path, src_file_or_dir)
                    if not os.path.isdir(dst_dir):
                        shutil.copytree(src_dir, dst_dir)
                        _logger.debug(f"Directory copied from {src_dir} to {dst_dir}")
                    else:
                        _logger.debug(
                            f"Did not copy {src_dir} to {dst_dir}; {dst_dir} already exists")
        except IOError as e:
            _logger.warning(f"Unable to copy directory. {e}")

    def start_tensorboard(self, logdir):
        """run tensorboard"""
        cmd = f"tensorboard --logdir {logdir} --host $host --port $port --reload_interval 30"
        self.tensorboard.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def start_fiftyone(self):
        """run fiftyone"""
        cmd = "fiftyone app launch --address $host --port $port --remote --wait -1"
        self.fiftyone.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def update_trained_models_list(self, timer):
        self.project_ui.run(action="update_trained_models_list", timer=timer)
        if self.project_ui.trained_models:
            self.train_ui.trained_models = self.project_ui.trained_models

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")
            return

        # don't interfere w/ train; since all Works use the same filesystem when running locally,
        # one Work updating the filesystem which is also used by the trainer can corrupt data, etc.
        run_while_training = True
        if self.train_ui.run_script_train:
            run_while_training = False

        # don't interfere w/ inference
        run_while_inferring = True
        if self.train_ui.run_script_infer:
            run_while_inferring = False

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # update paths if we know which project we're working with
        self.project_ui.run(action="update_paths", project_name=self.proj_name)
        self.train_ui.proj_dir = self.project_ui.proj_dir
        self.streamlit_frame.proj_dir = self.project_ui.proj_dir
        self.streamlit_video.proj_dir = self.project_ui.proj_dir
        self.streamlit_video_player.proj_dir = self.project_ui.proj_dir
        # write demo data to the FileSystem so other Works have access (run once)
        if not self.demo_data_transferred:
            # we call the run method twice with two sets of arguments so the run cache will always
            # be overwritten; therefore if we put these two calls outside of the boolean flag they
            # will be continuously called as the app is running
            # update config file
            self.project_ui.run(
                action="update_project_config",
                new_vals_dict={"data": {
                    "data_dir": os.path.join(os.getcwd(), self.project_ui.proj_dir)[1:]}
                },
            )
            # send params to train ui
            self.train_ui.config_dict = self.project_ui.config_dict
            self.demo_data_transferred = True

        # find previously trained models for project, expose to training and diagnostics UIs
        # timer is to force later runs
        self.update_trained_models_list(timer=self.train_ui.submit_count_train)

        # start background services (only run once)
        self.start_tensorboard(logdir=self.project_ui.model_dir[1:])
        self.start_fiftyone()
        self.streamlit_frame.run(action="initialize")
        self.streamlit_video.run(action="initialize")

        # -------------------------------------------------------------
        # train models on ui button press
        # -------------------------------------------------------------
        if self.train_ui.run_script_train and run_while_inferring:
            self.train_ui.run(action="train", config_filename=self.project_ui.config_name)
            self.project_ui.update_models = True
            self.train_ui.run_script_train = False

        # set the new outputs for UIs
        if self.project_ui.update_models:
            self.project_ui.update_models = False
            self.update_trained_models_list(timer=self.train_ui.submit_count_train)

        # -------------------------------------------------------------
        # run inference on ui button press (single model, multiple vids)
        # -------------------------------------------------------------
        if self.train_ui.run_script_infer and run_while_training:
            self.train_ui.run(
                action="run_inference",
                video_files=self.train_ui.st_inference_videos,  # add arg for run caching purposes
            )
            self.train_ui.run_script_infer = False

    def configure_layout(self):

        # training tabs
        train_tab = {"name": "Train/Infer", "content": self.train_ui}
        train_status_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        st_frame_tab = {"name": "Labeled Diagnostics", "content": self.streamlit_frame.work}
        st_video_tab = {"name": "Video Diagnostics", "content": self.streamlit_video.work}
        st_video_player_tab = {"name": "Video Player", "content": self.streamlit_video_player}
        fo_tab = {"name": "Fiftyone", "content": self.fiftyone}

        return [
            train_tab,
            train_status_tab,
            st_frame_tab,
            st_video_tab,
            st_video_player_tab,
            fo_tab,
        ]


app = LightningApp(LitPoseApp())
