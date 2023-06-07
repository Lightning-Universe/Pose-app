"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app demo_app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
import os
import time
import yaml

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.ui.fifty_one import FiftyoneConfigUI
from lightning_pose_app.ui.project import ProjectDataIO
from lightning_pose_app.ui.streamlit import StreamlitAppLightningPose
from lightning_pose_app.ui.train_infer import TrainUI
from lightning_pose_app.build_configs import TensorboardBuildConfig, lightning_pose_dir


# TODO
# - ProjectDataIO._put_to_drive_remove_local does NOT overwrite directories already on Drive - bad!
# - launch training in parallel (get this working with `extract_frames` standalone app first)


class LitPoseApp(LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps; NOTE: this is hard-coded in the run_inference method below too
        drive_name = "lit://lpa"

        self.proj_name = "demo"

        # -----------------------------
        # paths
        # -----------------------------
        config_dir = os.path.join(lightning_pose_dir, "scripts", "configs")
        self.data_dir = "data"  # relative to base of filesystem

        # load default config and pass to project manager
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # project manager (work)
        self.project_io = ProjectDataIO(
            drive_name=drive_name,
            data_dir=self.data_dir,
            default_config_dict=default_config_dict,
        )

        # training tab (flow + work)
        self.train_ui = TrainUI(drive_name=drive_name)
        self.train_ui.n_labeled_frames = 90  # hard-code these values for now
        self.train_ui.n_total_frames = 90

        # fiftyone tab (flow + work)
        self.fiftyone_ui = FiftyoneConfigUI(drive_name=drive_name)

        # streamlit tabs (flow + work)
        self.streamlit_frame = StreamlitAppLightningPose(drive_name=drive_name, app_type="frame")
        self.streamlit_video = StreamlitAppLightningPose(drive_name=drive_name, app_type="video")

        # tensorboard tab (work)
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
            drive_name=drive_name,
            component_name="tensorboard",
        )

        # -----------------------------
        # copy toy data to project
        # -----------------------------
        # here we copy the toy dataset config file, frames, and labels that come packaged with the 
        # lightning-pose repo and move it to a new directory that is consistent with the project 
        # structure the app expects
        # later we will write that newly copied data to the Drive so other Works have access

        # copy config file
        toy_config_file_src = os.path.join(
            lightning_pose_dir, "scripts/configs/config_toy-dataset.yaml")
        toy_config_file_dst = os.path.join(self.data_dir, self.proj_name, "model_config_demo.yaml")
        self.project_io._copy_file(toy_config_file_src, toy_config_file_dst)

        # frames, videos, and labels
        toy_data_src = os.path.join(lightning_pose_dir, "toy_datasets/toymouseRunningData")
        toy_data_dst = os.path.join(self.data_dir, self.proj_name)
        self.project_io._copy_dir(toy_data_src, toy_data_dst)

        self.demo_data_transferred = False

    # @property
    # def ready(self) -> bool:
    #     """Return true once all works have an assigned url"""
    #     return all([
    #         self.project_io.url != "",
    #         self.fiftyone_ui.work.url != "",
    #         self.streamlit_frame.work.url != "",
    #         self.streamlit_video.work.url != "",
    #         self.train_ui.work.url != "",
    #     ])

    def start_tensorboard(self, logdir):
        """run tensorboard"""
        cmd = f"tensorboard --logdir {logdir} --host $host --port $port --reload_interval 30"
        self.tensorboard.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def train_models(self):

        # check to see if we're in demo mode or not
        base_dir = os.path.join(os.getcwd(), self.project_io.proj_dir)
        cfg_overrides = {
            "data": {
                "data_dir": base_dir,
                "video_dir": os.path.join(base_dir, 'unlabeled_videos'),
            },
            "eval": {
                "test_videos_directory": os.path.join(base_dir, 'unlabeled_videos'),
                "predict_vids_after_training": True,
            },
            "training": {
                "imgaug": "dlc",
                "max_epochs": self.train_ui.st_max_epochs,
            }
        }

        # list files needed from Drive
        inputs = [
            os.path.join(self.project_io.proj_dir, self.project_io.config_name),
            os.path.join(self.project_io.proj_dir, "barObstacleScaling1"),
            os.path.join(self.project_io.proj_dir, "unlabeled_videos"),
            os.path.join(self.project_io.proj_dir, "CollectedData_.csv"),
        ]

        # train models
        for m in ["super", "semisuper"]:
            status = self.train_ui.st_train_status[m]
            if status == "initialized" or status == "active":
                self.train_ui.st_train_status[m] = "active"
                outputs = [os.path.join(self.project_io.model_dir, self.train_ui.st_datetimes[m], "")]
                cfg_overrides["model"] = {"losses_to_use": self.train_ui.st_losses[m]}
                self.train_ui.progress = self.train_ui.work.progress
                self.train_ui.work.run(
                    action="train", inputs=inputs, outputs=outputs, cfg_overrides=cfg_overrides,
                    results_dir=os.path.join(base_dir, "models", self.train_ui.st_datetimes[m])
                )
                self.train_ui.st_train_status[m] = "complete"
                self.train_ui.work.progress = 0.0
                self.train_ui.progress = 0.0

        self.train_ui.count += 1

    def update_trained_models_list(self, timer):
        self.project_io.run(action="update_trained_models_list", timer=timer)
        if self.project_io.trained_models:
            self.train_ui.trained_models = self.project_io.trained_models
            self.fiftyone_ui.trained_models = self.project_io.trained_models

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # don't interfere w/ train; since all Works use the same filesystem when running locally,
        # one Work updating the filesystem which is also used by the trainer can corrupt data, etc.
        run_while_training = True
        if not is_running_in_cloud() and self.train_ui.run_script_train:
            run_while_training = False

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # update paths if we know which project we're working with
        self.project_io.run(action="update_paths", project_name=self.proj_name)
        self.train_ui.proj_dir = self.project_io.proj_dir
        self.streamlit_frame.proj_dir = self.project_io.proj_dir
        self.streamlit_video.proj_dir = self.project_io.proj_dir
        self.fiftyone_ui.proj_dir = self.project_io.proj_dir
        self.fiftyone_ui.config_name = self.project_io.config_name

        # write demo data to the Drive so other Works have access (run once)
        if not self.demo_data_transferred:
            # we call the run method twice with two sets of arguments so the run cache will always
            # be overwritten; therefore if we put these two calls outside of the boolean flag they
            # will be continuously called as the app is running
            # update config file
            self.project_io.run(
                action="update_project_config",
                new_vals_dict={"data": {  # TODO: will this work on cloud?
                    "data_dir": os.path.join(os.getcwd(), self.project_io.proj_dir)}
                },
            )
            # put demo data onto Drive
            self.project_io.run(
                action="put_file_to_drive", 
                file_or_dir=self.project_io.proj_dir, 
                remove_local=False,
            )
            print("Demo data transferred to Drive")
            self.demo_data_transferred = True

        # find previously trained models for project, expose to training and diagnostics UIs
        self.update_trained_models_list(timer=self.train_ui.count)  # timer is to force later runs

        # start background services (only run once)
        self.start_tensorboard(logdir=self.project_io.model_dir)
        self.streamlit_frame.run(action="initialize")
        self.streamlit_video.run(action="initialize")
        self.fiftyone_ui.run(action="start_fiftyone")

        # find previously constructed fiftyone datasets
        self.fiftyone_ui.run(action="find_fiftyone_datasets")

        # -------------------------------------------------------------
        # train models on ui button press
        # -------------------------------------------------------------
        if self.train_ui.run_script_train:
            self.train_models()
            inputs = [self.project_io.model_dir]
            # have tensorboard pull the new data
            self.tensorboard.run(
                "null command",
                cwd=os.getcwd(),
                input_output_only=True,  # pull inputs from Drive, but do not run commands
                inputs=inputs,
            )
            # have streamlit pull the new data
            self.streamlit_frame.run(action="pull_models", inputs=inputs)
            self.streamlit_video.run(action="pull_models", inputs=inputs)
            self.project_io.update_models = True
            self.train_ui.run_script_train = False

        # set the new outputs for UIs
        if self.project_io.update_models:
            self.project_io.update_models = False
            self.update_trained_models_list(timer=self.train_ui.count)

        # -------------------------------------------------------------
        # run inference on ui button press (single model, multiple vids)
        # -------------------------------------------------------------
        if self.train_ui.run_script_infer and run_while_training:
            print("Cannot run inference in demo right now")
            # self.run_inference(
            #     model=self.train_ui.st_inference_model,
            #     videos=self.train_ui.st_inference_videos,
            # )
            self.train_ui.run_script_infer = False

        # -------------------------------------------------------------
        # build fiftyone dataset on button press from FiftyoneUI
        # -------------------------------------------------------------
        if self.fiftyone_ui.run_script:
            self.fiftyone_ui.run(action="build_fiftyone_dataset")
            self.fiftyone_ui.run_script = False

    def configure_layout(self):

        # training tabs
        train_tab = {"name": "Train Infer", "content": self.train_ui}
        train_status_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        st_frame_tab = {"name": "Labeled Diagnostics", "content": self.streamlit_frame.work}
        st_video_tab = {"name": "Video Diagnostics", "content": self.streamlit_video.work}
        fo_prep_tab = {"name": "Prepare Fiftyone", "content": self.fiftyone_ui}
        fo_tab = {"name": "Fiftyone", "content": self.fiftyone_ui.work}

        return [
            train_tab,
            train_status_tab,
            st_frame_tab,
            st_video_tab,
            fo_prep_tab,
            fo_tab,
        ]


app = LightningApp(LitPoseApp())
