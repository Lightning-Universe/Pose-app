from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
import lightning.app as L
from lightning.app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import os
import shutil
import streamlit as st
import time
from typing import Optional, Union, List
import yaml

from lai_components.args_utils import args_to_dict, dict_to_args
from lai_components.build_utils import lightning_pose_dir, tracking_diag_dir
from lai_components.build_utils import lightning_pose_venv, tensorboard_venv
from lai_components.build_utils import (
    TensorboardBuildConfig,
    FiftyOneBuildConfig,
    StreamlitBuildConfig,
)
from lai_components.landing_ui import LandingUI
from lai_components.project_ui import ProjectUI, ProjectDataIO
from lai_components.extract_frames_ui import ExtractFramesUI
from lai_components.label_studio.component import LitLabelStudio
from lai_components.train_ui import TrainDemoUI
from lai_components.fo_ui import FoRunUI
from lai_components.video_ui import VideoUI
from lai_components.lpa_utils import output_with_video_prediction
from lai_components.vsc_streamlit import StreamlitFrontend
from lai_work.bashwork import LitBashWork


# TODO
# - revisit project config page and trigger the following
#   - get username/password


class LitPoseApp(L.LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps
        drive_name = "lit://lpa"
        self.drive = Drive(drive_name)

        # -----------------------------
        # UIs
        # -----------------------------
        # landing tab
        self.landing_ui = LandingUI()

        # project manager tab
        config_dir = os.path.join(lightning_pose_dir, "scripts")
        default_config_file = os.path.join(config_dir, "config_default.yaml")
        data_dir = os.path.join(drive_name, "data")
        self.project_ui = ProjectUI(
            config_dir=config_dir,
            data_dir=data_dir,
            default_config_file=default_config_file,
        )
        self.project_io = ProjectDataIO(
            drive_name=drive_name,
            config_dir=config_dir,
            data_dir=data_dir,
            default_config_file=default_config_file,
        )

        # extract frames tab
        self.extract_ui = ExtractFramesUI(
            drive_name=drive_name,
            script_dir=lightning_pose_dir,
            script_name="scripts/extract_frames.py",
            script_args="",
            proj_dir=None,  # to be set upon project load/creation
        )

        # training tab; this will default to the toy dataset, unless a project is created/loaded
        self.train_ui = TrainDemoUI(
            script_dir=lightning_pose_dir,
            script_name="scripts/train_hydra.py",
            script_args="",
            script_env="HYDRA_FULL_ERROR=1",
            config_dir=None,  # to be set upon project load/creation
            config_name=None,  # to be set upon project load/creation
            test_videos_dir="toy_datasets/toymouseRunningData/unlabeled_videos",
            outputs_dir=None,  # to be set upon project load/creation
            max_epochs=200,
        )

        # fiftyone tab (images only for now)
        self.fo_ui = FoRunUI(
            script_dir=lightning_pose_dir,
            script_name="scripts/create_fiftyone_dataset.py",
            script_env="HYDRA_FULL_ERROR=1",
            script_args="""
                eval.fiftyone.dataset_name=test1 
                eval.fiftyone.model_display_names=["test1"]
                eval.fiftyone.dataset_to_create="images"
                eval.fiftyone.build_speed="fast" 
                eval.fiftyone.launch_app_from_script=False 
            """
        )

        # video tab
        self.video_ui = VideoUI(video_file=None)

        # -----------------------------
        # workers
        # -----------------------------
        # tensorboard
        self.my_tb = LitBashWork(
            cloud_compute=L.CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
            drive_name=drive_name,
        )

        # training/fiftyone
        self.my_work = LitBashWork(
            cloud_compute=L.CloudCompute("gpu"),
            cloud_build_config=FiftyOneBuildConfig(),
            drive_name=drive_name,
        )
        self.work_is_done_extract_frames = False

        # label studio
        self.label_studio = LitLabelStudio(
            cloud_compute=L.CloudCompute("default"),
            drive_name=drive_name,
        )

        # streamlit labeled
        self.my_streamlit_frame = LitBashWork(
            cloud_compute=L.CloudCompute("gpu"),
            cloud_build_config=StreamlitBuildConfig(),
            drive_name=drive_name,
        )

        # streamlit video
        self.my_streamlit_video = LitBashWork(
            cloud_compute=L.CloudCompute("gpu"),
            cloud_build_config=StreamlitBuildConfig(),
            drive_name=drive_name,
        )

    def init_lp_outputs_to_ui(self, search_dir=None):

        # get existing model directories that contain test*.csv
        if not search_dir:
            search_dir = self.train_ui.outputs_dir

        cmd = f"find {search_dir} -maxdepth 4 -type f -name predictions.csv"
        self.my_work.run(cmd, cwd=lightning_pose_dir, save_stdout=True)
        if self.my_work.last_args() == cmd:
            outputs = output_with_video_prediction(self.my_work.last_stdout())
            self.train_ui.set_hydra_outputs(outputs)
            self.fo_ui.set_hydra_outputs(outputs)
            self.my_work.reset_last_args()

    def init_fiftyone_outputs_to_ui(self):
        # get existing fiftyone datasets
        cmd = "fiftyone datasets list"
        self.my_work.run(cmd, venv_name=lightning_pose_venv)
        if self.my_work.last_args() == cmd:
            options = []
            for x in self.my_work.stdout:
                if x.endswith("No datasets found"):
                    continue
                if x.startswith("Migrating database"):
                    continue
                if x.endswith("python"):
                    continue
                options.append(x)
            self.fo_ui.set_fo_dataset(options)
        else:
            pass

    def start_extract_frames(self):

        # set videos to select frames from
        vid_file_args = ""
        for vid_file in self.extract_ui.st_video_files:
            vid_file_args += f" --video_files={vid_file}"

        data_dir = os.path.join(self.extract_ui.proj_dir, "labeled-data")
        cmd = "python" \
              + " " + self.extract_ui.script_name \
              + vid_file_args \
              + f" --data_dir={data_dir}" \
              + f" --n_frames_per_video={self.extract_ui.st_n_frames_per_video}" \
              + f" --context_frames=2" \
              + f" --export_idxs_as_csv"
        self.my_work.run(
            cmd,
            venv_name=lightning_pose_venv,
            cwd=self.train_ui.st_script_dir,
            # outputs=[self.extract_ui.data_dir],
        )

        self.work_is_done_extract_frames = True

    def start_tensorboard(self):
        """run tensorboard"""
        cmd = "tensorboard --logdir outputs --host $host --port $port"
        self.my_tb.run(
            cmd,
            venv_name=tensorboard_venv,
            wait_for_exit=False,
            cwd=lightning_pose_dir,
        )

    def start_fiftyone(self):
        """run fiftyone"""
        # TODO:
        #   right after fiftyone, the previous find command is triggered should not be the case.
        cmd = "fiftyone app launch --address $host --port $port"
        self.my_work.run(
            cmd,
            venv_name=lightning_pose_venv,
            wait_for_exit=False,
            cwd=lightning_pose_dir,
        )

    def start_lp_train_video_predict(self):

        # check to see if we're in demo mode or not
        if self.train_ui.config_name is not None:
            config_cmd = " --config-path=%s --config-name=%s" % (
                self.train_ui.config_dir, self.train_ui.config_name)
        else:
            config_cmd = ""

        # train supervised model
        if self.train_ui.st_train_super:
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["super"]
            self.my_work.run(
                cmd,
                venv_name=lightning_pose_venv,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                outputs=[self.train_ui.outputs_dir],
            )

        # train semi-supervised model
        if self.train_ui.st_train_semisuper:
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["semisuper"]
            self.my_work.run(
                cmd,
                venv_name=lightning_pose_venv,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                outputs=[self.train_ui.outputs_dir],
            )

        # have TB pull the new data
        # input_output_only=True means that we'll pull inputs from drive, but not run commands
        cmd = "null command"  # make this unique
        self.my_tb.run(
            cmd,
            venv_name=tensorboard_venv,
            cwd=lightning_pose_dir,
            input_output_only=True,
            inputs=[self.train_ui.outputs_dir],
        )

        # set the new outputs for UIs
        cmd = f"find {self.train_ui.outputs_dir} -maxdepth 4 -type f -name predictions.csv"
        self.my_work.run(cmd, cwd=lightning_pose_dir, save_stdout=True)
        if self.my_work.last_args() == cmd:
            outputs = output_with_video_prediction(self.my_work.last_stdout())
            self.train_ui.set_hydra_outputs(outputs)
            self.fo_ui.set_hydra_outputs(outputs)
            self.my_work.reset_last_args()

    def start_fiftyone_dataset_creation(self):

        cmd = "python" \
              + " " + self.fo_ui.st_script_name \
              + " " + self.fo_ui.st_script_args \
              + " " + self.fo_ui.script_args_append \
              + " " + "eval.fiftyone.dataset_to_create=images"
        self.my_work.run(
            cmd,
            venv_name=lightning_pose_venv,
            env=self.fo_ui.st_script_env,
            cwd=self.fo_ui.st_script_dir,
          )

        # add dataset name to list for user to see
        self.fo_ui.add_fo_dataset(self.fo_ui.st_dataset_name)

    def start_labeled_video_creation(self):

        # set prediction files (hard code some paths for now)
        # select random video
        prediction_file_args = ""
        video_file = "test_vid.csv"  # TODO: find random vid in predictions directory
        for model_dir in self.fo_ui.st_model_dirs:
            abs_file = os.path.abspath(os.path.join(
                lightning_pose_dir, self.fo_ui.outputs_dir, model_dir, "video_preds", video_file))
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.fo_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # get absolute path of video file
        video_file_abs = os.path.abspath(os.path.join(
            lightning_pose_dir, self.train_ui.test_videos_dir, video_file.replace(".csv", ".mp4")))

        # set absolute path of labeled video file
        save_file_abs = video_file_abs.replace(".mp4", f"_{self.fo_ui.st_dataset_name}.mp4")

        # set reasonable defaults for video creation
        cmd = "python ./tracking-diagnostics/scripts/create_labeled_video.py" \
              + f" " + prediction_file_args \
              + f" " + model_name_args \
              + f" --video_file={video_file_abs}" \
              + f" --save_file={save_file_abs}" \
              + f" --likelihood_thresh=0.9" \
              + f" --max_frames=100" \
              + f" --markersize=6" \
              + f" --framerate=20" \
              + f" --height=4"
        self.my_work.run(
            cmd,
            venv_name=lightning_pose_venv,
            outputs=[save_file_abs],
            cwd="."
        )

        self.video_ui.video_file = save_file_abs

    def start_st_frame(self):
        """run streamlit for labeled frames"""

        # set labeled csv
        # TODO: extract this directly from hydra
        csv_file = os.path.join(
            lightning_pose_dir, "toy_datasets/toymouseRunningData/CollectedData_.csv")
        labeled_csv_args = f"--labels_csv={csv_file}"

        # set prediction files (hard code some paths for now)
        prediction_file_args = ""
        for model_dir in self.fo_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.fo_ui.outputs_dir, model_dir, "predictions.csv")
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.fo_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.fo_ui.outputs_dir, self.fo_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/labeled_frame_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + labeled_csv_args \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_frame.run(
            cmd,
            venv_name=lightning_pose_venv,
            wait_for_exit=False,
            cwd="."
        )

    def start_st_video(self):
        """run streamlit for videos"""

        # set prediction files (hard code some paths for now)
        # select random video
        prediction_file_args = ""
        video_file = "test_vid.csv"  # TODO: find random vid in predictions directory
        for model_dir in self.fo_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.fo_ui.outputs_dir, model_dir, "video_preds", video_file)
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.fo_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.fo_ui.outputs_dir, self.fo_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/video_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_video.run(
            cmd,
            venv_name=lightning_pose_venv,
            wait_for_exit=False,
            cwd="."
        )

    def run(self):

        # -----------------------------
        # init UIs (find prev artifacts)
        # -----------------------------
        # find previously trained models, expose to training UI
        # self.init_lp_outputs_to_ui()

        # find previously constructed fiftyone datasets, expose to fiftyone UI
        # self.init_fiftyone_outputs_to_ui()

        # -----------------------------
        # init background services once
        # -----------------------------
        self.label_studio.start_label_studio()
        self.start_tensorboard()
        # self.start_fiftyone()

        # -----------------------------
        # run work
        # -----------------------------
        # update project configuration
        if self.project_ui.run_script:  # better name?

            print("===============run proj ui script====================")

            # update user-supplied parameters in config yaml file
            self.project_io.update_paths(project_name=self.project_ui.st_project_name)
            self.project_io.update_project_config(
                project_name=self.project_ui.st_project_name,
                new_vals_dict=self.project_ui.st_new_vals,
            )

            # update paths data dir for frame extraction object
            self.extract_ui.proj_dir = self.project_io.proj_dir

            # update label studio object, create label studio xml file, create/load project
            self.label_studio.proj_dir = self.project_io.proj_dir
            self.label_studio.project_name = self.project_ui.st_project_name
            if self.project_ui.create_new_project and self.project_ui.count == 0:
                self.label_studio.create_labeling_config_xml(self.project_ui.keypoints)
                self.label_studio.create_new_project()

            # point training UI to config file
            self.train_ui.config_dir = self.project_io.proj_dir
            self.train_ui.outputs_dir = self.project_io.model_dir
            self.train_ui.config_name = "config_%s" % self.project_ui.st_project_name
            self.train_ui.test_videos_dir = os.path.join(self.project_io.proj_dir, "videos")

            self.project_ui.run_script = False
            self.project_ui.count += 1

        # extract frames for labeleding from uploaded videos
        if self.extract_ui.run_script:
            self.start_extract_frames()
            self.extract_ui.run_script = False

        # wait until litbashwork is done extracting frames, then update tasks
        if self.work_is_done_extract_frames:
            # force a run of update_tasks by using the timer; control how often it gets run by
            # immediately setting the boolean flag that controls access to False
            self.work_is_done_extract_frames = False
            self.project_io.update_frame_shapes()
            self.label_studio.update_tasks(timer=time.time())

        # check labeling task and export new labels (with timer wrapper)
        if self.project_ui.count > 0:
            self.label_studio.run()
            # find fraction of labeled frames to return to UI
            n_labeled_frames, n_total_frames = self.project_io.compute_labeled_frame_fraction()
            self.train_ui.n_labeled_frames = n_labeled_frames
            self.train_ui.n_total_frames = n_total_frames

        # train on ui button press
        if self.train_ui.run_script:
            self.train_ui.run_script = False
            self.start_lp_train_video_predict()

        # initialize diagnostics on button press
        if self.fo_ui.run_script:
            self.fo_ui.run_script = False
            self.start_fiftyone_dataset_creation()
            self.start_labeled_video_creation()
            # self.start_st_frame()
            # self.start_st_video()

    def configure_layout(self):

        # init tabs
        landing_tab = {"name": "Lightning Pose", "content": self.landing_ui}
        project_tab = {"name": "Manage Project", "content": self.project_ui}
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        annotate_tab = {"name": "Annotate Frames", "content": self.label_studio.label_studio}

        # training tabs
        train_demo_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Status", "content": self.my_tb}

        # diagnostics tabs
        fo_prep_tab = {"name": "Prepare Diagnostics", "content": self.fo_ui}
        fo_tab = {"name": "Labeled Preds", "content": self.my_work}
        st_frame_tab = {"name": "Labeled Diagnostics", "content": self.my_streamlit_frame}
        video_tab = {"name": "Video Preds", "content": self.video_ui}
        st_video_tab = {"name": "Video Diagnostics", "content": self.my_streamlit_video}

        if self.landing_ui.st_mode == "demo":
            return [
                landing_tab,
                train_demo_tab,
                train_diag_tab,
                fo_prep_tab,
                fo_tab,
                # st_frame_tab,
                video_tab,
                # st_video_tab,
            ]

        elif self.landing_ui.st_mode == "project":
            if not self.project_ui.st_project_name:
                # need to create/load new project before moving on to other tabs
                return [landing_tab, project_tab]
            else:
                # show all tabs
                return [
                    landing_tab,
                    project_tab,
                    extract_tab,
                    annotate_tab,
                    train_demo_tab,
                    train_diag_tab,
                    # fo_prep_tab,
                    # fo_tab,
                    # video_tab,
                ]

        else:
            return [landing_tab]


app = L.LightningApp(LitPoseApp())
