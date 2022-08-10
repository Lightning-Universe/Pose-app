# app.py

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
import lightning_app as L
from lightning_app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import os
import streamlit as st
from typing import Optional, Union, List

from lai_components.args_utils import args_to_dict, dict_to_args
from lai_components.build_utils import lightning_pose_dir, label_studio_dir, tracking_diag_dir
from lai_components.build_utils import lightning_pose_venv, label_studio_venv, tensorboard_venv
from lai_components.build_utils import (
    TensorboardBuildConfig,
    LabelStudioBuildConfig,
    FiftyOneBuildConfig,
    StreamlitBuildConfig,
)
from lai_components.landing_ui import LandingUI
from lai_components.train_ui import TrainDemoUI
from lai_components.fo_ui import FoRunUI
from lai_components.lpa_utils import output_with_video_prediction
from lai_components.vsc_streamlit import StreamlitFrontend
from lai_work.bashwork import LitBashWork


class TestUI(LightningFlow):

    def __init__(self, *args, text="default", **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):
    st.markdown(state.text)


class LitPoseApp(L.LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps
        self.drive_lpa = Drive("lit://lpa")

        # -----------------------------
        # UIs
        # -----------------------------
        # landing tab
        self.landing_ui = LandingUI()

        # training tab
        self.train_ui = TrainDemoUI(
            script_dir=lightning_pose_dir,
            script_name="scripts/train_hydra.py",
            script_args="",
            script_env="HYDRA_FULL_ERROR=1",
            test_videos_dir="toy_datasets/toymouseRunningData/unlabeled_videos"
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
                eval.fiftyone.launch_app_from_script=True 
            """
        )

        # dummy tabs
        self.test_ui_a = TestUI(text="Test A")
        self.test_ui_b = TestUI(text="Test B")

        # -----------------------------
        # workers
        # -----------------------------
        # tensorboard
        self.my_tb = LitBashWork(
            cloud_compute=L.CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
        )
        # training/fiftyone
        self.my_work = LitBashWork(
            cloud_compute=L.CloudCompute("gpu"),
            cloud_build_config=FiftyOneBuildConfig(),
        )

    def init_lp_outputs_to_ui(self, search_dir=None):

        # get existing model directories that contain test*.csv
        if not search_dir:
            search_dir = self.train_ui.outputs_dir

        cmd = f"find {search_dir} -maxdepth 4 -type f -name predictions.csv"
        self.my_work.run(cmd, cwd=lightning_pose_dir)
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

    def start_tensorboard(self):
        """run tensorboard"""
        cmd = "tensorboard --logdir outputs --host {host} --port {port}"
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
        cmd = "fiftyone app launch --address {host} --port {port}"
        self.my_work.run(
            cmd,
            venv_name=lightning_pose_venv,
            wait_for_exit=False,
            cwd=lightning_pose_dir,
        )

    def start_lp_train_video_predict(self):

        # train model
        # multirun issues; something with gradients not being computed correctly, does not happen
        # with non-multiruns
        # cmd = "python" \
        #       + " " + self.train_ui.st_script_name \
        #       + " --multirun" \
        #       + " " + self.train_ui.st_script_args
        # self.my_work.run(
        #     cmd,
        #     venv_name=lightning_pose_venv,
        #     env=self.train_ui.st_script_env,
        #     cwd=self.train_ui.st_script_dir,
        #     outputs=[os.path.join(self.train_ui.st_script_dir, self.train_ui.outputs_dir)],
        # )

        # train supervised model
        if self.train_ui.st_train_super:
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + " " + self.train_ui.st_script_args["super"]
            self.my_work.run(
                cmd,
                venv_name=lightning_pose_venv,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                outputs=[os.path.join(self.train_ui.st_script_dir, self.train_ui.outputs_dir)],
            )

        # train semi-supervised model
        if self.train_ui.st_train_semisuper:
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + " " + self.train_ui.st_script_args["semisuper"]
            self.my_work.run(
                cmd,
                venv_name=lightning_pose_venv,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                outputs=[os.path.join(self.train_ui.st_script_dir, self.train_ui.outputs_dir)],
            )

        # have TB pull the new data
        # input_output_only=True means that we'll pull inputs from drive, but not run commands
        cmd = "null command"  # make this unique
        self.my_tb.run(
            cmd,
            venv_name=tensorboard_venv,
            cwd=lightning_pose_dir,
            input_output_only=True,
            inputs=[os.path.join(self.train_ui.script_dir, self.train_ui.outputs_dir)],
        )

        # set the new outputs for UIs
        cmd = f"find {self.train_ui.outputs_dir} -maxdepth 4 -type f -name predictions.csv"
        self.my_work.run(cmd, cwd=lightning_pose_dir)
        if self.my_work.last_args() == cmd:
            outputs = output_with_video_prediction(self.my_work.last_stdout())
            self.train_ui.set_hydra_outputs(outputs)
            self.fo_ui.set_hydra_outputs(outputs)
            self.my_work.reset_last_args()

        # indicate to UI
        self.train_ui.run_script = False

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

        # add name
        self.fo_ui.add_fo_dataset(self.fo_ui.st_dataset_name)

        # indicate to UI
        self.fo_ui.run_script = False

    def run(self):

        # -----------------------------
        # init UIs (find prev artifacts)
        # -----------------------------
        # find previously trained models, expose to training UI
        self.init_lp_outputs_to_ui()

        # find previously constructed fiftyone datasets, expose to fiftyone UI
        self.init_fiftyone_outputs_to_ui()

        # -----------------------------
        # init background services once
        # -----------------------------
        self.start_tensorboard()
        self.start_fiftyone()
        # self.start_st_labeled()
        # self.start_label_studio()

        # -----------------------------
        # run work
        # -----------------------------
        # train on ui button press
        if self.train_ui.run_script:
            self.start_lp_train_video_predict()

        # create fiftyone dataset on button press
        if self.fo_ui.run_script:
            self.start_fiftyone_dataset_creation()

        # elif self.config_ui.st_mode == "new project":
        # else:

    def configure_layout(self):

        landing_tab = {"name": "Lightning Pose", "content": self.landing_ui}
        train_demo_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Status", "content": self.my_tb}
        fo_prep_tab = {"name": "Prepare Diagnostics", "content": self.fo_ui}
        fo_tab = {"name": "View Preds", "content": self.my_work}
        # st_frame_tab = {"name": "Frame Diag", "content": self.my_streamlit}
        # st_video_tab = {"name": "Video Diag", "content": self.my_streamlit}

        test_tab_a = {"name": "Test A", "content": self.test_ui_a}
        test_tab_b = {"name": "Test B", "content": self.test_ui_b}

        if self.landing_ui.st_mode == "demo":
            return [landing_tab, train_demo_tab, train_diag_tab, fo_prep_tab, fo_tab]

        elif self.landing_ui.st_mode == "new":
            return [landing_tab, test_tab_a]

        elif self.landing_ui.st_mode == "resume":
            return [landing_tab, test_tab_b]

        else:
            return [landing_tab]


app = L.LightningApp(LitPoseApp())
