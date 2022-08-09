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

        #
        self.args_append = None

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
        # trainig/fiftyone
        self.my_work = LitBashWork(
            cloud_compute=L.CloudCompute("gpu"),
            cloud_build_config=FiftyOneBuildConfig(),
        )

    def init_lp_outputs_to_ui(self, search_dir=None):

        # get existing model directories that contain test*.csv
        if not search_dir:
            search_dir = self.train_ui.outputs_dir

        cmd = f"find {search_dir} -maxdepth 3 -type f -name predictions.csv"
        self.my_work.run(cmd, cwd=lightning_pose_dir)
        if self.my_work.last_args() == cmd:
            outputs = output_with_video_prediction(self.my_work.last_stdout())
            self.train_ui.set_hydra_outputs(outputs)
            # self.fo_ui.set_hydra_outputs(outputs)
            self.my_work.reset_last_args()

    def start_tensorboard(self):
        """run tensorboard"""
        cmd = "tensorboard --logdir outputs --host {host} --port {port}"
        self.my_tb.run(
            cmd,
            venv_name=tensorboard_venv,
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

        # set the new outputs for UIs
        self.init_lp_outputs_to_ui()

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

        # indicate to UI
        self.train_ui.run_script = False

    def run(self):

        # -----------------------------
        # init once
        # -----------------------------
        # find previously trained models, expose to UI
        self.init_lp_outputs_to_ui()

        # TODO
        # self.init_fiftyone_outputs_to_ui()

        # -----------------------------
        # background services once
        # -----------------------------
        self.start_tensorboard()
        # self.start_st_labeled()
        # self.start_label_studio()
        # self.start_fiftyone()

        # train on ui button press
        if self.train_ui.run_script:
            self.start_lp_train_video_predict()

        # elif self.config_ui.st_mode == "new project":
        # else:

    def configure_layout(self):

        landing_tab = {"name": "Lightning Pose", "content": self.landing_ui}
        train_demo_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Diag", "content": self.my_tb}

        test_tab_a = {"name": "Test A", "content": self.test_ui_a}
        test_tab_b = {"name": "Test B", "content": self.test_ui_b}

        if self.landing_ui.st_mode == "demo":
            return [landing_tab, train_demo_tab, train_diag_tab]

        elif self.landing_ui.st_mode == "new":
            return [landing_tab, test_tab_a]

        elif self.landing_ui.st_mode == "resume":
            return [landing_tab, test_tab_b]

        else:
            return [landing_tab]


app = L.LightningApp(LitPoseApp())
