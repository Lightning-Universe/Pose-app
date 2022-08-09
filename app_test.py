# app.py
import os
import sys
import shlex
from string import Template
from typing import Optional, Union, List

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
import lightning_app as L
from lightning_app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import streamlit as st

from lai_work.bashwork import LitBashWork

from lai_components.build_utils import lightning_pose_dir, label_studio_dir, tracking_diag_dir
from lai_components.build_utils import lightning_pose_venv, label_studio_venv, tensorboard_venv
from lai_components.build_utils import (
    TensorboardBuildConfig,
    LabelStudioBuildConfig,
    FiftyOneBuildConfig,
    StreamlitBuildConfig,
)
from lai_components.args_utils import args_to_dict, splitall
from lai_components.lpa_utils import output_with_video_prediction
from lai_components.vsc_streamlit import StreamlitFrontend

import logging
import time


# hydra.run.dir
#   outputs/YY-MM-DD/HH-MM-SS
# eval.hydra_paths
# eval_hydra_paths
#   YY-MM-DD/HH-MM-SS
predict_args = """
eval.hydra_paths=["${eval_hydra_paths}"] \
eval.test_videos_directory=${root_dir}/${eval_test_videos_directory} \
eval.saved_vid_preds_dir="${root_dir}/${hydra.run.dir}/
"""


class ConfigUI(LightningFlow):
  """UI to enter training parameters
  Input and output variables with streamlit must be pre decleared
  """

  def __init__(self,
      *args,
      script_dir,
      script_env,
      config_dir = "./",
      config_name = "config.yaml",
      eval_test_videos_directory,
      **kwargs):
    super().__init__(*args, **kwargs)
    # input to UI
    self.eval_test_videos_directory = eval_test_videos_directory

    self.script_dir = script_dir
    self.script_env = script_env

    self.config_dir = config_dir
    self.config_name = config_name

    # output from the UI
    self.st_mode = None
    self.st_action = None
    self.st_proceed_str = None

    self.st_script_dir = None
    self.st_script_env = None
    self.st_hydra_config_name = None
    self.st_hydra_config_dir = None
    self.st_eval_test_videos_directory = None

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)


class TestUI(LightningFlow):

  def __init__(self,
      *args,
      text="default",
      **kwargs):
    super().__init__(*args, **kwargs)
    # input to UI
    self.text = text

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn2)


def _render_streamlit_fn2(state: AppState):
    st.markdown(state.text)


def _render_streamlit_fn(state: AppState):
    """Config
    """

    st.markdown("""<img src="https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png" alt="Wide Lightning Pose Logo" width="200"/>

Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, 
supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.

#### A single application with pre-integrated components
* Train Models
* Training Diagnostics
* View Predictions on Images
* Diagnostics on Labeled Images
* Diagnostics on Unlabeled Videos

""", unsafe_allow_html=True)

    st.markdown(
        """
        #### Run demo
        Train a baseline supervised model and a semi-supervised model on an example dataset.
        """
    )
    button_demo = st.button("Run demo")
    if button_demo:
        state.st_mode = "demo"
        state.st_action = "review and train several baseline models on an example dataset"
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "demo":
        st.markdown(
            "<p style='font-family:sans-serif; color:Red;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        #### Start new project
        Start a new project: upload videos, label frames, and train models.
        """
    )
    button_new = st.button("New project")
    if button_new:
        state.st_mode = "new"
        state.st_action = "start a new project"
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "new":
        st.markdown(
            "<p style='font-family:sans-serif; color:Red;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        #### Load existing project
        Train new models, label new frames, process new videos.
        """
    )
    button_load = st.button("Load project")
    if button_load:
        state.st_mode = "resume"
        state.st_action = "resume a previously initialized project"
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "resume":
        st.markdown(
            "<p style='font-family:sans-serif; color:Red;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )


class LitPoseApp(L.LightningFlow):

    def __init__(self):
        super().__init__()
        # shared data for apps
        self.drive_lpa = Drive("lit://lpa")
        #
        self.args_append = None
        # UIs
        self.config_ui = ConfigUI(
            script_dir=lightning_pose_dir,
            script_env="HYDRA_FULL_ERROR=1",
            config_dir="./scripts",
            eval_test_videos_directory="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos",
        )
        self.test_ui_a = TestUI(
            text="Test A"
        )
        self.test_ui_b = TestUI(
            text="Test B"
        )

    def run(self):

        # init once
        # self.init_lp_outputs_to_ui()
        # self.init_fiftyone_outputs_to_ui()

        # background services once
        # self.start_tensorboard()
        # self.start_st_labeled()
        # self.start_label_studio()
        # self.start_fiftyone()

        # train on ui button press
        # if self.config_ui.st_mode == "demo":
        # elif self.config_ui.st_mode == "new project":
        # else:
        pass

    def configure_layout(self):

        config_tab = {"name": "Lightning Pose", "content": self.config_ui}
        test_tab_a = {"name": "Test A", "content": self.test_ui_a}
        test_tab_b = {"name": "Test B", "content": self.test_ui_b}

        if self.config_ui.st_mode == "demo":
            return [config_tab, test_tab_a]

        elif self.config_ui.st_mode == "new":
            return [config_tab, test_tab_b]

        else:
            return [config_tab]


app = L.LightningApp(LitPoseApp())
