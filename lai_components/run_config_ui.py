import os
import logging
import string
import sh
import shlex
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace

from lai_components.hydra_ui import hydra_config, get_hydra_config_name, get_hydra_dir_name 
from lai_components.args_utils import args_to_dict, dict_to_args 

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning_app.components.python import TracerPythonScript
from lightning_app.frontend import StreamlitFrontend
from lightning_app.utilities.state import AppState
from lightning_app.storage.path import Path


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
    self.st_script_dir = None
    self.st_script_env = None  
    self.st_hydra_config_name = None
    self.st_hydra_config_dir = None   
    self.st_eval_test_videos_directory = None

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):
    """Config
    """

    st.markdown("""<img src="https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png" alt="Wide Lightning Pose Logo" width="200"/>

Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, 
supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.

### A Single application with pre-integrated components
* Lightning Pose Configuration
* Train
* Train Diagnostics
* Image/Video Diagnostics Preparation
* Image/Video Diagnostics
* Image/Video Annotation

### Built with the coolest Deep Learning packages
* `pytorch-lightning` for multiple-GPU training and to minimize boilerplate code
* `nvidia-DALI` for accelerated GPU dataloading
* `Hydra` to orchestrate the config files and log experiments
* `kornia` for differntiable computer vision ops
* `torchtyping` for type and shape assertions of `torch` tensors
* `FiftyOne` for visualizing model predictions
* `Tensorboard` to visually diagnoze training performance

### Configuration

""", unsafe_allow_html=True)

    st_eval_test_videos_directory = st.text_input("Eval Test Videos Directory", value=state.eval_test_videos_directory)

    st_script_env = st.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")
    st_script_dir = st.text_input("Script Dir", value=state.script_dir, placeholder=".")

    st_hydra_config = hydra_config(context=st, config_dir=state.config_dir, config_name=state.config_name, root_dir=st_script_dir)

    # save the outputs 
    state.st_eval_test_videos_directory = st_eval_test_videos_directory

    state.st_script_dir  = st_script_dir
    state.st_script_env = st_script_env

    state.st_hydra_config_name = get_hydra_config_name()
    state.st_hydra_config_dir = get_hydra_dir_name()

