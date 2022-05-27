import os
import logging
import string
import sh
import shlex
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.components.python import TracerPythonScript
from lightning.frontend import StreamlitFrontend
from lightning.utilities.state import AppState
from lightning.storage.path import Path


class ConfigUI(LightningFlow):
  """UI to enter training parameters
  Input and output variables with streamlit must be pre decleared
  """

  def __init__(self, *args, script_dir, config_dir, config_ext, script_env, **kwargs):
    super().__init__(*args, **kwargs)   
    # input to UI
    self.script_dir = script_dir
    self.script_env = script_env

    self.config_dir = config_dir
    self.config_ext = config_ext        

    # output from the UI
    self.st_config_dir = None
    self.st_config_ext = None
    self.st_script_dir = None
    self.st_script_env = None  

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)

def hydra_config(language="yaml"):
    try:
      basename = st.session_state.hydra_config[0]
      filename = st.session_state.hydra_config[1]
    except:
      st.error("no files found")
      return
    logging.debug(f"selectbox {st.session_state}")
    if basename in st.session_state:
        content_raw = st.session_state[basename]
    else:
        try:
            with open(filename) as input:
                content_raw = input.read()
        except FileNotFoundError:
            st.error("File not found.")
        except Exception as err:
            st.error(f"can't process select item. {err}")
#    content_new = st.text_area("hydra", value=content_raw)
    content_new = st_ace(value=content_raw, language=language)
    if content_raw != content_new:
        st.write("content changed")
        st.session_state[basename] = content_new


def _render_streamlit_fn(state: AppState):
    """Config
    """

    st.markdown("""<img src="https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png" alt="Wide Lightning Pose Logo" width="200"/>

Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, 
supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.

### A Single application with pre-integrated components:
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

    st_script_env = st.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")
    st_script_dir = st.text_input("Script Dir", value=state.script_dir, placeholder=".")

    st_config_dir = st.text_input("Config Dir", value=state.config_dir, placeholder=".")
    st_config_ext = st.text_input("Config File Extensions", value=state.config_ext, placeholder="*.yaml")

    # TODO: is refresh needed everytime?
    options = []
    print("building options")
    for file in Path(st_config_dir).rglob(st_config_ext):
        basename = os.path.basename(file)
        options.append([basename, str(file)])
    show_basename = lambda opt: opt[0]
    st.selectbox(
        "override hydra config", options, key="hydra_config", format_func=show_basename
    )

    options = hydra_config()
    
    state.st_script_dir  = st_script_dir
    state.st_script_env  = st_script_env



