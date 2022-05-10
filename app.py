import os

from lightning.frontend import StreamlitFrontend
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.components.python import TracerPythonScript
from lightning.storage.path import Path
import logging

from lightning.components.python import TracerPythonScript
import streamlit as st
from lightning.utilities.state import AppState

import os
import streamlit as st
from streamlit_ace import st_ace
import fire
import logging
from pathlib import Path
import yaml

script_fo = TracerPythonScript(
  script_path = "run_fo.py",
  script_args = ["--address","0.0.0.0"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  exposed_ports = {"server": 5151},
  raise_exception = True,
  )

script_tb = TracerPythonScript(
  script_path = "run_tb.py",
  script_args = ["--server","0.0.0.0"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  exposed_ports = {"server": 6006},
  raise_exception = True,
  )

script_train = TracerPythonScript(
  script_path = "/Users/robertlee/github/mnist-hydra-grid/mnist-hydra-01.py",
  script_args = None,
  env = None,
  cloud_compute = None,
  blocking = True,
  run_once = True,
  exposed_ports = None,
  raise_exception = True,
  )

class ScriptUI(LightningFlow):
  def __init__(self):
    super().__init__()
    # input to UI
    self.script_dir = "/Users/robertlee/github/mnist-hydra-grid"
    self.script_name = "mnist-hydra-01.py"
    self.config_dir = "/Users/robertlee/github/mnist-hydra-grid"
    self.config_ext = "*.yaml"
    # output from the UI
    self.st_train      = False
    self.st_script_dir = None
    self.st_script_name = None
    self.st_script_arg = None    # must match state.script_arg in _render_streamlit_fn
    self.st_script_env = None    # must exist state.script_env in _render_streamlit_fn
    self.st_config     = None # must exist state.script_env in _render_streamlit_fn

  def run(self, destination_dir: str):
    self.destination_dir = destination_dir

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)

def hydra_config(language="yaml"):
  basename = st.session_state.hydra_config[0]
  filename = st.session_state.hydra_config[1]
  logging.debug(f"selectbox {st.session_state}")
  if basename in st.session_state:
    content_raw = st.session_state[basename]
  else:
    try:
      with open(filename) as input:
        content_raw = input.read()  
    except FileNotFoundError:
      st.error('File not found.')
    except Exception as err:  
      st.error(f"can't process select item. {err}")      
  content_new = st_ace(value=content_raw, language=language)
  if content_raw != content_new:
    st.write("content changed")
    st.session_state[basename] = content_new

def _render_streamlit_fn(state: AppState, dir="./"):
  """Display YAML file and return arguments and env variable fields

  :dir (str): dir name to pull the yaml file from
  :return (dict): {'script_arg':script_arg, 'script_env':script_env}
  """
  st_script_dir = st.text_input('Script Dir', value=state.script_dir or ".")
  st_script_name = st.text_input('Script Name', value=state.script_name or "run.py")

  st_config_dir = st.text_input('Dir of Hydra YAMLs', value=state.config_dir or ".")
  st_config_ext = st.text_input('YAML Extensions', value=state.config_ext or "*.yaml")

  options = []
  if not options:
    print("building options")
    for file in Path(st_config_dir).rglob(st_config_ext):
      basename = os.path.basename(file)
      options.append([basename, str(file)])

  st_script_arg = st.text_input('Script Args', placeholder="training.max_epochs=11")

  st_script_env = st.text_input('Script Env Vars')

  show_basename = lambda opt: opt[0]
  st.selectbox("override hydra config", options, key="hydra_config", format_func=show_basename)  

  st_submit_button = st.button('Train')

  options = hydra_config()
  
  if st_submit_button:
    state.st_script_dir = st_script_dir
    state.st_script_name = st_script_name

    state.st_script_arg = st_script_arg
    state.st_script_env = st_script_env
    state.st_train       = True           # must the last to prevent race condition

class App(LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.script_fo = script_fo
    self.script_tb = script_tb
    self.script_ui = ScriptUI()
    self.script_train = script_train

  def run(self):
    self.script_fo.run()
    self.script_tb.run()
    if self.script_ui.st_train:
      self.script_train.script_path = os.path.join(self.script_ui.st_script_dir, self.script_ui.st_script_name)
      self.script_train.script_args = self.script_ui.st_script_arg  
      self.script_train.env = self.script_ui.st_script_env
      self.script_train.run()

  def configure_layout(self):
    tab1 = { "name": "Lightning Pose Param", "content": self.script_ui }

    if self.script_fo.has_started:
      tab2 = {"name": "Fiftyone", "content": self.script_fo.exposed_url('server')}
    else:
      tab2 = {"name": "Fiftyone", "content": ""}

    if self.script_tb.has_started:
      tab3 = {"name": "Tensorboard", "content": self.script_tb.exposed_url('server')}
    else:
      tab3 = {"name": "Tensorboard", "content": ""}

    return[tab1, tab2, tab3]  

app = LightningApp(App())