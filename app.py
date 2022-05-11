import os
from typing import Any, Dict, Optional, Union
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
import shlex

script_fo = TracerPythonScript(
  script_path = "run_fo.py",
  script_args = ["--address","0.0.0.0"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  port = 5151,
  raise_exception = True,
  )

class ScriptTB(TracerPythonScript):
  def __init__(self,*args, **kwargs):
    super().__init__(*args, **kwargs)  
  def run(self,script_path: str="run_tb.py", script_args: str = "--logdir=./", script_env: str  = None):
    pass

script_tb = TracerPythonScript(
  script_path = "run_tb.py",
  script_args = ["--server=0.0.0.0","--logdir=./"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  port = 6006,
  raise_exception = True,
  )

class ScriptTrain(TracerPythonScript):
  """Run a training script given arguments
    Args:
      script_path: ex: run.py
      script_args: ex: --name me --dir ./
      script_env: ex: user=xxx password=123}
  """
  def __init__(self,*args, **kwargs):
    super().__init__(*args, **kwargs)
  def run(self,script_path: str, script_args: str = None, script_env: str  = None):
    self.script_path = script_path
    self.script_args = shlex.split(script_args)
    self.env = {} 
    for x in shlex.split(script_env):
      k,v = x.split("=",2)
      self.env[k]=v
    print(f"{self.script_path} {self.script_args} {self.env}")
    super().run()

class ScriptUI(LightningFlow):
  """UI to enter training parameters
    Input and output variables with streamlit must be pre decleared

  """  
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
    # input to UI
    self.script_dir = "./"
    self.script_name = "app.py"
    self.config_dir = "./"
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

  st_config_dir = st.text_input('Config Dir', value=state.config_dir or ".")
  st_config_ext = st.text_input('Config File Extensions', value=state.config_ext or "*.yaml")

  st_script_arg = st.text_input('Script Args', placeholder="training.max_epochs=11")
  st_script_env = st.text_input('Script Env Vars', placeholder="ABC=123 DEF=345")

  options = []
  if not options:
    print("building options")
    for file in Path(st_config_dir).rglob(st_config_ext):
      basename = os.path.basename(file)
      options.append([basename, str(file)])
  show_basename = lambda opt: opt[0]
  st.selectbox("override hydra config", options, key="hydra_config", format_func=show_basename)  

  st_submit_button = st.button('Train')

  options = hydra_config()
  
  # Lightning way of returning the parameters
  if st_submit_button:
    state.st_script_dir = st_script_dir
    state.st_script_name = st_script_name

    state.st_script_arg = st_script_arg
    state.st_script_env = st_script_env
    state.st_train       = True    # must the last to prevent race condition

class App(LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.script_fo = script_fo
    self.script_tb = script_tb
    self.script_ui = ScriptUI()
    self.script_train = ScriptTrain(
      script_path="./app.py",
      blocking=True)

  def run(self):
    self.script_fo.run()
    self.script_tb.run()
    if self.script_ui.st_train:
      self.script_train.run(
        script_path=os.path.join(self.script_ui.st_script_dir, self.script_ui.st_script_name),
        script_args = self.script_ui.st_script_arg,
        script_env = self.script_ui.st_script_env)

  def configure_layout(self):
    tab1 = { "name": "Lightning Pose Param", "content": self.script_ui }

    if self.script_fo.has_started:
      tab2 = {"name": "Fiftyone", "content": f"http://127.0.0.1:{self.script_fo.port}" }
    else:
      tab2 = {"name": "Fiftyone", "content": ""}

    if self.script_tb.has_started:
      tab3 = {"name": "Tensorboard", "content": f"http://127.0.0.1:{self.script_tb.port}" }
    else:
      tab3 = {"name": "Tensorboard", "content": ""}

    return[tab1, tab2, tab3]  

app = LightningApp(App())