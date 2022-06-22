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


class ScriptRunUI(LightningFlow):
  """UI to enter training parameters
  Input and output variables with streamlit must be pre decleared
  """

  def __init__(self, 
      *args, 
      script_dir, 
      script_name, 
      config_dir = "./", 
      config_name = "config.yaml", 
      script_args, 
      script_env, 
      eval_test_videos_directory,
      outputs_dir = "outputs",
      **kwargs):
    super().__init__(*args, **kwargs)
    # control runners
    # True = Run Jobs.  False = Do not Run jobs
    # UI sets to True to kickoff jobs
    # Job Runner sets to False when done
    self.run_script = False       
    # input to UI
    self.eval_test_videos_directory = eval_test_videos_directory

    self.script_dir = script_dir
    self.script_name = script_name
    self.script_env = script_env

    self.config_dir = config_dir
    self.config_name = config_name   

    self.script_args = script_args
    self.outputs_dir = outputs_dir
    # output from the UI

    self.st_eval_test_videos_directory = None

    self.st_script_dir  = None
    self.st_script_name = None

    self.st_script_args = None
    self.st_script_env  = None
    self.st_run_script  = True  
    self.run_script = False
    self.st_hydra_config_name = None
    self.st_hydra_config_dir = None       

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)

def set_script_args(script_args:str):
  script_args_dict = args_to_dict(script_args)

  # only set if not alreay present
  if not('hydra.run.dir' in script_args_dict):
    run_date_time=datetime.today().strftime('%Y-%m-%d/%H-%M-%S')
    script_args_dict['hydra.run.dir'] = f"outputs/{run_date_time}"
 
  # change back to array
  return(dict_to_args(script_args_dict))
  
def get_existing_outpts(state):
  options=[]
  try:
    options = ["/".join(x.strip().split("/")[-3:-1]) for x in sh.find(f"{state.script_dir}/{state.outputs_dir}","-type","d", "-name", "tb_logs",)]
    options.sort(reverse=True)
  except:
    pass  
  return(options)

def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset
    """
    st_output_dir = st.selectbox("existing output", get_existing_outpts(state))

    # edit the script_args
    st_script_args = st.text_area("Script Args", value=state.script_args, placeholder='--a 1 --b 2')

    st_submit_button = st.button("Submit",disabled=True if (state.run_script == True) else False )
    if state.run_script == True:
      st.warning(f"waiting for existing training to finish")      

    # these are not used as often
    expander = st.expander("Change Defaults")

    st_eval_test_videos_directory = expander.text_input("Eval Test Videos Directory", value=state.eval_test_videos_directory)

    st_script_env = expander.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")

    st_script_dir = expander.text_input("Script Dir", value=state.script_dir, placeholder=".")
    st_script_name = expander.text_input("Script Name", value=state.script_name, placeholder="run.py")

    st_hydra_config = hydra_config(context=expander, config_dir=state.config_dir, config_name=state.config_name, root_dir=st_script_dir)
    
    # Lightning way of returning the parameters
    if st_submit_button:
      # add default options
      st_script_args = set_script_args(st_script_args) 
      # save them
      state.st_eval_test_videos_directory = st_eval_test_videos_directory

      state.st_script_dir  = st_script_dir
      state.st_script_name = st_script_name

      state.st_script_args = st_script_args
      state.st_script_env  = st_script_env

      state.st_hydra_config_name = get_hydra_config_name()
      state.st_hydra_config_dir = get_hydra_dir_name()

      state.run_script  = True  # must the last to prevent race condition