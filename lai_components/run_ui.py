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


class ScriptRunUI(LightningFlow):
  """UI to enter training parameters
  Input and output variables with streamlit must be pre decleared
  """

  def __init__(self, *args, script_dir, script_name, config_dir, config_ext, script_args, script_env, outputs_dir = "outputs", **kwargs):
    super().__init__(*args, **kwargs)
    # control runners
    # True = Run Jobs.  False = Do not Run jobs
    # UI sets to True to kickoff jobs
    # Job Runner sets to False when done
    self.run_script = False       
    # input to UI
    self.script_dir = script_dir
    self.script_name = script_name
    self.script_env = script_env

    self.config_dir = config_dir
    self.config_ext = config_ext        

    self.script_args = script_args
    self.outputs_dir = outputs_dir
    # output from the UI
    self.st_submit = False
    self.st_script_dir = None
    self.st_script_name = None
    self.st_script_args = None  
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

def set_script_args(script_args:str):
  script_args_dict = {}
  script_args_array = []
  for x in shlex.split(script_args, posix=False):
    k,v = x.split("=",1)
    script_args_dict[k] = v

  # these will be controlled by the runners.  remove if set manually
  script_args_dict['+hydra.run.out'] = datetime.today().strftime('outputs/%Y-%m-%d/%M-%H-%S')
 
  # change back to array
  for k,v in script_args_dict.items():
    script_args_array.append(f"{k}={v}")
  # return as a text
  return(" \n".join(script_args_array)) 
  
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

    # edit the scritp_args
    state.script_args = set_script_args(state.script_args) 
    st_script_args = st.text_area("Script Args", value=state.script_args, placeholder='--a 1 --b 2')

    st_submit_button = st.button("Submit",disabled=True if (state.run_script == True) else False )
   if state.run_script == True:
      st.warning(f"waiting for existing training to finish")      

    # these are not used as often
    expander = st.expander("Change Defaults")

    st_script_env = expander.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")

    st_script_dir = expander.text_input("Script Dir", value=state.script_dir, placeholder=".")
    st_script_name = expander.text_input("Script Name", value=state.script_name, placeholder="run.py")

    st_config_dir = expander.text_input("Config Dir", value=state.config_dir, placeholder=".")
    st_config_ext = expander.text_input("Config File Extensions", value=state.config_ext, placeholder="*.yaml")

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
    
    # Lightning way of returning the parameters
    if st_submit_button:
        state.st_script_dir  = st_script_dir
        state.st_script_name = st_script_name

        state.st_script_args = st_script_args
        state.st_script_env  = st_script_env
        state.run_script     = True  # must the last to prevent race condition

        print(f"x{state.st_script_dir}")
        print(f"x{state.st_script_name}")
        print(f"x{state.st_script_args}")
        print(f"x{state.st_script_env}")
        print(f"x{state.run_script}")


