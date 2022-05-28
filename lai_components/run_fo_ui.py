import os
import logging
import string
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace
import sh
import shlex
import fiftyone as fo

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.components.python import TracerPythonScript
from lightning.frontend import StreamlitFrontend
from lightning.utilities.state import AppState
from lightning.storage.path import Path


class FoRunUI(LightningFlow):
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
    self.script_dir  = script_dir
    self.script_name = script_name
    self.script_env = script_env

    self.config_dir = config_dir
    self.config_ext = config_ext        

    self.script_args = script_args
    self.outputs_dir = outputs_dir
    # submit count
    self.submit_count = 0

    # output from the UI
    self.st_output_dir = None
    self.st_model_display_names = None
    self.st_submit = False
    self.st_script_dir = None
    self.st_script_name = None
    self.st_script_args = None  
    self.st_script_env = None  
    self.st_dataset_name = None

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
    content_new = st_ace(value=content_raw, language=language)
    if content_raw != content_new:
        st.write("content changed")
        st.session_state[basename] = content_new

def set_script_args(st_output_dir:[str], script_args:str):
  script_args_dict = {}
  script_args_array = []
  for x in shlex.split(script_args, posix=False):
    k,v = x.split("=",1)
    script_args_dict[k] = v
  # enrich the args  
  if st_output_dir:  
    path_list = ','.join([f"'{x}'" for x in st_output_dir])
    script_args_dict["eval.hydra_paths"]=f"[{path_list}]"

  if script_args_dict['eval.video_file_to_plot']: 
    script_args_dict['eval.video_file_to_plot'] = os.path.abspath(script_args_dict['eval.video_file_to_plot'])

  if script_args_dict['eval.pred_csv_files_to_plot']:
    print(script_args_dict['eval.pred_csv_files_to_plot'])
    x = eval(script_args_dict['eval.pred_csv_files_to_plot'])
    z = ",".join([f"'{os.path.abspath(y)}'" for y in x])
    script_args_dict['eval.pred_csv_files_to_plot'] = f"[{z}]"


  # these will be controlled by the runners.  remove if set manually
  script_args_dict.pop('eval.fiftyone.address', None)
  script_args_dict.pop('eval.fiftyone.port', None)
  script_args_dict.pop('eval.fiftyone.launch_app_from_script', None)
  script_args_dict.pop('eval.fiftyone.dataset_to_create', None) 
  script_args_dict.pop('eval.fiftyone.dataset_name', None)
  script_args_dict.pop('eval.fiftyone.model_display_names', None)
  # only set if not alreay present
  if not('+hydra.run.out' in script_args_dict):
    script_args_dict['+hydra.run.out'] = datetime.today().strftime('outputs/%Y-%m-%d/%M-%H-%S')
    
  for k,v in script_args_dict.items():
    script_args_array.append(f"{k}={v}")
  return(" \n".join(script_args_array)) 
  
def get_existing_outputs(state):
  options=[]
  try:
    options = ["/".join(x.strip().split("/")[-3:-1]) for x in sh.find(f"{state.script_dir}/{state.outputs_dir}","-type","d", "-name", "tb_logs",)]
    options.sort(reverse=True)
  except:
    pass  
  return(options)

def get_existing_datasets():
  options = fo.list_datasets()
  try:
    options.remove('')
  except:
    pass  
  return(options)

def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset
    """

    # outputs to choose from
    st_output_dir = st.multiselect("select output", get_existing_outputs(state))

    # for each output, choose a name 
    model_display_names_str = st.text_input("display name for each output separated by space")
    st_model_display_names = shlex.split(model_display_names_str)
    if len(st_model_display_names) != len(st_output_dir):
      st.error("unique names must be given to each output")
    print(f"{st_output_dir} {st_output_dir}")

    # dataset names
    existing_datasets = get_existing_datasets()
    st.write(f"Existing Fifityone datasets {', '.join(existing_datasets)}")
    st_dataset_name = st.text_input("name other than the above existing names")
    if st_dataset_name in existing_datasets:
      st.error(f"{st_dataset_name} exists.  Please choose a new name.")
      st_dataset_name = None

    # parse
    state.script_args = set_script_args(st_output_dir, state.script_args) 
    st_script_args = st.text_area("Script Args", value=state.script_args, placeholder='--a 1 --b 2')
    if st_script_args != state.script_args:
      state.script_args = st_script_args 

    st_submit_button = st.button("Submit", disabled=True if ((len(st_output_dir)==0) or (st_dataset_name is None) or (st_dataset_name == "") or (state.run_script == True)) else False)
    if state.run_script == True:
      st.warning(f"waiting for existing dataset creation to finish")     
    if len(st_output_dir) == 0:
      st.warning(f"select at least one output to continue")
    if (st_dataset_name is None) or (st_dataset_name == ""):
      st.warning(f"enter a unique dataset name to continue")

    # these are not used as often
    expander = st.expander("Change Defaults")

    st_script_env = expander.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")

    st_script_dir = expander.text_input("Script Dir", value=state.script_dir, placeholder=".")
    st_script_name = expander.text_input("Script Name", value=state.script_name, placeholder="run.py")

    st_config_dir = expander.text_input("Config Dir", value=state.config_dir, placeholder=".")
    st_config_ext = expander.text_input("Config File Extensions", value=state.config_ext, placeholder="*.yaml")

    options = []
    if not options:
        print("building options")
        for file in Path(st_config_dir).rglob(st_config_ext):
            basename = os.path.basename(file)
            options.append([basename, str(file)])
    show_basename = lambda opt: opt[0]
    st.selectbox(
        "override hydra config", options, key="hydra_config", format_func=show_basename
    )

    options = hydra_config()

    if state.script_args != st_script_args:
      print(f"value changed {st_script_args}")
      # state.script_args = st_script_args   TODO: this causes infinite loop to kick in.

    # Lightning way of returning the parameters
    if st_submit_button:
        state.submit_count    += 1
        state.st_output_dir   = st_output_dir
        state.st_script_dir   = st_script_dir
        state.st_script_name  = st_script_name

        state.st_script_args  = st_script_args
        state.st_script_env   = st_script_env

        state.st_dataset_name = st_dataset_name
        state.st_model_display_names = st_model_display_names 

        state.run_script      = True  # must the last to prevent race condition

        print(f"x{state.st_script_dir}")
        print(f"x{state.st_script_name}")
        print(f"x{state.st_script_args}")
        print(f"x{state.st_script_env}")
        print(f"x{state.run_script}")

if __name__ == "__main__":
  # unit test to run SelectDatasetUI by itself 
  # lightning run app select_fo_dataset.py
  import time
  class LitApp(LightningFlow):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ui = FoRunUI(
        script_dir="../lightning-pose", 
        script_name="run_fo_ui.py", 
        config_dir="../lightning-pose/scripts/configs", 
        config_ext="*.yaml", 
        script_args="""
eval.fiftyone.dataset_to_create="videos" \
eval.fiftyone.dataset_name=testvid \
eval.fiftyone.model_display_names=["testvid"] \
eval.fiftyone.build_speed="fast" \
eval.hydra_paths=['2022-05-15/16-06-45'] \
eval.fiftyone.launch_app_from_script=True \
eval.video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4" \
eval.pred_csv_files_to_plot=["./lightning-pose/toy_datasets/toymouseRunningData/test_vid_heatmap.csv"] 
""",
        script_env="script_env")
    #def run(self):
    #  print(self.ui.state)    
    def configure_layout(self):
      return{"name": "home", "content": self.ui}
  app = LightningApp(LitApp())
