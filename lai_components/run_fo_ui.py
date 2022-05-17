import os
import logging
import string

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

def set_script_args(st_output_dir:str, script_args:str):
  script_args_dict = {}
  script_args_array = []
  for x in shlex.split(script_args, posix=False):
    k,v = x.split("=",1)
    script_args_dict[k] = v
  # enrich the args  
  if st_output_dir:  
    script_args_dict["eval.hydra_paths"]=st_output_dir

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
  script_args_dict.pop('eval.launch_app_from_script', None)
  script_args_dict.pop('eval.fiftyone.dataset_to_create', None) 
  script_args_dict.pop('eval.fiftyone.dataset_name', None)
  script_args_dict.pop('eval.fiftyone.model_display_names', None)


  for k,v in script_args_dict.items():
    script_args_array.append(f"{k}={v}")
  return(" \n".join(script_args_array)) 
  
def get_existing_outpts(state):
  options = ["/".join(x.strip().split("/")[-2:]) for x in sh.find(f"{state.script_dir}/{state.outputs_dir}","-mindepth","2","-maxdepth","2","-type","d")]
  options.sort(reverse=True)
  return(options)

def get_existing_datasets():
  options = fo.list_datasets()
  options.remove('')
  return(options)

def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset
    """

    # outputs to choose from
    st_output_dir = st.selectbox("select output", get_existing_outpts(state))

    # dataset names
    existing_datasets = get_existing_datasets()
    st.write(f"Existing Fifityone datasets {', '.join(existing_datasets)}")
    st_dataset_name = st.text_input("Fiftyone dataset name", placeholder=f"other than above existing names")
    if st_dataset_name in existing_datasets:
      st.error(f"{st_dataset_name} exists.  Please choose a new name.")
      st_dataset_name = None

    # parse
    state.script_args = set_script_args(st_output_dir, state.script_args) 
    st_script_args = st.text_area("Script Args", value=state.script_args, placeholder='--a 1 --b 2')
    state.script_args = st_script_args

    st_script_env = st.text_input("Script Env Vars", value=state.script_env, placeholder="ABC=123 DEF=345")

    st_submit_button = st.button("Submit", disabled=True if st_dataset_name is None else False)

    # these are not used as often
    expander = st.expander("Change Defaults")

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
    
    # Lightning way of returning the parameters
    if st_submit_button and st_dataset_name:
        state.submit_count    += 1
        state.st_output_dir   = st_output_dir
        state.st_script_dir   = st_script_dir
        state.st_script_name  = st_script_name

        state.st_script_args  = st_script_args
        state.st_script_env   = st_script_env

        state.st_dataset_name = st_dataset_name
        
        state.st_submit       = True  # must the last to prevent race condition

        print(f"x{state.st_script_dir}")
        print(f"x{state.st_script_name}")
        print(f"x{state.st_script_args}")
        print(f"x{state.st_script_env}")
        print(f"x{state.st_submit}")

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
eval.video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4" \
eval.pred_csv_files_to_plot=["./lightning-pose/toy_datasets/toymouseRunningData/test_vid_heatmap.csv"] 
""",
        script_env="script_env")
    #def run(self):
    #  print(self.ui.state)    
    def configure_layout(self):
      return{"name": "home", "content": self.ui}
  app = LightningApp(LitApp())
