import os
import logging
import string
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace
import shlex

from lai_components.hydra_ui import hydra_config, get_hydra_config_name, get_hydra_dir_name 
from lai_components.args_utils import args_to_dict, dict_to_args 
from lai_components.vsc_streamlit import StreamlitFrontend

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.components.python import TracerPythonScript
from lightning.app.utilities.state import AppState
from lightning.app.storage.path import Path


class FoRunUI(LightningFlow):
  """UI to enter training parameters
  Input and output variables with streamlit must be pre decleared
  """

  def __init__(
      self,
      *args, 
      script_dir, 
      script_name, 
      config_dir="./",
      config_name="config.yaml",
      script_args=None,
      script_env=None,
      outputs_dir="outputs",
      **kwargs
):
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
    self.config_name = config_name        

    self.script_args = script_args
    self.outputs_dir = outputs_dir
    # FO list
    self.fo_datasets = []
    self.hydra_outputs = {}

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
    self.st_hydra_config_name = None
    self.st_hydra_config_dir = None   

  def set_fo_dataset(self, names):
    self.fo_datasets = names

  def add_fo_dataset(self, name):
    self.fo_datasets.append(name)

  def set_hydra_outputs(self, names:dict):
    self.hydra_outputs.update(names)

  def add_hydra_output(self, name:str):
    self.hydra_outputs.update(name)

  def configure_layout(self):
    return StreamlitFrontend(render_fn=_render_streamlit_fn)


def get_existing_datasets():
  return(self.fo_datasets)

def set_script_args(st_output_dir:[str], script_args:str, script_dir:str, outputs_dir:str, hydra_outputs:dict):
  script_args_dict = args_to_dict(script_args)

  # enrich the args  
  # eval.video_file_to_plot="</ABSOLUTE/PATH/TO/VIDEO.mp4>" \

  # eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>","</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
  # eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"]
  # eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]

  if st_output_dir:  
    path_list = ','.join([f"'{x}'" for x in st_output_dir])
    script_args_dict["eval.hydra_paths"]=f"[{path_list}]"

    # set eval.pred_csv_files_to_plot

    # fiename is video_name*.csv
    video_file_name = script_args_dict["eval.video_file_to_plot"].split("/")[-1]
    video_file_name_basename = ".".join(video_file_name.split(".")[:-1])
    print(f"video_file_name_basename={video_file_name_basename}")
    
    pred_csv_files_to_plot=[]
    pred_csv_files_root_dir = os.path.abspath(f"{script_dir}/{outputs_dir}/")
    for hydra_name in st_output_dir:
      if hydra_name in hydra_outputs:
        file = hydra_outputs[hydra_name]
        print(f"found {file}")
        pred_csv_files_to_plot.append(f"'{pred_csv_files_root_dir}/{hydra_name}/{file}'")
    script_args_dict["eval.pred_csv_files_to_plot"] = "[%s]" % ",".join(pred_csv_files_to_plot) 
    print(f"pred_csv_files_to_plot={script_args_dict['eval.pred_csv_files_to_plot']}")

  # these will be controlled by the runners.  remove if set manually
  script_args_dict.pop('eval.fiftyone.address', None)
  script_args_dict.pop('eval.fiftyone.port', None)
  script_args_dict.pop('eval.fiftyone.launch_app_from_script', None)
  script_args_dict.pop('eval.fiftyone.dataset_to_create', None) 
  script_args_dict.pop('eval.fiftyone.dataset_name', None)
  script_args_dict.pop('eval.fiftyone.model_display_names', None)

  # convert to absolute
  script_args_dict["eval.video_file_to_plot"] = os.path.abspath(script_args_dict["eval.video_file_to_plot"]) 
  
  return(dict_to_args(script_args_dict), script_args_dict) 
  
def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset
    """

    # outputs to choose from
    st_output_dir = st.multiselect("select output", state.hydra_outputs)

    # for each output, choose a name 
    model_display_names_str = st.text_input("display name for each output separated by space")
    st_model_display_names = shlex.split(model_display_names_str)
    if len(st_model_display_names) != len(st_output_dir):
      st.error("unique names must be given to each output")
    print(f"{st_output_dir} {st_output_dir}")

    # dataset names
    existing_datasets = state.fo_datasets
    st.write(f"Existing Fifityone datasets {', '.join(existing_datasets)}")
    st_dataset_name = st.text_input("name other than the above existing names")
    if st_dataset_name in existing_datasets:
      st.error(f"{st_dataset_name} exists.  Please choose a new name.")
      st_dataset_name = None

    # parse

    state.script_args, script_args_dict = set_script_args(st_output_dir, script_args = state.script_args, script_dir = state.script_dir, outputs_dir = state.outputs_dir, hydra_outputs = state.hydra_outputs) 
    st_script_args = st.text_area("Script Args", value=state.script_args, placeholder='--a 1 --b 2')
    if st_script_args != state.script_args:
      state.script_args = st_script_args 

    # TODO:
    # do not show outputs that does not have predict.csv and another CSV

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

    st_hydra_config = hydra_config(context=expander, config_dir=state.config_dir, config_name=state.config_name, root_dir=st_script_dir)

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

        state.st_hydra_config_name = get_hydra_config_name()
        state.st_hydra_config_dir = get_hydra_dir_name()

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
        config_dir="../scripts/configs", 
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
