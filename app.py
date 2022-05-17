# app.py
import os
import lightning as L
import streamlit as st
from lai_components.run_ui import ScriptRunUI
from lai_components.chdir_script import ChdirPythonScript
from lai_components.run_tb import RunTensorboard
from lai_components.select_fo_dataset import RunFiftyone, SelectDatasetUI
import logging
import time

class LitPoseApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.dataset_ui = SelectDatasetUI()

        self.train_ui = ScriptRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/train_hydra.py",
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./lightning-pose/scripts/configs",
          config_ext = "*.yaml",        
          script_args = """training.max_epochs=11 
model.losses_to_use=[] 
          """
        )
        self.image_fo_ui = ScriptRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/create_fiftyone_dataset.py",
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./lightning-pose/scripts/configs",
          config_ext = "*.yaml",        
          script_args = """eval.fiftyone.dataset_name=test1 
eval.fiftyone.model_display_names=["test1"]
eval.hydra_paths=['2022-05-15/16-06-45']
eval.fiftyone.address=${host} 
eval.fiftyone.port=${port} 
eval.fiftyone.dataset_to_create="images"
eval.fiftyone.build_speed="fast" 
eval.fiftyone.launch_app_from_script=False 
            """  
        )    


        self.run_tb = RunTensorboard(logdir = "./lightning-pose/outputs", blocking=False, run_once=True)

        # script_path is required at init, but will be override in the run
        self.dataset_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)   
        self.train_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)
        self.fo_runner = RunFiftyone(blocking=True,run_once=True)

    def run(self):
      self.run_tb.run()

      if self.dataset_ui.st_submit:      
        self.dataset_ui.st_submit = False
        print(f"st_selectbox={self.dataset_ui.st_selectbox}")
        #time.sleep(10) # runs too fast will come in here twice
        if not(self.image_dataset_ui.st_selectbox is None):
          self.fo_runner.run(dataset_name = self.image_dataset_ui.st_selectbox)

      # create new dataset
      if self.image_fo_ui.st_submit:      
        self.image_fo_ui.st_submit = False
        self.dataset_runner.run(root_dir = self.image_fo_ui.st_script_dir, 
          script_name = self.image_fo_ui.st_script_name, 
          script_args=self.image_fo_ui.st_script_args,
          script_env=self.image_fo_ui.st_script_env,
          )
        self.dataset_ui.set_dateset_names()

      if self.train_ui.st_submit:      
        self.train_ui.st_submit = False
        self.train_runner.run(root_dir = self.train_ui.st_script_dir, 
          script_name = self.train_ui.st_script_name, 
          script_args=self.train_ui.st_script_args,
          script_env=self.train_ui.st_script_env,
          )

    def configure_layout(self):
        tab1 = {"name": "Train", "content": self.train_ui}
        tab2 = {"name": "Dataset", "content": self.fo_ui}
        tab3 = {"name": "Tensorboard", "content": self.run_tb}
        tab4 = {"name": "Pick Dataset", "content": self.image_dataset_ui}
        tab5 = {"name": "Fiftyone", "content": self.fo_runner}
        return [tab1, tab2, tab3, tab4, tab5]

logging.basicConfig(level=logging.DEBUG)
app = L.LightningApp(LitPoseApp())
