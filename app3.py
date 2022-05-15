# app.py
import os
import lightning as L
import streamlit as st
from lai_components.run_ui import ScriptRunUI
from lai_components.chdir_script import ChdirPythonScript
from lai_components.run_tb import RunTensorboard
import logging

class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.fo_ui = ScriptRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/train_hydra.py",
          config_dir = "./lightning-pose/configs",
          config_ext = "*.yaml",        
          script_args = """\
            training.max_epochs=11 \
            data.data_dir=./lightning-pose/toy_datasets/toymouseRunningData \
            model.losses_to_use=[] \
          """
        )
        self.train_ui = ScriptRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/create_fiftyone_dataset.py",
          config_dir = "./lightning-pose/configs",
          config_ext = "*.yaml",        
          script_args = """\
            eval.fiftyone.dataset_name=test1
            eval.fiftyone.model_display_names=["test1"] \
            eval.fiftyone.port=${port} \  
            eval.fiftyone.dataset_to_create="images" \
            eval.fiftyone.build_speed="fast" \
            eval.fiftyone.launch_app_from_script=True \
            test_videos_directory="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos"
            saved_vid_preds_dir="./lightning-pose/toy_datasets/toymouseRunningData
            video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4"
            pred_csv_files_to_plot=["./lightning-pose/test_vid_heatmap_pca_multiview_7.500.csv"]  
            eval.hydra_paths=['./lightning-pose'] \
            """  
        )        
        self.run_tb = RunTensorboard(blocking=False, run_once=True)

        # script_path is required at init, but will be override in the run
        self.fo_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)   
        self.train_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)

    def run(self):
      self.run_tb.run()

      if self.fo_ui.st_submit:      
        self.fo_ui.st_submit = False
        self.fo_runner.run(root_dir = self.fo_ui.st_script_dir, 
          script_name = self.fo_ui.st_script_name, 
          script_args=self.fo_ui.st_script_args,
          script_env=self.train_ui.st_script_env,
          )

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
        tab4 = {"name": "Fiftyone", "content": self.fo_runner}
        return [tab1, tab2, tab3]

logging.basicConfig(level=logging.DEBUG)
app = L.LightningApp(LitApp())
