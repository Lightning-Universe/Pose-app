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

# data.data_dir=./lightning-pose/toy_datasets/toymouseRunningData 
# Saved predictions to: pred_csv_files_to_plot=/home/jovyan/lightning-pose-app/lightning-pose/outputs/2022-05-15/16-06-45/predictions.csv
#             pred_csv_files_to_plot=["./lightning-pose/outputs/2022-05-15/16-06-45/predictions.csv"]  
#            test_videos_directory="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos" \##
#            saved_vid_preds_dir="./lightning-pose/toy_datasets/toymouseRunningData" \
#            video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4" \

class LitPoseApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # self.dataset_ui = SelectDatasetUI()

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

        self.fo_image_ui = ScriptRunUI(
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
eval.fiftyone.launch_app_from_script=True 
            """  
        )   

#eval.test_videos_directory="</ABSOLUTE/PATH/TO/VIDEOS/DIR>" \
#eval.video_file_to_plot="</ABSOLUTE/PATH/TO/VIDEO.mp4>" \
#eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]


        self.fo_video_ui = ScriptRunUI(
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
eval.fiftyone.dataset_to_create="videos"
eval.fiftyone.build_speed="fast" 
eval.fiftyone.launch_app_from_script=True 
            """  
        )   

        self.run_tb = RunTensorboard(logdir = "./lightning-pose/outputs", blocking=False, run_once=True)

        # script_path is required at init, but will be override in the run
        self.train_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)
        self.fo_image_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)   
        self.fo_video_runner = ChdirPythonScript("app.py",blocking=True,run_once=False)

        # self.fo_runner = RunFiftyone(blocking=True,run_once=True)

    def run(self):
      self.run_tb.run()

      #if self.dataset_ui.st_submit:      
      #  self.dataset_ui.st_submit = False
      #  print(f"st_selectbox={self.dataset_ui.st_selectbox}")
      #  #time.sleep(10) # runs too fast will come in here twice
      #  if not(self.dataset_ui.st_selectbox is None):
      #    self.fo_runner.run(dataset_name = self.dataset_ui.st_selectbox)

      # create new image dataset
      if self.fo_image_ui.st_submit:      
        self.fo_image_ui.st_submit = False
        self.fo_image_runner.run(root_dir = self.fo_image_ui.st_script_dir, 
          script_name = self.fo_image_ui.st_script_name, 
          script_args=self.fo_image_ui.st_script_args,
          script_env=self.fo_image_ui.st_script_env,
          )

      # create new video dataset
      if self.fo_video_ui.st_submit:      
        self.fo_video_ui.st_submit = False
        self.fo_video_runner.run(root_dir = self.fo_video_ui.st_script_dir, 
          script_name = self.fo_video_ui.st_script_name, 
          script_args=self.fo_video_ui.st_script_args,
          script_env=self.fo_video_ui.st_script_env,
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
        tab2 = {"name": "Create Image Dataset", "content": self.fo_image_ui}
        tab3 = {"name": "Create Video Dataset", "content": self.fo_video_ui}
        tab4 = {"name": "Tensorboard", "content": self.run_tb}
        tab5 = {"name": "Fiftyone Images", "content": self.fo_image_runner}
        tab6 = {"name": "Fiftyone Videos", "content": self.fo_video_runner}

        return [tab1, tab2, tab3, tab4, tab5, tab6]

logging.basicConfig(level=logging.DEBUG)
app = L.LightningApp(LitPoseApp())
