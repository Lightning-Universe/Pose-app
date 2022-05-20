# app.py
import os
import lightning as L
import streamlit as st
from lai_components.run_fo_ui import FoRunUI
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
        self.fo_names = None
        self.fo_launch = None

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

# the following are added for FO
#eval.hydra_paths=['2022-05-15/16-06-45']
#eval.fiftyone.address=${host} 
#eval.fiftyone.port=${port} 
# these are changed to absolute path
#eval.test_videos_directory="</ABSOLUTE/PATH/TO/VIDEOS/DIR>" \
#eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]

        self.fo_ui = FoRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/create_fiftyone_dataset.py",
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./lightning-pose/scripts/configs",
          config_ext = "*.yaml",        
          script_args = """eval.fiftyone.dataset_name=test1 
eval.fiftyone.model_display_names=["test1"]
eval.fiftyone.dataset_to_create="images"
eval.fiftyone.build_speed="fast" 
eval.fiftyone.launch_app_from_script=True 
eval.video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4"
eval.pred_csv_files_to_plot=["./lightning-pose/toy_datasets/toymouseRunningData/test_vid_heatmap.csv"]
            """  
        )   

        # tensorboard
        self.run_tb = RunTensorboard(logdir = "./lightning-pose/outputs")
        self.run_fo = RunFiftyone()

        # script_path is required at init, but will be override in the run
        self.train_runner = ChdirPythonScript("./lightning-pose/scripts/train_hydra.py")
        # 
        self.fo_predict_runner = ChdirPythonScript("./lightning-pose/scripts/predict_new_vids.py")
        self.fo_image_runner = ChdirPythonScript("./lightning-pose/scripts/create_fiftyone_dataset.py")
        self.fo_video_runner = ChdirPythonScript("./lightning-pose/scripts/create_fiftyone_dataset.py")


    def run(self):
      #self.run_tb.run()

      if self.train_ui.run_script == True:      
        self.train_runner.run(root_dir = self.train_ui.st_script_dir, 
          script_name = self.train_ui.st_script_name, 
          script_args=self.train_ui.st_script_args,
          script_env=self.train_ui.st_script_env,
          )  
        if self.train_runner.has_succeeded:
          self.train_ui.run_script = False    

      # create fo dataset
      if self.fo_ui.run_script == True:      
        self.fo_names = f"eval.fiftyone.dataset_name={self.fo_ui.st_dataset_name} eval.fiftyone.model_display_names=['{self.fo_ui.st_dataset_name}']"
        self.fo_launch=f"eval.fiftyone.launch_app_from_script=False"
        self.fo_predict_runner.run(root_dir = self.fo_ui.st_script_dir, 
          script_name = "scripts/predict_new_vids.py", 
          script_args=f"{self.fo_ui.st_script_args} {self.fo_names}",
          script_env=self.fo_ui.st_script_env,
          )
        if self.fo_predict_runner.has_succeeded:
          self.fo_image_runner.run(root_dir = self.fo_ui.st_script_dir, 
            script_name = "scripts/create_fiftyone_dataset.py", 
            script_args=f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=images {self.fo_names} {self.fo_launch}",
            script_env=self.fo_ui.st_script_env,
            )
        if self.fo_image_runner.has_succeeded:
          self.fo_video_runner.run(root_dir = self.fo_ui.st_script_dir, 
            script_name = "scripts/create_fiftyone_dataset.py", 
            script_args=f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=videos {self.fo_names} {self.fo_launch}",
            script_env=self.fo_ui.st_script_env,
            )
        if self.fo_video_runner.has_succeeded:   
          self.fo_launch=f"eval.fiftyone.address=$host eval.fiftyone.port=$port eval.fiftyone.launch_app_from_script=True eval.fiftyone.remote=False"
          self.run_fo.run()
          self.fo_ui.run_script = False

    def configure_layout(self):
        tab1 = {"name": "Train", "content": self.train_ui}
        tab2 = {"name": "Tensorboard", "content": self.run_tb}
        tab3 = {"name": "Create Image Dataset", "content": self.fo_ui}
        tab4 = {"name": "Fiftyone", "content": self.run_fo}

        return [tab1, tab2, tab3, tab4]

logging.basicConfig(level=logging.INFO)
app = L.LightningApp(LitPoseApp())
