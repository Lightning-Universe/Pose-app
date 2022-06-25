# app.py
import os
import sys
import shlex
from string import Template
import lightning as L
import streamlit as st

from lai_work.bashwork import LitBashWork

from lai_components.run_fo_ui import FoRunUI
from lai_components.run_ui import ScriptRunUI
from lai_components.run_config_ui import ConfigUI

import logging
import time

lightning_pose_dir = "./lightning-pose"

# hydra.run.dir
#   outputs/YY-MM-DD/HH-MM-SS
# eval.hydra_paths
# eval_hydra_paths
#   YY-MM-DD/HH-MM-SS
predict_args="""
eval.hydra_paths=["${eval_hydra_paths}"] \
eval.test_videos_directory=${root_dir}/${eval_test_videos_directory} \
eval.saved_vid_preds_dir="${root_dir}/${hydra.run.dir}/
"""

def args_to_dict(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  script_args_dict = {}
  for x in shlex.split(script_args, posix=False):
    k,v = x.split("=",1)
    script_args_dict[k] = v
  return(script_args_dict) 

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
        self.args_append = None

        self.config_ui = ConfigUI(
          script_dir = lightning_pose_dir,
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./scripts",
          eval_test_videos_directory = "./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos",     
        )

        self.train_ui = ScriptRunUI(
          script_dir = lightning_pose_dir,
          script_name = "scripts/train_hydra.py",
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./scripts",
          script_args = """training.max_epochs=11
model.losses_to_use=[]
""",
          eval_test_videos_directory = "./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos",     
        )

        self.fo_ui = FoRunUI(
          script_dir = "./lightning-pose",
          script_name = "scripts/create_fiftyone_dataset.py",
          script_env = "HYDRA_FULL_ERROR=1",
          config_dir = "./scripts",
          script_args = """eval.fiftyone.dataset_name=test1 
eval.fiftyone.model_display_names=["test1"]
eval.fiftyone.dataset_to_create="images"
eval.fiftyone.build_speed="fast" 
eval.fiftyone.launch_app_from_script=True 
eval.video_file_to_plot=./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4
"""  
        )   

        # workers
        self.my_tb = LitBashWork(cloud_compute=L.CloudCompute("small-cpu"))
        self.my_work = LitBashWork(cloud_compute=L.CloudCompute("gpu"))

    def run(self):
      # run tb
      self.my_tb.run("tensorboard --logdir outputs --host %s --port %d" % (self.my_tb.host, self.my_tb.port),
        wait_for_exit=False, cwd=lightning_pose_dir)
      # get fiftyone datasets  
      self.my_work.run("fiftyone datasets list", save_stdout = True)
      if not (self.my_work.stdout is None):
        self.fo_ui.set_fo_dataset(self.my_work.stdout)
        self.my_work.stdout = None
      # start the fiftyone
      self.my_work.run("fiftyone app launch --address %s --port %d" % (self.my_work.host, self.my_work.port),
        wait_for_exit=False, cwd=lightning_pose_dir)
      # train on ui button press  
      if self.train_ui.run_script == True:      
        # train 
        cmd = "python " + self.train_ui.st_script_name + " " + self.train_ui.st_script_args 
        self.my_work.run(cmd,
          env=self.train_ui.st_script_env,
          cwd = self.train_ui.st_script_dir, 
          )    
        # video
        train_args = args_to_dict(self.train_ui.st_script_args)
        hydra_run_dir = train_args['hydra.run.dir']
        eval_hydra_paths = "/".join(hydra_run_dir.split("/")[-2:])
        eval_test_videos_directory = os.path.abspath(self.train_ui.st_eval_test_videos_directory)
        root_dir = os.path.abspath(self.train_ui.st_script_dir)
        script_args = f"eval.hydra_paths=[{eval_hydra_paths}] eval.test_videos_directory={eval_test_videos_directory} eval.saved_vid_preds_dir={hydra_run_dir}"
        cmd = "python " + "scripts/predict_new_vids.py" + " " + script_args
        self.my_work.run(cmd,
          env=self.train_ui.st_script_env,
          cwd = self.train_ui.st_script_dir,
          )          
        # indicate to UI  
        self.train_ui.run_script = False    
  
      # create fo dateset on ui button press  
      if self.fo_ui.run_script == True:      
        self.args_append = f"eval.fiftyone.dataset_name={self.fo_ui.st_dataset_name}"
        self.args_append += " " + "eval.fiftyone.model_display_names=[%s]" % ','.join([f"'{x}'" for x in self.fo_ui.st_model_display_names]) 
        self.args_append += " " + f"eval.fiftyone.launch_app_from_script=False"
        self.args_append += " " + self.fo_ui.st_hydra_config_name
        self.args_append += " " + self.fo_ui.st_hydra_config_dir
        cmd = "python " + "scripts/create_fiftyone_dataset.py" + " " + f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=images {self.args_append}"
        self.my_work.run(cmd,
          env=self.fo_ui.st_script_env,
          cwd = self.fo_ui.st_script_dir, 
          )
        cmd = "python " + "scripts/create_fiftyone_dataset.py" + " " + f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=videos {self.args_append}"
        self.my_work.run(cmd,
          env=self.fo_ui.st_script_env,
          cwd = self.fo_ui.st_script_dir, 
          )
        # add both names
        self.fo_ui.add_fo_dataset(self.fo_ui.st_dataset_name)
        self.fo_ui.add_fo_dataset(f"{self.fo_ui.st_dataset_name}_video")
        # indicate to UI  
        self.fo_ui.run_script = False

    def configure_layout(self):
        config_tab = {"name": "Lightning Pose", "content": self.config_ui}
        train_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Diag", "content": self.my_tb}
        image_diag_prep_tab = {"name": "Image/Video Diag Prep", "content": self.fo_ui}
        image_diag_tab = {"name": "Image/Video Diag", "content": self.my_work}
        data_anntate_tab = {"name": "Image/Video Annotation", "content": "https://cvat.org/"}
        return [config_tab, train_tab, train_diag_tab, image_diag_prep_tab, image_diag_tab, data_anntate_tab]

logging.basicConfig(level=logging.INFO)
app = L.LightningApp(LitPoseApp())
