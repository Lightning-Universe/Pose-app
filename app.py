# app.py
import os
import sys
import shlex
from string import Template
from typing import Optional, Union, List

import lightning_app as L
from lightning.app.storage.drive import Drive
import streamlit as st


from lai_work.bashwork import LitBashWork

from lai_components.run_fo_ui import FoRunUI
from lai_components.run_ui import ScriptRunUI
from lai_components.run_config_ui import ConfigUI
from lai_components.args_utils import args_to_dict, splitall
from lai_components.lpa_utils import output_with_video_prediction


import logging
import time

# sub dirs
#   lightening-pose
#   label-studio
lightning_pose_dir  = "lightning-pose"
label_studio_dir    = "label-studio"
# virtualenv names located in ~
label_studio_venv   = "venv-label-studio"
lightning_pose_venv = "venv-lightning-pose"
tensorboard_venv    = "venv-tensorboard"

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

class TensorboardBuildConfig(L.BuildConfig):
  def build_commands(self) -> List[str]:
    return [
      f"virtualenv ~/{tensorboard_venv}",
      f". ~/{tensorboard_venv}/bin/activate; python -m pip install tensorflow tensorboard;deactivate",
    ]

class LabelStudioBuildConfig(L.BuildConfig):
  def build_commands(self) -> List[str]:
    return [
      f"virtualenv ~/{label_studio_venv}",
      "git clone https://github.com/robert-s-lee/label-studio",
      "cd label-studio; git checkout x-frame-options; cd ..",
      # source is not available, so using "."
      f". ~/{label_studio_venv}/bin/activate; cd label-studio; which python; python -m pip install -e .;deactivate",
      # TODO: after PR is merged,
      # f". ~/{label_studio_venv}/bin/activate; which python; python -m pip install label-studio",
    ]

class FiftyOneBuildConfig(L.BuildConfig):
  def build_commands(self) -> List[str]:
    return [
      "sudo apt-get update",
      "sudo apt-get install -y ffmpeg libsm6 libxext6",
      f"virtualenv ~/{lightning_pose_venv}",
      f". ~/{lightning_pose_venv}/bin/activate;python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102; deactivate",
      f". ~/{lightning_pose_venv}/bin/activate;python -m pip install -e {lightning_pose_dir}; deactivate",
    ]

# data.data_dir=./lightning-pose/toy_datasets/toymouseRunningData 
# Saved predictions to: pred_csv_files_to_plot=/home/jovyan/lightning-pose-app/lightning-pose/outputs/2022-05-15/16-06-45/predictions.csv
#             pred_csv_files_to_plot=["./lightning-pose/outputs/2022-05-15/16-06-45/predictions.csv"]  
#            test_videos_directory="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos" \##
#            saved_vid_preds_dir="./lightning-pose/toy_datasets/toymouseRunningData" \
#            video_file_to_plot="./lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4" \

class LitPoseApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # shared data for apps
        self.drive_lpa = Drive("lit://lpa")
        # 
        self.args_append = None
        # UIs
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
training.num_workers=2
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
        self.my_tb = LitBashWork(
          cloud_compute=L.CloudCompute("default"), 
          cloud_build_config=TensorboardBuildConfig(),
          )
        self.my_label_studio = LitBashWork(
          cloud_compute=L.CloudCompute("default"), 
          cloud_build_config=LabelStudioBuildConfig(),
          )
        self.my_work = LitBashWork(
          cloud_compute=L.CloudCompute("gpu"), 
          cloud_build_config=FiftyOneBuildConfig(),
          )

    def init_lp_outputs_to_ui(self):
      # get existing hydra datasets 
      # that has test*.csv
      cmd = f"find {self.train_ui.outputs_dir} -maxdepth 3 -type f -name *.csv -not -name predictions.csv"  
      self.my_work.run(cmd,
        cwd=lightning_pose_dir)
      if (self.my_work.last_args() == cmd):
        outputs = output_with_video_prediction(self.my_work.last_stdout())
        print(outputs)
        self.train_ui.set_hydra_outputs(outputs)
        self.fo_ui.set_hydra_outputs(outputs)
        self.my_work.reset_last_args()

    def start_label_studio(self):
        """run label studio migrate, then runserver"""
        # Install for local development
        # https://github.com/heartexlabs/label-studio#install-for-local-development
        self.my_label_studio.run(
          f"python label_studio/manage.py migrate", 
          venv_name=label_studio_venv,
          cwd=label_studio_dir)
        self.my_label_studio.run(
          "python label_studio/manage.py runserver {host}:{port}", 
          venv_name=label_studio_venv,
          wait_for_exit=False,    
          env={
            # label-studio/label_studio/core/settings/label_studio.py
            # label-studio/label_studio/core/settings/base.py
            # label-studio/label_studio/core/middleware.py
            'USE_ENFORCE_CSRF_CHECKS':'false',
            'LABEL_STUDIO_X_FRAME_OPTIONS':'allow-from *', 
            'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED':'true', 
            'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT':os.path.abspath(os.getcwd())
            },
          cwd=label_studio_dir)

        # TODO: use this once https://github.com/heartexlabs/label-studio/issues/2596 is resolved
        #self.subprocess_call(f"label-studio start --port={self.port} --internal-host={self.host}")

    def start_tensorboard(self):
      """run tensorboard"""
      cmd = "tensorboard --logdir outputs --host {host} --port {port}"
      self.my_tb.run(cmd,
        venv_name=tensorboard_venv,
        wait_for_exit=False, 
        cwd=lightning_pose_dir, 
      )     
    def init_fiftyone_outputs_to_ui(self):
      # get existing fiftyone datasets
      cmd = "fiftyone datasets list"  
      self.my_work.run(cmd,
          venv_name=lightning_pose_venv,
        )
      if (self.my_work.last_args() == cmd):
        options = []
        for x in self.my_work.stdout:
          if x.endswith("No datasets found"):
            continue
          if x.startswith("Migrating database"):
            continue
          options.append(x)    
        self.fo_ui.set_fo_dataset(options)

    def start_fiftyone(self):
      """start the background service"""
      # start the fiftyone
      # TODO:
      #   right after fiftyone, the previous find command is triggered should not be the case.
      cmd = "fiftyone app launch --address {host} --port {port}"
      self.my_work.run(cmd,
        venv_name=lightning_pose_venv,
        wait_for_exit=False, 
        cwd=lightning_pose_dir)

    def start_lp_train_video_predict(self):
        # output for the train
        train_args = args_to_dict(self.train_ui.st_script_args)         # dict version of arg to trainer
        hydra_run_dir = train_args['hydra.run.dir']                     # outputs/%Y-%m-%d/%H-%M-%S
        eval_hydra_paths = os.path.join(*splitall(hydra_run_dir)[-2:])  # %Y-%m-%d/%H-%M-%S
        # train
        cmd = "python " + self.train_ui.st_script_name + " " + self.train_ui.st_script_args 
        self.my_work.run(cmd,
          venv_name=lightning_pose_venv,
          env = self.train_ui.st_script_env,
          cwd = self.train_ui.st_script_dir, 
          outputs = [os.path.join(self.train_ui.st_script_dir,self.train_ui.outputs_dir)],
          )    

        # video
        eval_test_videos_directory = os.path.abspath(self.train_ui.st_eval_test_videos_directory)
        root_dir = os.path.abspath(self.train_ui.st_script_dir)
        script_args = f"eval.hydra_paths=[{eval_hydra_paths}] eval.test_videos_directory={eval_test_videos_directory} eval.saved_vid_preds_dir={hydra_run_dir}"
        cmd = "python " + "scripts/predict_new_vids.py" + " " + script_args
        self.my_work.run(cmd,
          venv_name=lightning_pose_venv,
          env = self.train_ui.st_script_env,
          cwd = self.train_ui.st_script_dir,
          )          

        # set the new outputs for UIs
        cmd = f"find {hydra_run_dir} -maxdepth 3 -type f -name *.csv -not -name predictions.csv"  
        self.my_work.run(cmd,
          cwd=lightning_pose_dir)
        if (self.my_work.last_args() == cmd):
          outputs = output_with_video_prediction(self.my_work.last_stdout())
          self.train_ui.set_hydra_outputs(outputs)
          self.fo_ui.set_hydra_outputs(outputs)
          self.my_work.reset_last_args()

        # have TB pull the new data
        cmd = f"{hydra_run_dir}"  # make this unique
        self.my_tb.run(cmd,
          venv_name=tensorboard_venv,
          cwd=lightning_pose_dir, 
          input_output_only = True,
          inputs = [os.path.join(self.train_ui.script_dir,self.train_ui.outputs_dir)],
        )
        
        # indicate to UI  
        self.train_ui.run_script = False    

    def start_lp_image_video_eval(self):
        self.args_append = f"eval.fiftyone.dataset_name={self.fo_ui.st_dataset_name}"
        self.args_append += " " + "eval.fiftyone.model_display_names=[%s]" % ','.join([f"'{x}'" for x in self.fo_ui.st_model_display_names]) 
        self.args_append += " " + f"eval.fiftyone.launch_app_from_script=False"
        self.args_append += " " + self.fo_ui.st_hydra_config_name
        self.args_append += " " + self.fo_ui.st_hydra_config_dir
        cmd = "python " + "scripts/create_fiftyone_dataset.py" + " " + f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=images {self.args_append}"
        self.my_work.run(cmd,
          venv_name=lightning_pose_venv,
          env= self.fo_ui.st_script_env,
          cwd = self.fo_ui.st_script_dir, 
          )
        cmd = "python " + "scripts/create_fiftyone_dataset.py" + " " + f"{self.fo_ui.st_script_args} eval.fiftyone.dataset_to_create=videos {self.args_append}"
        self.my_work.run(cmd,
          venv_name=lightning_pose_venv,
          env = self.fo_ui.st_script_env,
          cwd = self.fo_ui.st_script_dir, 
          )
        # add both names
        self.fo_ui.add_fo_dataset(self.fo_ui.st_dataset_name)
        self.fo_ui.add_fo_dataset(f"{self.fo_ui.st_dataset_name}_video")
        # indicate to UI  
        self.fo_ui.run_script = False

    def run(self):
      # init once 
      self.init_lp_outputs_to_ui()
      self.init_fiftyone_outputs_to_ui()

      # background services once
      self.start_tensorboard()
      self.start_label_studio()
      self.start_fiftyone()

      # train on ui button press  
      if self.train_ui.run_script == True:      
        self.start_lp_train_video_predict()
      # create fo dateset on ui button press  
      if self.fo_ui.run_script == True:      
        self.start_lp_image_video_eval()

    def configure_layout(self):
        config_tab = {"name": "Lightning Pose", "content": self.config_ui}
        train_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Diag", "content": self.my_tb}
        image_diag_prep_tab = {"name": "Image/Video Diag Prep", "content": self.fo_ui}
        image_diag_tab = {"name": "Image/Video Diag", "content": self.my_work}
        data_anntate_tab = {"name": "Image/Video Annotation", "content": self.my_label_studio}
        return [config_tab, train_tab, train_diag_tab, image_diag_prep_tab, image_diag_tab, data_anntate_tab]

app = L.LightningApp(LitPoseApp())
