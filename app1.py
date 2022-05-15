import os
import logging
from typing import Any, Dict, Optional, Union
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from pose_app.train import *
from pose_app.tb import *
from pose_app.fo import *


class App(LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.ui_train = TrainUI()
    self.work_train = TrainScript(script_path="./app1.py")
    self.work_tb = RunTensorboard(logdir="/home/jovyan/lightning-pose/outputs")
    self.ui_fo = CreateFiftyoneUI()
    self.work_fo = CreateFiftyoneDataset(script_path="./app1.py")

  def run(self):
    self.ui_train.run()
    self.work_tb.run()
    # train
    if self.ui_train.st_submit:
      self.ui_train.st_submit = False
      self.work_train.run(
        script_path=os.path.join(self.ui_train.st_script_dir, self.ui_train.st_script_name),
        script_args=self.ui_train.st_script_arg,
        script_env=self.ui_train.st_script_env)
    # fo      
    self.ui_fo.run()
    if self.ui_fo.st_submit:
      self.ui_fo.st_submit = False
      self.work_fo.run(
        script_path=os.path.join(self.ui_fo.st_script_dir, self.ui_fo.st_script_name),
        script_args=self.ui_fo.st_script_arg,
        script_env=self.ui_fo.st_script_env)

    def configure_layout(self):
        tab1 = {"name": "Lightning Pose Param", "content": self.ui_train}
        tab2 = {"name": "Fiftyone Dataset Param", "content": self.ui_fo}

        if self.work_fo.has_started:
            tab3 = { "name": "Fiftyone","content": self.work_fo}
        else:
            tab3 = {"name": "Fiftyone", "content": ""}

        if self.work_tb.has_started:
            tab4 = {"name": "Tensorboard", "content": self.work_tb}
        else:
            tab4 = {"name": "Tensorboard", "content": ""}

        return [tab1, tab2, tab3, tab4]

app = LightningApp(App())
