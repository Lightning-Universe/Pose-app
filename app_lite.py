import os
import sys
import shlex
from string import Template
import lightning as L
import streamlit as st

from lightning_pose.bashwork import LitBashWork

import logging
import time

class LitPoseApp(L.LightningFlow):
  def __init__(self):
    super().__init__()
    # self.dataset_ui = SelectDatasetUI()
    self.fo_names = None
    self.fo_launch = None

    self.my_work = LitBashWork(cloud_compute=L.CloudCompute("gpu-fast"))
      
  def run(self):
    self.my_work.run("tensorboard --logdir outputs --host %s --port %d" % (self.my_work.host, self.my_work.get_port("TB")),
      cwd="lightning_pose")
    self.my_work.run("fiftyone app launch --address %s --port %d" % (self.my_work.host, self.my_work.get_port("FO")),
      cwd="lightning_pose" )
    self.my_work.run("python scripts/train.py",
      cwd="lightning_pose" )

  def configure_layout(self):
      train_diag_tab = {"name": "Train Diag", "content":"http://%s:%d" % (self.my_work.host, self.my_work.get_port("TB"))}
      image_diag_tab = {"name": "Image/Video Diag", "content":"http://%s:%d" % (self.my_work.host, self.my_work.get_port("FO"))}
      return [train_diag_tab, image_diag_tab]

app = L.LightningApp(LitPoseApp())
