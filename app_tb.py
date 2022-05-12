import os
import logging
from typing import Any, Dict, Optional, Union
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from pose_app.train import *
from pose_app.tb import *
from pose_app.fo import *

script_tb = TracerPythonScript(
  script_path = "run_tb.py",
  script_args = ["--server=127.0.0.1","--logdir=./"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  port = 6006,
  raise_exception = True,
  )

class App(LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.work_tb = script_tb

  def run(self):
    self.work_tb.run()

    def configure_layout(self):


        #if self.work_tb.has_started:
        tab4 = {"name": "Tensorboard", "content": "http://127.0.0.1:6006"}
        #else:
        #    tab4 = {"name": "Tensorboard", "content": ""}

        return [tab4]

app = LightningApp(App())
