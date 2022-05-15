import os
import logging
from typing import Any, Dict, Optional, Union
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import numpy as np
import signal
from lightning.components.python import TracerPythonScript

work_tb = TracerPythonScript(
  script_path = "run_tb.py",
  script_args = [],
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
    self.work_tb = work_tb

  def run(self):
    self.work_tb.run()

  def configure_layout(self):
    tab4 = {"name": "Tensorboard", "content": self.work_tb}
    return [tab4]

app = LightningApp(App())
