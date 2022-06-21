import os
import numpy as np
import subprocess
import signal

from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from datetime import datetime
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning_app.storage.path import Path
import logging


class RunTensorboard(LightningWork):
  def __init__(self, *args, log_dir:str = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.log_dir = log_dir

  def generate_log(self, dataset_name):
    # fake tensorboard logs (fake loss)
    logging.info(f"Generating fake Tensorboard output on {dataset_name}")
    writer = SummaryWriter(log_dir=dataset_name)
    offset = np.random.uniform(0, 5, 1)[0]
    for x in range(1, 10000):
        y = -np.log(x) + offset + (np.sin(x) * 0.1)
        writer.add_scalar('y=-log(x) + c + 0.1sin(x)', y, x)
        writer.add_scalar('fake_metric', -y, x)

  def run(self):
    if self.log_dir is None:
      self.log_dir = f"lightning_logs/hello"
      self.generate_log(Path(f"{self.log_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))

    # below method produces a lot of log outout
    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--port', f"{self.port}", '--host', self.host, '--logdir', self.logdir])
    #url = tb.launch()

    subprocess.Popen(
      [
        "tensorboard",
        "--logdir",
        str(self.log_dir),
        "--host",
        self.host,
        "--port",
        str(self.port),
      ]
    )
    signal.pause()




