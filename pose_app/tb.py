import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import signal
from datetime import datetime
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.storage.path import Path


class RunTensorboard(LightningWork):
  def __init__(self, logdir:str = None):
    super().__init__()
    self.logdir = logdir

  def generate_log(self, dataset_name):
    # fake tensorboard logs (fake loss)
    writer = SummaryWriter(log_dir=dataset_name)
    offset = np.random.uniform(0, 5, 1)[0]
    for x in range(1, 10000):
        y = -np.log(x) + offset + (np.sin(x) * 0.1)
        writer.add_scalar('y=-log(x) + c + 0.1sin(x)', y, x)
        writer.add_scalar('fake_metric', -y, x)

  def run(self):
    if self.logdir is None:
      self.logdir = f"lightning_logs/hello"
      self.generate_log(Path(f"{self.logdir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))

    tb = program.TensorBoard()

    tb.configure(argv=[None, '--port', f"{self.port}", '--host', self.host, '--logdir', self.logdir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    signal.pause()




class CreateTensorboard(TracerPythonScript):
  def __init__(self, *args, **kwargs):
    super().__init__(script_path="run_tb.py",
    script_args=["--server=127.0.0.1", "--logdir=./outputs"],
    env=None,
    cloud_compute=None,
    blocking=False,
    run_once=True,
    port=6006,
    raise_exception=True,
)
