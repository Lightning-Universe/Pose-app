import fire
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import signal

def generate_log(dataset_name="lightning_logs/hello"):
  # fake tensorboard logs (fake loss)
  writer = SummaryWriter(log_dir=dataset_name)
  offset = np.random.uniform(0, 5, 1)[0]
  for x in range(1, 10000):
      y = -np.log(x) + offset + (np.sin(x) * 0.1)
      writer.add_scalar('y=-log(x) + c + 0.1sin(x)', y, x)
      writer.add_scalar('fake_metric', -y, x)

if __name__ == '__main__':
  fire.Fire(generate_log)