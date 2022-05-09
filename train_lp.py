import fire
import subprocess

subprocess.call("test1.py", shell=True)

def run(*args, max_epochs=11,**kwargs):
  assert max_epochs >= 11
  subprocess.call("scripts/train_hydra.py *args **kwargs")

  