import os
import logging
import shlex
import string

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.components.python import TracerPythonScript
from lightning.utilities.state import AppState
from lightning.storage.path import Path

class ChdirPythonScript(TracerPythonScript):
  """chdir, delimited args and envs with python template substitution, then run a script
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run(self, root_dir:str, script_name:str, script_args: str = None, script_env: str = None):
    """chdir, delimited args and envs with python template substitution, then run a script
    
      Args:
        root_dir: ex: ./
        script_name: ex: scripts/ls.py
        script_args: ex:  --name=me --dir=./ 
        script_env: ex: user=xxx password=123
    """
    # save the CWD and restore later
    orig_cwd = os.getcwd()

    logging.debug(f"input root_dir={root_dir} script_name={script_name} script_args={script_args} script_env={script_env}")
    os.chdir(root_dir)
    logging.debug(f"cwd={os.getcwd()}")
    # args = []
    self.script_path = os.path.join(script_name)
    logging.debug(f"script_path={self.script_path}")

    # replace host and port
    host_port = dict(port=str(self.port), host=self.host)
    logging.debug(f"host_port={host_port}")
    # script_args expects [k,v,k,v]
    self.script_args=[]
    if script_args:
      script_args = string.Template(script_args).substitute(host_port)
      self.script_args = shlex.split(script_args)   # shlex locks on None
      logging.debug("script_args={self.script_args}")

    # env expects {k=v,k=v}
    self.env = {}
    if script_env:
      script_env = string.Template(script_env).substitute(host_port)
      for x in shlex.split(script_env):
          k, v = x.split("=", 2)
          self.env[k] = v
      logging.debug(f"env={self.env}")

    # run
    super().run()

    # restore the cwd
    os.chdir(orig_cwd)

