import lightning_app as L
from lightning_app.storage.path import Path
from lightning.app.storage.drive import Drive
from lightning_app.structures import Dict, List
from lightning_app.utilities.app_helpers import _collect_child_process_pids

import os
import subprocess
import shlex
from string import Template
import signal
import time

def args_to_dict(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  script_args_dict = {}
  for x in shlex.split(script_args, posix=False):
    try:
      k,v = x.split("=",1)
    except:
      k=x
      v=None
    script_args_dict[k] = v
  return(script_args_dict) 

def add_to_system_env(env_key='env', **kwargs) -> dict:
  """add env to the current system env"""
  new_env = None
  if env_key in kwargs: 
    env = kwargs[env_key]
    if isinstance(env,str):
      env = args_to_dict(env)  
    if not(env is None) and not(env == {}):
      new_env = os.environ.copy()
      new_env.update(env)
  return(new_env)

class LitBashWork(L.LightningWork):
  def __init__(self, *args, 
    wait_seconds_after_run = 10,
    drive_name = "lit://lpa",
    **kwargs):
    super().__init__(*args, **kwargs,
      # required to to grab self.host and self.port in the cloud.  
      # otherwise, the values flips from 127.0.0.1 to 0.0.0.0 causing two runs
      # host='0.0.0.0',  
    )
    self.wait_seconds_after_run = wait_seconds_after_run
    self.drive_lpa = Drive(drive_name)

    self.pid = None
    self.exit_code = None
    self.stdout = []
    self.inputs = None
    self.outputs = None
    self.args = ""


  def reset_last_args(self) -> str:
    self.args = ""

  def last_args(self) -> str:
    return(self.args)

  def last_stdout(self):
    return(self.stdout)

  def on_before_run(self):
    """Called before the python script is executed."""

  def on_after_run(self):
      """Called after the python script is executed. Wrap outputs in Path so they will be available"""

  def get_from_drive(self,inputs):
    for i in inputs:
      print(f"drive get {i}")
      try:                     # file may not be ready 
        self.drive_lpa.get(i)  # Transfer the file from this drive to the local filesystem.
      except:
        pass  
      os.system(f"find {i} -print")

  def put_to_drive(self,outputs):
    for o in outputs:
      print(f"drive put {o}")
      # make sure dir end with / so that put works correctly
      if os.path.isdir(o):
        o = os.path.join(o,"")
      os.system(f"find {o} -print")
      self.drive_lpa.put(o)  

  def popen_wait(self, cmd, save_stdout, exception_on_error, **kwargs):
    with subprocess.Popen(
      cmd, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.STDOUT, 
      bufsize=0, 
      close_fds=True, 
      shell=True, 
      executable='/bin/bash',
      **kwargs
    ) as proc:
        self.pid = proc.pid
        if proc.stdout:
            with proc.stdout:
                for line in iter(proc.stdout.readline, b""):
                    #logger.info("%s", line.decode().rstrip())
                    line = line.decode().rstrip() 
                    print(line)
                    if save_stdout:
                      self.stdout.append(line)
    if exception_on_error and self.exit_code != 0:
      raise Exception(self.exit_code)  

  def popen_nowait(self, cmd, **kwargs):
    proc = subprocess.Popen(
      cmd, 
      shell=True, 
      executable='/bin/bash',
      close_fds=True,
      **kwargs
    )
    self.pid = proc.pid

  def subprocess_call(self, cmd, save_stdout=True, exception_on_error=False, venv_name = "", wait_for_exit=True, **kwargs):
    """run the command"""
    cmd = cmd.format(host=self.host,port=self.port) # replace host and port
    cmd = ' '.join(shlex.split(cmd))                # convert multiline to a single line
    print(cmd, kwargs)
    kwargs['env'] = add_to_system_env(**kwargs)
    pwd = os.path.abspath(os.getcwd())
    if venv_name:
      cmd = f"source ~/{venv_name}/bin/activate; which python; {cmd}; deactivate"
      
    if wait_for_exit:
      print("wait popen")
      self.popen_wait(cmd, save_stdout=save_stdout, exception_on_error=exception_on_error, **kwargs)
      print("wait completed",cmd)
    else:
      print("no wait popen")
      self.popen_nowait(cmd, **kwargs)
      print("no wait completed",cmd)

  def run(self, args, 
    venv_name="",
    save_stdout=True,
    wait_for_exit=True, 
    input_output_only = False, 
    inputs=[], outputs=[], 
    **kwargs):

    print(args, kwargs)
    
    # pre processing
    self.on_before_run()    
    self.get_from_drive(inputs)
    self.args = args
    self.stdout = []

    # run the command
    if not(input_output_only):
      self.subprocess_call(
        cmd=args, venv_name = venv_name, save_stdout=save_stdout, wait_for_exit=wait_for_exit, **kwargs)

    # post processing
    self.put_to_drive(outputs) 
    # give time for REDIS to catch up and propagate self.stdout back to flow
    if save_stdout: 
      time.sleep(self.wait_seconds_after_run) 
    self.on_after_run()

  def on_exit(self):
      for child_pid in _collect_child_process_pids(os.getpid()):
          os.kill(child_pid, signal.SIGTERM)
