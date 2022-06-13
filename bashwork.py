import lightning as L
from lightning.utilities.network import find_free_network_port

import subprocess
import shlex
from string import Template

class LitBashWork(L.LightningWork):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs,
      host='0.0.0.0', # required to to grab self.host and self.port in the cloud.  otherwise, grab then in run
    )
    self.ports = {}

  def str_to_dict(self, script_args:str) -> dict:
    """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
    return(dict(x.split("=",maxsplit=1) for x in shlex.split(script_args)))

  def get_port(self,key):  
    """init class can be called multiple times.  the ports should be set as JIT and not a part of the init""" 
    if not(key in self.ports):
      self.ports[key] = find_free_network_port()
    return(self.ports[key])  

  def subprocess_popen(self, args, **kwargs):
    print(f"Running {args}")
    po = subprocess.Popen(shlex.split(args), **kwargs) 
    return(po)    
  def run(self, args, **kwargs):
    print(args, kwargs)
    if 'env' in kwargs and isinstance(kwargs['env'],str):
      if kwargs['env'] == "":
        kwargs['env'] = None
      else:  
        kwargs['env'] = self.str_to_dict(kwargs['env'])
    print(args, kwargs)
    po = self.subprocess_popen(args, **kwargs)

