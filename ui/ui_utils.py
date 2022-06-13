import shlex
import sh
import os
import sys

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

def args_to_dict_v2(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  return(dict(x.split("=",maxsplit=1) for x in shlex.split(script_args)))

  return(dict(x.split("=",maxsplit=1) for x in shlex.split(script_args)))

def dict_to_args(script_args_dict:dict) -> str:
  """convert dict {'A':1, 'B':2} to str A=1 B=2 to """
  script_args_array = []
  for k,v in script_args_dict.items():
    if v is None:
      script_args_array.append(f"{k}")
    else:  
      script_args_array.append(f"{k}={v}")
  # return as a text
  return(" \n".join(script_args_array)) 

def get_dir_of_dir(root_dir=".", sub_dir=".", include="tb_logs"):
  """return dirs that has tb_logs and maintain relative dir"""
  dir=os.path.join(os.path.expanduser(root_dir),sub_dir)
  options=[]
  try:
    # x.strip() removes \n at then end
    # os.path.dirname removes dir_name at the end
    # os.path.relpath removes root_dir, sub_dir in the beginning
    options = [os.path.relpath(os.path.dirname(x.strip()),dir) for x in sh.find(dir,"-type","d", "-name", include,)]
    options.sort(reverse=True)
  except:
    pass  
  return(options)

def get_dir_of_files(root_dir=".", sub_dir=".", include="predictions.csv"):
  """return dirs that has predictions.csv and maintain relative dir"""
  dir=os.path.join(os.path.expanduser(root_dir),sub_dir)
  options=[]
  try:
    # x.strip() removes \n at then end
    # os.path.dirname removes dir_name at the end
    # os.path.relpath removes root_dir, sub_dir in the beginning
    options = [os.path.relpath(os.path.dirname(x.strip()),dir) for x in sh.find(dir,"-type","f", "-name", include,)]
    options.sort(reverse=True)
  except:
    pass  
  return(options)

def splitall(path):
  allparts = []
  while 1:
    parts = os.path.split(path)
    if parts[0] == path:  # sentinel for absolute paths
      allparts.insert(0, parts[0])
      break
    elif parts[1] == path: # sentinel for relative paths
      allparts.insert(0, parts[1])
      break
    else:
      path = parts[0]
      allparts.insert(0, parts[1])
  return allparts