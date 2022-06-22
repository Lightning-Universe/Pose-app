import shlex

def args_to_dict_v2(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  script_args_dict = {}
  for x in shlex.split(script_args, posix=False):
    k,v = x.split("=",1)
    script_args_dict[k] = v
  return(script_args_dict) 

def args_to_dict(script_args:str) -> dict:
  """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
  return(dict(x.split("=",maxsplit=1) for x in shlex.split(script_args)))

def dict_to_args(script_args_dict:dict) -> str:
  """convert dict {'A':1, 'B':2} to str A=1 B=2 to """
  script_args_array = []
  for k,v in script_args_dict.items():
    script_args_array.append(f"{k}={v}")
  # return as a text
  return(" \n".join(script_args_array)) 

