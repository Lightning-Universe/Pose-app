import os
import shlex


def args_to_dict(script_args:str) -> dict:
    """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
    script_args_dict = {}
    for x in shlex.split(script_args, posix=False):
        try:
            k, v = x.split("=", 1)
        except:
            k = x
            v = None
        script_args_dict[k] = v
    return script_args_dict


def dict_to_args(script_args_dict:dict) -> str:
    """convert dict {'A':1, 'B':2} to str A=1 B=2 to """
    script_args_array = []
    for k,v in script_args_dict.items():
        script_args_array.append(f"{k}={v}")
    # return as a text
    return " \n".join(script_args_array)


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
