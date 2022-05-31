import os
import logging
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace

# NOTE: AttributeError: 'Path' object has no attribute '_origin'
#from lightning.storage.path import Path
from pathlib import Path

global key_hydra_config_name
global key_hydra_config_dir
global key_hydra_config_file
global key_hydra_config_file_edit

key_hydra_config_name="hydra_config_name" # "--config_name={st.state[key_hydra_config_name]}")
key_hydra_config_dir="hydra_config_dir"   # "--config-dir={st.state[key_hydra_config_dir]}")
key_hydra_config_file="hydra_config_file"
key_hydra_config_file_edit="hydra_config_file_edit"
key_hydra_run_out="hydra_run_out"


def hydra_config_dir_selected(context=st) -> str:
  try:
    config_dir = st.session_state[key_hydra_config_dir][0]
  except:
    config_dir = None
  return(config_dir)

def get_hydra_config():
  """return --config_name --config_dir "+hydra.run.out as dict"""
  ret = {}
  if key_hydra_config_name in st.session_state and not(st.session_state[key_hydra_config_name] is None) and st.session_state[key_hydra_config_name] != "":
    ret["--config_name"]="'%s'" % st.session_state[key_hydra_config_name]
  if not (hydra_config_dir_selected() is None):
    ret["--config_dir"]="'%s'" % hydra_config_dir_selected()
  if key_hydra_run_out in st.session_state and not(st.session_state[key_hydra_config_name] is None) and st.session_state[key_hydra_config_name] != "":
    ret["+hydra.run.out"]="'%s'" % st.session_state[key_hydra_run_out]
  return(ret)  

def get_hydra_config_name():
  if key_hydra_config_name in st.session_state and not(st.session_state[key_hydra_config_name] is None):
    return(f"--config_name={st.session_state[key_hydra_config_name]}")
  else:
    return("")  

def get_hydra_dir_name():
  if not (hydra_config_dir_selected() is None):
    return(f"--config_dir={hydra_config_dir_selected()}")
  else:
    return("")

def set_hydra_run_out(hydra_run_out=None, context=st):
  """set +hydra.run.out=outputs/YY-MM-DD/HH-MM-SS"""
  if hydra_run_out is None:
    if key_hydra_run_out in st.session_state:
      hydra_run_out = st.session_state[key_hydra_run_out]
    else:  
      hydra_run_out = datetime.today().strftime('outputs/%Y-%m-%d/%H-%M-%S')    
  x = context.text_input("hydra run out dir", value=hydra_run_out, placeholder="outputs/YY-MM-DD/HH-MM-SS", key=key_hydra_run_out)
  if x=="":
    context.error(f"hydra run out dir cannot be empty")

def set_hydra_config_name(config_name="config.yaml", context=st):
  """set --config_name=config.yaml"""
  x = context.text_input("hydra default config name", value=config_name, placeholder=config_name, key=key_hydra_config_name)
  if x=="":
    context.error(f"config name cannot be empty")

def set_hydra_config_dir(config_dir=".", context=st, root_dir="."):
  """set --config_dir=dir from a list of dir that has config.yaml"""

  config_name = st.session_state[key_hydra_config_name]

  # options = [ [dirname,full_path] ...]
  options_show_basename = lambda opt: opt[0]

  # options should have just .yaml
  options=[]  
  try:
    if not options:
      for file in Path(os.path.join(os.path.expanduser(root_dir),config_dir)).rglob(config_name):
        dirname = os.path.dirname(file)
        options.append([dirname, str(file)])
  except:
    pass      

  # show it 
  context.selectbox("hydra config dir", options, key=key_hydra_config_dir, format_func=options_show_basename)


def set_hydra_config_file(config_dir=None, config_ext="*.yaml", context=st):
  """select a file from a list of *.yaml files"""
  config_dir = hydra_config_dir_selected(context=context)

  # options should have just .yaml 
  options=[] 
  try:
    if not options:
      for file in Path(config_dir).rglob(config_ext):
        options.append(str(file))
  except:
    pass

  # NOTE: pasing array of array somtimes produces error when format_func=options_show_basename is used
  # don't use format_func here
  # ValueError: ['config.yaml', 'configs/rick-configs-1/config.yaml'] is not in iterable
  context.selectbox("override hydra config", options, key=key_hydra_config_file)

def edit_hydra_config_file(language="yaml", context=st):
  """edit a content of .yaml file"""    
  filename = st.session_state[key_hydra_config_file]
  if filename is None:
    return

  if not(key_hydra_config_file_edit in st.session_state):
    st.session_state[key_hydra_config_file_edit]={}

  content_changed = False
  if filename in st.session_state[key_hydra_config_file_edit]:
    content_raw = st.session_state[key_hydra_config_file_edit][filename]
    content_changed = True
  else:
    try:
      with open(filename) as input:
        content_raw = input.read()
    except FileNotFoundError:
      context.error("File not found.")
    except Exception as err:
      context.error(f"can't process select file. {err}")
      return
  content_new = st_ace(value=content_raw, language=language)
  if content_changed or content_raw != content_new:
    context.warning("content changed")
    if not(key_hydra_config_file_edit in st.session_state):
      st.session_state[key_hydra_config_file_edit]={}
    st.session_state[key_hydra_config_file_edit][filename] = content_new

def hydra_config(root_dir=".", hydra_run_out=None, config_name="config.yaml", config_dir=".", context=st):
  set_hydra_run_out(hydra_run_out=hydra_run_out, context=context)
  set_hydra_config_name(config_name=config_name, context=context)
  set_hydra_config_dir(root_dir=root_dir, config_dir=config_dir, context=context)
  set_hydra_config_file(context=context)
  edit_hydra_config_file(context=context)
  return(get_hydra_config())
  
# only for testing streamlit run hydra_ui.py
# check out the gallery at https://share.streamlit.io/okld/streamlit-gallery/main?p=ace-editor
if __name__ == "__main__":
  expander = st.expander("expand")
  hydra_config(root_dir="lightning-pose", context=expander)
