import os
import streamlit as st
from streamlit_ace import st_ace
import fire
import logging
from pathlib import Path
import yaml
from lightning.utilities.state import AppState

def hydra_config(language="yaml"):
  basename = st.session_state.hydra_config[0]
  filename = st.session_state.hydra_config[1]
  logging.debug(f"selectbox {st.session_state}")
  if basename in st.session_state:
    content_raw = st.session_state[basename]
  else:
    try:
      with open(filename) as input:
        content_raw = input.read()  
    except FileNotFoundError:
      st.error('File not found.')
    except Exception as err:  
      st.error(f"can't process select item. {err}")      
  content_new = st_ace(value=content_raw, language=language)
  if content_raw != content_new:
    st.write("content changed")
    st.session_state[basename] = content_new

def run(script_dir=None, script_name=None, config_dir=None, config_ext=None):
  """Display YAML file and return arguments and env variable fields

  :dir (str): dir name to pull the yaml file from
  :return (dict): {'script_arg':script_arg, 'script_env':script_env}
  """
  
  # one time activity to build out the list

  st_script_dir = st.text_input('Script Dir', value=script_dir or ".")
  st_script_name = st.text_input('Script Name', value=script_name or "run.py")

  st_config_dir = st.text_input('Dir of Hydra YAMLs', value=config_dir or ".")
  st_config_ext = st.text_input('YAML Extensions', value=config_ext or "*.yaml")

  options = []
  if not options:
    print("building options")
    for file in Path(st_config_dir).rglob(st_config_ext):
      basename = os.path.basename(file)
      options.append([basename, str(file)])

  st_script_arg = st.text_input('Script Args', placeholder="training.max_epochs=11")

  st_script_env = st.text_input('Script Env Vars')

  show_basename = lambda opt: opt[0]
  st.selectbox("override hydra config", options, key="hydra_config", format_func=show_basename)  

  st_submit_button = st.button('Train')

  options = hydra_config()

  if st_submit_button:
    return([st_script_arg, st_script_env, st.session_state])

if __name__ == '__main__':
  fire.Fire(run)
