import os
import streamlit as st
from streamlit_ace import st_ace
import fire
import logging
from pathlib import Path
import yaml

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

def run(dir='.', filter='*.yaml',options=[]):
  """Display YAML file and return arguments and env variable fields

  :dir (str): dir name to pull the yaml file from
  :return (dict): {'script_arg':script_arg, 'script_env':script_env}
  """
  
  # one time activity to build out the list
  if not options:
    print("building options")
    for file in Path(dir).rglob(filter):
      basename = os.path.basename(file)
      options.append([basename, str(file)])

  script_arg = st.text_input('Script Args', placeholder="training.max_epochs=11")

  script_env = st.text_input('Script Env Vars')

  show_basename = lambda opt: opt[0]
  st.selectbox("override hydra config", options, key="hydra_config", format_func=show_basename)  

  options = hydra_config()

if __name__ == '__main__':
  fire.Fire(run)
