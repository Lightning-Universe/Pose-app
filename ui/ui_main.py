import streamlit as st
from hydra import compose, initialize
from omegaconf import OmegaConf

import ui_about
import ui_hydra
import ui_data
import ui_train
import ui_evaluate
import args_utils

name = "myapp"
menu = ["About", "Configure", "Data", "Train", "Evaluate", "Annotate"]

def read_hydra_config():
  initialize(config_path="configs", job_name="test_app")
  cfg = compose(config_name="config")
  print(OmegaConf.to_yaml(cfg))

def on_menu_click(*args, **kwargs):
  # do something useful
  # st.write(args)
  #st.write(kwargs)
  #st.write(st.session_state)
  st.session_state[name+"Menu"] = kwargs["key"]

def my_menu(key, title="About"):
  st.subheader(title)
  for m in menu:
    st.button(m,on_click=on_menu_click, key=key+m, kwargs={'key':key+m})

def run():
  if not(name+"Menu" in st.session_state):
    st.session_state[name+"Menu"] = ""

  with st.sidebar:
    my_menu(name)

  if st.session_state[name+"Menu"] == name+"Train":
    ui_train.run("../lightning_pose")
  elif st.session_state[name+"Menu"] == name+"Configure":
    ui_hydra.run("../lightning_pose")
  elif st.session_state[name+"Menu"] == name+"Data":
   ui_data.run("../lightning_pose")    
  elif st.session_state[name+"Menu"] == name+"Evaluate":
    ui_evaluate.run("../lightning_pose")    
  elif st.session_state[name+"Menu"] == name+"Annotate":
    st.write("Annotate stuff")
  else:
    ui_about.run()

if __name__ == "__main__":
  run()