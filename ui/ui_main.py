import streamlit as st

import ui_about
import ui_hydra
import ui_data
import ui_train

name = "myapp"
menu = ["About", "Configure", "Data", "Train", "Evaluate", "Annotate"]

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
    st.write("Evaluate stuff")
  elif st.session_state[name+"Menu"] == name+"Annotate":
    st.write("Annotate stuff")
  else:
    ui_about.run()

if __name__ == "__main__":
  run()