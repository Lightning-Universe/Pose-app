import streamlit as st
import sh
import args_utils

def run(root_dir="."):
  options = args_utils.get_dir_of_dir(root_dir=root_dir, include="tb_logs")
  st.selectbox("train outputs", options=options)
if __name__ == "__main__":
  run(root_dir="../lightning_pose")