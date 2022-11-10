import streamlit as st
import sh
import os

import args_utils

# NOTE: AttributeError: 'Path' object has no attribute '_origin'
#from lightning.storage.path import Path
from pathlib import Path


def run(root_dir=".",sub_dir=".",dir_name="unlabeled_videos"):
  dirs = args_utils.get_dir_of_dir(root_dir=root_dir, sub_dir=sub_dir,include=dir_name)
  data_dir = st.selectbox("data dir", options=dirs)

  if data_dir:
    dir=os.path.join(os.path.expanduser(root_dir),sub_dir,data_dir)
    options = []
    options_show_basename = lambda opt: opt["rel_path"]
    for file in Path(dir).rglob("*.mp4"):
      abs_dir = os.path.dirname(file)
      rel_dir = os.path.relpath(os.path.dirname(file),root_dir)
      rel_path = os.path.relpath(file, root_dir)
      options.append({"rel_dir":rel_dir, "abs_dir":abs_dir, "root_dir":root_dir, "rel_path":rel_path, "abs_path":str(file)})
    video_file = st.selectbox("unlabeled video",options,format_func=options_show_basename)

    if video_file:
      st.video(video_file["abs_path"], format="video/mp4", start_time=0)


if __name__ == "__main__":
  run(root_dir="../lightning_pose")
