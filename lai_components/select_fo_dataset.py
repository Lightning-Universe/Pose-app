import os
import lightning as L
import streamlit as st
import logging

import fiftyone as fo
import fiftyone.zoo as foz

class RunFiftyone(L.LightningWork):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run(self, dataset_name = None, use_quickstart = False, remote = False, address=None, port=None):
    try:
      dataset = fo.load_dataset(dataset_name)
    except:
      dataset = None  
    # remote = False is a must for Lightning.  otherwise, the UI will not show up
    session = fo.launch_app(dataset, remote = remote, address=address or self.host, port=port or self.port)
    session.wait()

class SelectDatasetUI(L.LightningFlow):
  def __init__(self, *args, dataset_names = None, **kwargs):
    super().__init__(*args, **kwargs)
    if dataset_names is None:
      dataset_names = fo.list_datasets()
    self.dataset_names = dataset_names  
    self.st_submit = False
    self.st_selectbox = None

  def set_dateset_names(self, dateset_names = None):
    if dataset_names is None:
      dataset_names = fo.list_datasets()
    self.dataset_names = dataset_names 

  def configure_layout(self):
    return L.frontend.StreamlitFrontend(render_fn=render_fn)

def render_fn(app_state):
  st_selectbox = st.selectbox("Select Dataset", app_state.dataset_names or [], key="dataset_names")  
  st_submit = st.button("Submit")
  if st_submit:
    app_state.st_submit = st_submit
    app_state.st_selectbox = st_selectbox