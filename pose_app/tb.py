import os
import streamlit as st
from streamlit_ace import st_ace
from lightning.frontend import StreamlitFrontend
import logging
from lightning.components.python import TracerPythonScript
from lightning.utilities.state import AppState
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork

class CreateTensorboard(TracerPythonScript):
  def __init__(self, *args, **kwargs):
    super().__init__(script_path="run_tb.py",
    script_args=["--server=127.0.0.1", "--logdir=/home/jovyan/lightning-pose/outputs"],
    env=None,
    cloud_compute=None,
    blocking=False,
    run_once=True,
    port=6006,
    raise_exception=True,
)
