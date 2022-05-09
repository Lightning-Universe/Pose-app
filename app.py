import os

from lightning.frontend import StreamlitFrontend
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.components.python import TracerPythonScript
from lightning.storage.path import Path
import logging

from lightning.components.python import TracerPythonScript
import streamlit as st
from lightning.utilities.state import AppState

script_fo = TracerPythonScript(
  script_path = "run_fo.py",
  script_args = ["--address","0.0.0.0"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  exposed_ports = {"server": 5151},
  raise_exception = True,
  )

script_tb = TracerPythonScript(
  script_path = "run_tb.py",
  script_args = ["--server","0.0.0.0"],
  env = None,
  cloud_compute = None,
  blocking = False,
  run_once = True,
  exposed_ports = {"server": 6006},
  raise_exception = True,
  )

script_train = TracerPythonScript(
  script_path = "scripts/train_hydra.py",
  script_args = None,
  env = None,
  cloud_compute = None,
  blocking = True,
  run_once = True,
  exposed_ports = None,
  raise_exception = True,
  )


class ScriptUI(LightningFlow):
    def __init__(self):
        super().__init__()
        self.script_arg = None  # must match state.script_arg in _render_streamlit_fn
        self.script_env = None  # must exist state.script_env in _render_streamlit_fn

    def run(self, destination_dir: str):
        self.destination_dir = destination_dir

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)

def file_selector(form, dir='.'):
  filenames = os.listdir(dir)
  selected_filename = form.selectbox('Select a file', filenames)
  return os.path.join(dir, selected_filename)

def _render_streamlit_fn(state: AppState, dir="./"):
  """Display YAML file and return arguments and env variable fields

  :dir (str): dir name to pull the yaml file from
  :return (dict): {'script_arg':script_arg, 'script_env':script_env}
  """
  form = st

  script_arg = form.text_input('Script Args',placeholder="training.max_epochs=11")
  script_env = form.text_input('Script Env Vars')

  filename = file_selector(form=form, dir=dir)
  form.write('You selected `%s`' % filename)

  submit_button = form.button('Train')
  submit_button = form.button('Config')

  try:
    with open(filename) as input:
      content_raw = input.read()
      # view https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py 
      content_new = st_ace(value=content_raw, language="yaml")
  except FileNotFoundError:
    form.error('File not found.')
  except Exception:  
    form.error("can't process select item.")

  if submit_button:
    state.script_arg = script_arg
    state.script_env = script_env
  
class App(LightningFlow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.script_fo = script_fo
    self.script_tb = script_tb
    self.script_ui = ScriptUI()
    self.script_train = script_train

  def run(self):
    self.script_fo.run()
    self.script_tb.run()
    if self.script_ui.script_arg is not None:
      print("self.script_train.run()")
      self.script_train.script_args = self.script_ui.script_arg
      self.script_train.script_env = self.script_ui.script_env
      self.script_train.run()

  def configure_layout(self):
    tab1 = { "name": "Train Lightning Pose", "content": self.script_ui }

    if self.script_fo.has_started:
      tab2 = {"name": "Fiftyone", "content": self.script_fo.exposed_url('server')}
    else:
      tab2 = {"name": "Fiftyone", "content": ""}

    if self.script_tb.has_started:
      tab3 = {"name": "Tensorboard", "content": self.script_tb.exposed_url('server')}
    else:
      tab3 = {"name": "Tensorboard", "content": ""}

    return[tab1, tab2, tab3]  

app = LightningApp(App())