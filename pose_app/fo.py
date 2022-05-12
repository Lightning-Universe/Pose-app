import os
import streamlit as st
from streamlit_ace import st_ace
from lightning.frontend import StreamlitFrontend
import logging
from lightning.components.python import TracerPythonScript
from lightning.utilities.state import AppState
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.storage.path import Path
import shlex

class CreateFiftyoneUI(LightningFlow):
    """UI to enter training parameters
    Input and output variables with streamlit must be pre decleared
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # input to UI
        self.script_dir = "/home/jovyan/lightning-pose"
        self.script_name = "scripts/create_fiftyone_dataset.py"

        self.config_dir = "/home/jovyan/lightning-pose/scripts/configs"
        self.config_ext = "*.yaml"        

        self.dataset_name = "pose_image"
        self.fiftyone_port = "5151"
#eval.fiftyone.dataset_name=$DATASET_NAME \
#eval.fiftyone.port=5151"
        self.script_args = """eval.fiftyone.dataset_to_create="images" \
eval.fiftyone.build_speed="fast" \
eval.hydra_paths=['/home/jovyan/lightning-pose'] \
eval.fiftyone.model_display_names=[f"{self.dataset_name}"] \
eval.fiftyone.launch_app_from_script=True"""
        # output from the UI
        self.st_submit = False
        self.st_script_dir = None
        self.st_script_name = None
        self.st_script_arg = None  # must match state.script_arg in _render_streamlit_fn
        self.st_script_env = None  # must exist state.script_env in _render_streamlit_fn
        self.st_dataset_name = None  # must exist state.script_env in _render_streamlit_fn
        self.st_fiftyone_port = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)

def hydra_config(language="yaml"):
    try:
      basename = st.session_state.hydra_config[0]
      filename = st.session_state.hydra_config[1]
    except:
      st.error("no files found")
    logging.debug(f"selectbox {st.session_state}")
    if basename in st.session_state:
        content_raw = st.session_state[basename]
    else:
        try:
            with open(filename) as input:
                content_raw = input.read()
        except FileNotFoundError:
            st.error("File not found.")
        except Exception as err:
            st.error(f"can't process select item. {err}")
    content_new = st_ace(value=content_raw, language=language)
    if content_raw != content_new:
        st.write("content changed")
        st.session_state[basename] = content_new

def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset
    """
    st_script_dir = st.text_area("Script Dir", value=state.script_dir or ".")
    st_script_name = st.text_input("Script Name", value=state.script_name or "run.py")

    st_config_dir = st.text_input("Dataset Name ", value=state.dataset_name or "DATASET_NAME")
    st_config_ext = st.text_input("Fifityone Port", value=state.fiftyone_port or "5151")

    st_script_arg = st.text_input("Script Args", value=state.script_args)
    st_script_env = st.text_input("Script Env Vars", placeholder="ABC=123 DEF=345")

    options = []
    if not options:
        print("building options")
        for file in Path(st_config_dir).rglob(st_config_ext):
            basename = os.path.basename(file)
            options.append([basename, str(file)])
    show_basename = lambda opt: opt[0]
    st.selectbox(
        "override hydra config", options, key="hydra_config", format_func=show_basename
    )

    st_submit_button = st.button("Create Dataset")

    options = hydra_config()

    # Lightning way of returning the parameters
    if st_submit_button:
        state.st_script_dir = st_script_dir
        state.st_script_name = st_script_name

        state.st_script_arg = st_script_arg
        state.st_script_env = st_script_env
        state.st_submit     = True  # must the last to prevent race condition



class CreateFiftyoneDataset(TracerPythonScript):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run(self, script_path: str, script_args: str = None, script_env: str = None):
      import os
      def getcwd():
        return os.path.dirname(os.path.dirname(script_path))
      print(script_path, os.getcwd())
      os.getcwd = getcwd
      print(os.getcwd())
      self.script_path = script_path
      self.script_args = shlex.split(script_args)
      self.env = {}
      for x in shlex.split(script_env):
          k, v = x.split("=", 2)
          self.env[k] = v
      print(f"{self.script_path} {self.script_args} {self.env}")
      super().run()


