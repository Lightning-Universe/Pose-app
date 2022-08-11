import os
import logging
import string
from datetime import datetime

import streamlit as st
from streamlit_ace import st_ace
import shlex

from lai_components.hydra_ui import hydra_config, get_hydra_config_name, get_hydra_dir_name
from lai_components.args_utils import args_to_dict, dict_to_args
from lai_components.vsc_streamlit import StreamlitFrontend

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning_app.components.python import TracerPythonScript
from lightning_app.utilities.state import AppState
from lightning_app.storage.path import Path


class FoRunUI(LightningFlow):
    """UI to run Fiftyone."""

    def __init__(
            self,
            *args,
            script_dir,
            script_name,
            script_args,
            script_env,
            outputs_dir="outputs",
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script = False

        # save parameters for later run
        self.script_dir = script_dir
        self.script_name = script_name
        self.script_args = script_args
        self.script_env = script_env
        self.outputs_dir = outputs_dir

        # FO list (updated externally by top-level flow)
        self.fo_datasets = []
        self.hydra_outputs = {}

        self.script_args_append = None

        # submit count
        self.submit_count = 0

        # output from the UI
        self.st_model_display_names = None
        self.st_submit = False
        self.st_script_args = None
        self.st_dataset_name = None
        self.st_model_dirs = None
        self.st_hydra_config_name = None
        self.st_hydra_config_dir = None

        # copy over for now, we can add these to the UI later if we want
        self.st_script_dir = script_dir
        self.st_script_name = script_name
        self.st_script_env = script_env
        self.st_outputs_dir = outputs_dir

    def set_fo_dataset(self, names):
        self.fo_datasets = names

    def add_fo_dataset(self, name):
        self.fo_datasets.append(name)

    def set_hydra_outputs(self, names: dict):
        self.hydra_outputs.update(names)

    def add_hydra_output(self, name: str):
        self.hydra_outputs.update(name)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def set_script_args(model_dirs: [str], script_args: str):

    script_args_dict = args_to_dict(script_args)

    # enrich the args
    # eval.video_file_to_plot="</ABSOLUTE/PATH/TO/VIDEO.mp4>" \

    # eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>","</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
    # eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"]
    # eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]

    if model_dirs:
        path_list = ','.join([f"'{x}'" for x in model_dirs])
        script_args_dict["eval.hydra_paths"] = f"[{path_list}]"

    # these will be controlled by the runners. remove if set manually
    script_args_dict.pop('eval.fiftyone.address', None)
    script_args_dict.pop('eval.fiftyone.port', None)
    script_args_dict.pop('eval.fiftyone.launch_app_from_script', None)
    script_args_dict.pop('eval.fiftyone.dataset_to_create', None)
    script_args_dict.pop('eval.fiftyone.dataset_name', None)
    script_args_dict.pop('eval.fiftyone.model_display_names', None)

    return dict_to_args(script_args_dict), script_args_dict


def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset"""

    st.markdown(
        """
        ### Prepare diagnostics

        Choose a supervised model and a semi-supervised model for evaluation.

        If you just trained models these will be provided as defaults, but can be updated by using 
        the drop-down menus.

        Click 'Prepare' to begin preparation of the diagnostics. These diagnostics will be 
        displayed in the following three tabs:
        * View Preds: view model predictions and ground truth on all images using FiftyOne
        * Frame Diag: TODO
        * Video Diag: TODO

        """
    )

    st.markdown(
        """
        #### Select models
        """
    )

    # hard-code two models for now
    st_model_dirs = [None for _ in range(2)]
    st_model_display_names = [None for _ in range(2)]

    # select first model (supervised)
    tmp = st.selectbox(
        "Select Model 1",
        [k for k, v in sorted(state.hydra_outputs.items(), reverse=True)]
    )
    st_model_dirs[0] = tmp
    tmp = st.text_input("Display name for Model 1")
    st_model_display_names[0] = tmp

    # select second model (semi-supervised)
    options = [k for k, v in sorted(state.hydra_outputs.items(), reverse=True)]
    if st_model_dirs[0]:
        options.remove(st_model_dirs[0])

    tmp = st.selectbox("Select Model 2", options)
    st_model_dirs[1] = tmp
    tmp = st.text_input("Display name for Model 2")
    st_model_display_names[1] = tmp

    # dataset names
    existing_datasets = state.fo_datasets
    st.write(f"Existing Fifityone datasets:\n{', '.join(existing_datasets)}")
    st_dataset_name = st.text_input("Choose dataset name other than the above existing names")
    if st_dataset_name in existing_datasets:
        st.error(f"{st_dataset_name} exists. Please choose a new name.")
        st_dataset_name = None

    # parse
    state.script_args, script_args_dict = set_script_args(
        model_dirs=st_model_dirs, script_args=state.script_args)

    # build dataset
    st_submit_button = st.button(
        "Initialize diagnostics",
        disabled=True if ((st_dataset_name is None) or (st_dataset_name == "") or state.run_script)
        else False)
    if state.run_script:
        st.warning(f"waiting for existing dataset creation to finish")
    if st_model_dirs[0] is None or st_model_dirs[1] is None:
        st.warning(f"select at least one model to continue")
    if (st_dataset_name is None) or (st_dataset_name == ""):
        st.warning(f"enter a unique dataset name to continue")

    # Lightning way of returning the parameters
    if st_submit_button:

        state.submit_count += 1

        # save streamlit options to flow object
        state.st_dataset_name = st_dataset_name
        state.st_model_display_names = st_model_display_names
        state.st_model_dirs = st_model_dirs
        state.st_script_args = state.script_args
        state.st_hydra_config_name = get_hydra_config_name()
        state.st_hydra_config_dir = get_hydra_dir_name()

        # set key-value pairs that will be used as script args
        script_args_append = f"eval.fiftyone.dataset_name={st_dataset_name}"
        script_args_append += " " + "eval.fiftyone.model_display_names=[%s]" % \
                              ','.join([f"'{x}'" for x in st_model_display_names])
        script_args_append += " " + "eval.fiftyone.launch_app_from_script=False"
        script_args_append += " " + state.st_hydra_config_name
        script_args_append += " " + state.st_hydra_config_dir
        state.script_args_append = script_args_append

        state.run_script = True  # must the last to prevent race condition
