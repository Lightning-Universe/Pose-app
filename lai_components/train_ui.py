"""Note: this replaces run_ui.py"""

from datetime import datetime
from lightning import LightningFlow
from lightning_app.utilities.state import AppState
import streamlit as st
import time

from lai_components.args_utils import args_to_dict, dict_to_args
from lai_components.hydra_ui import hydra_config, get_hydra_config_name, get_hydra_dir_name
from lai_components.vsc_streamlit import StreamlitFrontend


class TrainDemoUI(LightningFlow):
    """UI to enter training parameters for demo data."""

    def __init__(
            self,
            *args,
            script_dir,
            script_name,
            script_args,
            script_env,
            test_videos_dir="",
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
        self.test_videos_dir = test_videos_dir
        self.outputs_dir = outputs_dir

        # hydra outputs list (updated externally by top-level flow)
        self.hydra_outputs = {}

        # input to the UI
        self.max_epochs = 100
        self.train_frames = 1
        self.train_super = True
        self.train_semisuper = True
        self.loss_pcamv = True
        self.loss_pcasv = True
        self.loss_temp = True

        # output from the UI
        self.st_max_epochs = None
        self.st_train_frames = None
        self.st_train_super = None
        self.st_train_semisuper = None
        self.st_loss_pcamv = None
        self.st_loss_pcasv = None
        self.st_loss_temp = None
        self.st_script_args = None
        self.st_semi_losses = None

        # copy over for now, we can add these to the UI later if we want
        self.st_script_dir = script_dir
        self.st_script_name = script_name
        self.st_script_env = script_env
        self.st_outputs_dir = outputs_dir

    def set_hydra_outputs(self, names: dict):
        # this function is called by the top-level app when model training completes.
        self.hydra_outputs.update(names)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def set_script_args(script_args: str):

    script_args_dict = args_to_dict(script_args)
    run_date_time = datetime.today().strftime('%Y-%m-%d/%H-%M-%S')

    # only set if not alreay present
    if not ('hydra.run.dir' in script_args_dict):
        script_args_dict['hydra.run.dir'] = f"outputs/{run_date_time}"

    if not ('hydra.sweep.dir' in script_args_dict):
        script_args_dict['hydra.sweep.dir'] = f"outputs/multirun/{run_date_time}"

    # change back to array
    return dict_to_args(script_args_dict)


def _render_streamlit_fn(state: AppState):

    st.markdown(
        """
        ## Train models
        This tab presents options for training two models:
        * fully supervised model
        * semi-supervised model
        
        Default settings are provided, but can be updated by expanding the 'Change Defaults' 
        drop-down menu below.
         
        Click 'Train models' to launch the training job.
        
        If you have already trained models, you will be able to select those models for further 
        analysis in the next tab (see the 'Existing Models' drop-down menu to see a list of already 
        trained models).
        
        """
    )
    st.selectbox(
        "Existing Models",
        [k for k, v in sorted(state.hydra_outputs.items(), reverse=True)]
    )

    st.markdown(
        """
        #### Defaults
        """
    )
    expander = st.expander("Change Defaults")

    # max epochs
    st_max_epochs = expander.text_input(
        "Max training epochs (supervised and semi-supervised)",
        value=state.max_epochs,
    )

    # train frames
    st_train_frames = expander.text_input(
        "Training frames (enter '1' to keep all training frames)",
        value=state.train_frames,
    )

    # unsupervised losses (semi-supervised only)
    expander.write("Select losses for semi-supervised model")
    st_loss_pcamv = expander.checkbox("PCA Multiview", value=state.loss_pcamv)
    st_loss_pcasv = expander.checkbox("PCA Singleview", value=state.loss_pcasv)
    st_loss_temp = expander.checkbox("Temporal", value=state.loss_temp)

    st.markdown(
        """
        #### Select models to train
        """
    )
    st_train_super = st.checkbox("Supervised", value=state.train_super)
    st_train_semisuper = st.checkbox("Semi-supervised", value=state.train_semisuper)

    st_submit_button = st.button("Train models", disabled=True if state.run_script else False)
    if state.run_script:
        st.warning(f"waiting for existing training to finish")

    # Lightning way of returning the parameters
    if st_submit_button:

        if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) and st_train_semisuper:
            st.warning("must select at least one semi-supervised loss if training that model")
            st_submit_button = False

    if st_submit_button:

        # save streamlit options to flow object
        state.st_max_epochs = st_max_epochs
        state.st_train_frames = st_train_frames
        state.st_train_super = st_train_super
        state.st_train_semisuper = st_train_semisuper
        state.st_loss_pcamv = st_loss_pcamv
        state.st_loss_pcasv = st_loss_pcasv
        state.st_loss_temp = st_loss_temp

        # construct semi-supervised loss string
        semi_losses = "["
        if st_loss_pcamv:
            semi_losses += "pca_multiview,"
        if st_loss_pcasv:
            semi_losses += "pca_singleview,"
        if st_loss_temp:
            semi_losses += "temporal,"
        semi_losses = semi_losses[:-1] + "]"
        state.st_semi_losses = semi_losses

        # set key-value pairs that will be used as script args
        state.st_script_args = {}
        for i in range(2):
            tmp = set_script_args(state.script_args)  # sets date/time for output dir
            tmp += f" training.max_epochs={st_max_epochs}"
            tmp += f" training.train_frames={st_train_frames}"
            tmp += f" training.profiler=null"
            tmp += f" eval.predict_vids_after_training=true"
            tmp += f" eval.test_videos_directory={state.test_videos_dir}"
            if i == 0:
                # supervised model
                state.st_script_args["super"] = tmp + " 'model.losses_to_use=[]'"
            if i == 1:
                # semi-supervised model
                state.st_script_args["semisuper"] = tmp + f" 'model.losses_to_use={semi_losses}'"
            time.sleep(1)  # allow date/time to update

        state.run_script = True  # must the last to prevent race condition
