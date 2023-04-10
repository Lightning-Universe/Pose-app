"""UI for training models."""

from datetime import datetime
from lightning import LightningFlow
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time

from lightning_pose_app.utilities import StreamlitFrontend


st.set_page_config(layout="wide")
IS_RUNNING_IN_CLOUD = is_running_in_cloud()


class TrainUI(LightningFlow):
    """UI to enter training parameters for demo data."""

    def __init__(
        self,
        *args,
        drive_name,
        max_epochs=200,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.drive = Drive(drive_name)

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script_train = False
        # self.run_script_update_models = False
        self.run_script_infer = False

        # save parameters for later run
        self.proj_dir = None

        self.n_labeled_frames = None  # set externally
        self.n_total_frames = None  # set externally

        # updated externally by top-level flow
        self.trained_models = []

        # input to the UI (train)
        self.max_epochs = max_epochs
        self.train_super = True
        self.train_semisuper = True
        self.loss_pcamv = True
        self.loss_pcasv = True
        self.loss_temp = True

        # output from the UI (train)
        self.st_max_epochs = None
        self.st_train_super = None
        self.st_train_semisuper = None
        self.st_loss_pcamv = None
        self.st_loss_pcasv = None
        self.st_loss_temp = None
        self.st_script_args = None
        self.st_semi_losses = None
        self.st_datetimes = None
        self.st_train_complete_flag = None

        # output from the UI (infer)
        self.st_inference_model = None
        self.st_inference_videos = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

<<<<<<< main:lightning_pose_app/ui/train.py
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

    # constantly refresh so that labeled frames are updated
    if state.n_labeled_frames != state.n_total_frames:
        st_autorefresh(interval=2000, key="refresh_train_ui")

    # TODO: update with st_radial
    st.text(f"Note: you have labeled {state.n_labeled_frames} / {state.n_total_frames} frames")

    st.selectbox(
        "Existing Models", sorted(state.trained_models, reverse=True))

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
        st_script_args = {}
        st_datetimes = {}
        for i in range(2):
            tmp = ""
            tmp += f" training.max_epochs={st_max_epochs}"
            tmp += f" training.train_frames={st_train_frames}"
            tmp += f" training.profiler=null"
            tmp += f" training.log_every_n_steps=1"  # for debugging
            tmp += f" eval.predict_vids_after_training=true"
            dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
            if i == 0:
                # supervised model
                st_script_args["super"] = tmp + " 'model.losses_to_use=[]'"
                st_datetimes["super"] = dtime
                time.sleep(1)  # allow date/time to update
            if i == 1:
                # semi-supervised model
                st_script_args["semisuper"] = tmp + f" 'model.losses_to_use={semi_losses}'"
                st_datetimes["semisuper"] = dtime

        # NOTE: cannot set these dicts entry-by-entry in the above loop, o/w don't get set?
        state.st_script_args = st_script_args
        state.st_datetimes = st_datetimes
        state.st_train_complete_flag = {"super": False, "semisuper": False}
        st.text("Model training launched!")
        state.run_script = True  # must the last to prevent race condition
        # force rerun
        st_autorefresh(interval=2000, key="refresh_train_ui_submitted")
        # TODO: show training progress
=======
    # make a train tab and an inference tab
    train_tab, infer_tab = st.columns(2, gap="large")

    # add shadows around each column
    # box-shadow args: h-offset v-offset blur spread color
    st.markdown("""
        <style type="text/css">
        div[data-testid="column"] {
            box-shadow: 3px 3px 10px -1px rgb(0 0 0 / 20%);
            border-radius: 5px;
            padding: 2% 3% 3% 3%;
        }
        </style>
    """, unsafe_allow_html=True)

    with train_tab:

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
            analysis in the next tab 
            (see the 'Existing Models' drop-down menu to see a list of already trained models).
            
            """
        )

        # constantly refresh so that labeled frames are updated
        if state.n_labeled_frames != state.n_total_frames:
            st_autorefresh(interval=2000, key="refresh_train_ui")

        # TODO: update with st_radial
        st.text(f"Note: you have labeled {state.n_labeled_frames} / {state.n_total_frames} frames")

        st.selectbox(
            "Existing Models", [k for k, v in sorted(state.trained_models.items(), reverse=True)])

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

        st_submit_button_train = st.button("Train models", disabled=state.run_script_train)
        if state.run_script_train:
            st.warning("waiting for existing training to finish")

        # Lightning way of returning the parameters
        if st_submit_button_train:

            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) and st_train_semisuper:
                st.warning("must select at least one semi-supervised loss if training that model")
                st_submit_button_train = False

        if st_submit_button_train:

            # save streamlit options to flow object
            state.st_max_epochs = st_max_epochs
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
            st_script_args = {}
            st_datetimes = {}
            for i in range(2):
                tmp = ""
                tmp += f" training.max_epochs={st_max_epochs}"
                tmp += f" training.profiler=null"
                tmp += f" training.log_every_n_steps=1"  # for debugging
                tmp += f" eval.predict_vids_after_training=true"
                dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
                if i == 0:
                    # supervised model
                    st_script_args["super"] = tmp + " 'model.losses_to_use=[]'"
                    st_datetimes["super"] = dtime
                    time.sleep(1)  # allow date/time to update
                if i == 1:
                    # semi-supervised model
                    st_script_args["semisuper"] = tmp + f" 'model.losses_to_use={semi_losses}'"
                    st_datetimes["semisuper"] = dtime

            # NOTE: cannot set these dicts entry-by-entry in the above loop, o/w don't get set?
            state.st_script_args = st_script_args
            state.st_datetimes = st_datetimes
            state.st_train_complete_flag = {"super": False, "semisuper": False}
            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")
            # TODO: show training progress

    with infer_tab:

        st.markdown(
            """
            ## Run inference
            This tab allows you to upload new videos and run inference with a trained model.

            """
        )

        # st_submit_button_update = st.button(
        #     "Update models",
        #     disabled=state.run_script_update_models,
        # )
        # if state.run_script_update_models:
        #     st.warning(f"waiting for updates to finish")
        # st.text(state.run_script_update_models)
        # if st_submit_button_update:
        #     st.text("Request submitted!")
        #     state.run_script_update_models = True  # must the last to prevent race condition
        #     st.text(state.run_script_update_models)

        model_dir = st.selectbox(
            "Choose model to run inference",
            [k for k, v in sorted(state.trained_models.items(), reverse=True)]
        )

        # upload video files
        video_dir = os.path.join(state.proj_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        # initialize the file uploader
        uploaded_files = st.file_uploader("Choose video files", accept_multiple_files=True)

        # for each of the uploaded files
        st_videos = []
        for uploaded_file in uploaded_files:
            # read it
            bytes_data = uploaded_file.read()
            # name it
            filename = uploaded_file.name.replace(" ", "_")
            filepath = os.path.join(video_dir, filename)
            st_videos.append(filepath)
            # write the content of the file to the path
            with open(filepath, "wb") as f:
                f.write(bytes_data)
            # push the data to the Drive
            state.drive.put(filepath)

        st_submit_button_infer = st.button(
            "Run inference",
            disabled=len(st_videos) == 0 or state.run_script_infer,
        )
        if state.run_script_infer:
            st.warning(f"waiting for existing inference to finish")

        # Lightning way of returning the parameters
        if st_submit_button_infer:
            state.st_inference_model = model_dir
            state.st_inference_videos = st_videos
            st.text("Request submitted!")
            state.run_script_infer = True  # must the last to prevent race condition
