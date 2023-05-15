"""UI for training models."""

from datetime import datetime
from lightning import LightningFlow
from lightning.app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time

from lightning_pose_app.utilities import StreamlitFrontend

st.set_page_config(layout="wide")


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
        self.count = 0

        # save parameters for later run
        self.proj_dir = None

        self.n_labeled_frames = None  # set externally
        self.n_total_frames = None  # set externally

        # updated externally by top-level flow
        self.trained_models = []
        self.curr_epoch = 0

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
        self.st_semi_losses = None
        self.st_datetimes = None
        self.st_train_complete_flag = None

        # output from the UI (infer)
        self.st_inference_model = None
        self.st_inference_videos = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

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

    # constantly refresh so that:
    # - labeled frames are updated
    # - training progress is updated
    if (state.n_labeled_frames != state.n_total_frames) \
            or state.run_script_train:
        st_autorefresh(interval=2000, key="refresh_train_ui")

    with train_tab:

        st.markdown(
            """
            ## Train models
            
            If you have already trained models, you will be able to select those models for further 
            analysis in the following tabs
            (see the 'Existing Models' drop-down menu to see a list of already trained models).
            
            """
        )

        # TODO: update with st_radial
        st.write(f"Note: you have labeled {state.n_labeled_frames} / {state.n_total_frames} frames")

        st.selectbox("Existing models", sorted(state.trained_models, reverse=True))

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

        # give user training updates
        if state.run_script_train:
            if state.st_train_super:
                p = state.st_max_epochs if state.st_train_complete_flag["super"] else state.curr_epoch
                st.progress( float(p) / float(state.st_max_epochs), "Supervised progress")
            if state.st_train_semisuper:
                p = state.st_max_epochs if state.st_train_complete_flag["semisuper"] else state.curr_epoch
                st.progress( float(p) / float(state.st_max_epochs), "Semi-supervised progress")

        # Lightning way of returning the parameters
        if st_submit_button_train:

            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) and st_train_semisuper:
                st.warning("must select at least one semi-supervised loss if training that model")
                st_submit_button_train = False

        if st_submit_button_train:

            # save streamlit options to flow object
            state.st_max_epochs = int(st_max_epochs)
            state.st_train_super = st_train_super
            state.st_train_semisuper = st_train_semisuper
            state.st_loss_pcamv = st_loss_pcamv
            state.st_loss_pcasv = st_loss_pcasv
            state.st_loss_temp = st_loss_temp

            # construct semi-supervised loss list
            semi_losses = []
            if st_loss_pcamv:
                semi_losses.append("pca_multiview")
            if st_loss_pcasv:
                semi_losses.append("pca_singleview")
            if st_loss_temp:
                semi_losses.append("temporal")
            state.st_semi_losses = semi_losses

            # set model times
            st_datetimes = {}
            for i in range(2):
                dtime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
                if i == 0:  # supervised model
                    st_datetimes["super"] = dtime
                    time.sleep(1)  # allow date/time to update
                if i == 1:  # semi-supervised model
                    st_datetimes["semisuper"] = dtime

            # NOTE: cannot set these dicts entry-by-entry in the above loop, o/w don't get set?
            state.st_datetimes = st_datetimes
            state.st_train_complete_flag = {"super": False, "semisuper": False}
            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")

    with infer_tab:

        st.markdown(
            """
            ## Run inference
            Upload new videos and run inference with a trained model.

            """
        )

        model_dir = st.selectbox(
            "Choose model to run inference", sorted(state.trained_models, reverse=True))

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
            st.warning("waiting for existing inference to finish")

        # Lightning way of returning the parameters
        if st_submit_button_infer:
            state.st_inference_model = model_dir
            state.st_inference_videos = st_videos
            st.text("Request submitted!")
            state.run_script_infer = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_infer_ui_submitted")
