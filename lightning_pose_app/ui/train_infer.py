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

        # for controlling messages to user
        self.submit_count_train = 0

        # for controlling when models are broadcast to other flows/workers
        self.count = 0

        # save parameters for later run
        self.proj_dir = None

        self.n_labeled_frames = None  # set externally
        self.n_total_frames = None  # set externally

        # updated externally by top-level flow
        self.trained_models = []
        self.progress = 0

        # output from the UI (train; all will be dicts with keys=models, except st_max_epochs)
        self.st_max_epochs = None
        self.st_train_status = {}  # 'none' | 'initialized' | 'active' | 'complete'
        self.st_losses = {}
        self.st_datetimes = {}

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

        # add a sidebar to show the labeling progress
        # Calculate percentage of frames labeled
        labeling_progress = state.n_labeled_frames / state.n_total_frames
        st.sidebar.markdown('### Labeling Progress')
        st.sidebar.progress(labeling_progress)
        st.sidebar.write(f"You have labeled {state.n_labeled_frames} out of {state.n_total_frames} frames.")

        st.sidebar.markdown("""### Existing models""")
        st.sidebar.selectbox("Browse", sorted(state.trained_models, reverse=True))
        st.sidebar.write("Proceed to next tabs to analyze your previously trained models.")

        st.header("Train Networks")

        st.markdown(
            """
            #### Defaults
            """
        )
        expander = st.expander("Change Defaults")

        # max epochs
        st_max_epochs = expander.text_input(
            "Max training epochs (supervised and semi-supervised)",
            value=100,
        )

        # unsupervised losses (semi-supervised only)
        expander.write("Select losses for semi-supervised model")
        st_loss_pcamv = expander.checkbox("PCA Multiview", value=True)
        st_loss_pcasv = expander.checkbox("PCA Singleview", value=True)
        st_loss_temp = expander.checkbox("Temporal", value=True)

        st.markdown(
            """
            #### Select models to train
            """
        )
        st_train_super = st.checkbox("Supervised", value=True)
        st_train_semisuper = st.checkbox("Semi-supervised", value=True)

        st_submit_button_train = st.button("Train models", disabled=state.run_script_train)

        # give user training updates
        if state.run_script_train:
            for m in ["super", "semisuper"]:
                if m in state.st_train_status.keys() and state.st_train_status[m] != "none":
                    status = state.st_train_status[m]
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        p = state.progress
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)
                    st.progress(p / 100.0, f"{m} progress ({status})")

        if st_submit_button_train:
            if (st_loss_pcamv + st_loss_pcasv + st_loss_temp == 0) and st_train_semisuper:
                st.warning("Must select at least one semi-supervised loss if training that model")
                st_submit_button_train = False

        if state.submit_count_train > 0 \
                and not state.run_script_train \
                and not st_submit_button_train:
            proceed_str = "Training complete; see diagnostics in the following tabs."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

        if st_submit_button_train:

            # save streamlit options to flow object
            state.submit_count_train += 1
            state.st_max_epochs = int(st_max_epochs)
            state.st_train_status = {
                "super": "initialized" if st_train_super else "none", 
                "semisuper": "initialized" if st_train_semisuper else "none",
            }

            # construct semi-supervised loss list
            semi_losses = []
            if st_loss_pcamv:
                semi_losses.append("pca_multiview")
            if st_loss_pcasv:
                semi_losses.append("pca_singleview")
            if st_loss_temp:
                semi_losses.append("temporal")
            state.st_losses = {"super": [], "semisuper": semi_losses}

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
            st.text("Model training launched!")
            state.run_script_train = True  # must the last to prevent race condition
            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_train_ui_submitted")

    with infer_tab:

        st.header("Predict on New Videos")

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
