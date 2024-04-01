from lightning.app import CloudCompute, LightningFlow
from lightning.app.utilities.state import AppState
import os
import streamlit as st

from lightning_pose_app import MODELS_DIR
from lightning_pose_app.utilities import StreamlitFrontend, abspath


class StreamlitVideoViewer(LightningFlow):
    """UI to run Streamlit video viewer."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # params updated externally by top-level flow
        self.proj_dir = None

    def run(self, action, **kwargs):
        pass

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


@st.cache_resource
def get_models(model_dir):

    trained_models = []
    # this returns a list of model training days
    dirs_day = os.listdir(model_dir)
    # loop over days and find HH-MM-SS
    for dir_day in dirs_day:
        fullpath1 = os.path.join(model_dir, dir_day)
        dirs_time = os.listdir(fullpath1)
        for dir_time in dirs_time:
            fullpath2 = os.path.join(fullpath1, dir_time)
            trained_models.append('/'.join(fullpath2.split('/')[-2:]))
    return trained_models


@st.cache_resource
def list_labeled_mp4_files(model_dir, selected_model):
    """List labeled MP4 files in a directory."""
    labeled_mp4_files = []
    model_path = os.path.join(model_dir, selected_model)
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".labeled.mp4"):
                labeled_mp4_files.append(os.path.join(root, file))
    return labeled_mp4_files


def extract_video_name(full_path):
    """Extract the video name from a full file path."""
    base_filename = os.path.basename(full_path)  # Get the base filename
    video_name, _ = os.path.splitext(base_filename)  # Split filename and extension
    return video_name


def _render_streamlit_fn(state: AppState):

    # don't proceed if no project has been selected
    proj_dir = state.proj_dir
    if proj_dir is None:
        st.write("Must set project directory")
        return

    model_dir = abspath(os.path.join(proj_dir, MODELS_DIR))

    # Streamlit UI
    st.header("Visualise Model Predictions")

    with st.sidebar:

        st.write("Select the model and then the associated video you wish to inspect")

        trained_models = get_models(model_dir)  # create a list of all models
        selected_model = st.selectbox("Step 1: Select a model", trained_models)

        st.divider()

        # Labeled videos
        mp4_files = list_labeled_mp4_files(model_dir, selected_model)
        selected_video = st.selectbox(
            "Step 2: Select a video", mp4_files, format_func=extract_video_name,
        )

    if selected_video:
        # read and show the predictions labeled video
        video_file = open(selected_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
    else:
        st.write("No video to preview")
