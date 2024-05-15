import os
import zipfile
from io import BytesIO

import streamlit as st
from lightning.app import LightningFlow
from lightning.app.utilities.state import AppState

from lightning_pose_app import MODELS_DIR, MODEL_VIDEO_PREDS_INFER_DIR
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


def list_labeled_mp4_files(model_dir, selected_model):
    """List labeled MP4 files in a directory."""
    labeled_mp4_files = []
    if selected_model:
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


def list_results_files(video_name, model_dir, selected_model):
    """List results files for a given video."""
    results_files = {}
    results_dir = os.path.join(model_dir, selected_model, MODEL_VIDEO_PREDS_INFER_DIR)
    for file_name in os.listdir(results_dir):
        if video_name in file_name and file_name.endswith(".csv"):
            full_path = os.path.join(results_dir, file_name)
            results_files[file_name] = full_path
    return results_files


def _render_streamlit_fn(state: AppState):

    # don't proceed if no project has been selected
    proj_dir = state.proj_dir
    if proj_dir is None:
        st.write("Must set project directory")
        return

    model_dir = abspath(os.path.join(proj_dir, MODELS_DIR))

    # Streamlit UI
    st.header("Visualize Model Predictions")

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
        custom_css = """
        <style>
        video {
            width: 100% !important;
            height: 50% !important;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        st.video(video_bytes)
        st.caption(
            "To download the video, click the three dots on the right side and select Download"
        )
        # Add a section to download results files
        st.header("Download Results Files")
        video_name = extract_video_name(selected_video)
        video_name_org = video_name.replace('.labeled', '')
        results_files = list_results_files(video_name_org, model_dir, selected_model)

        if results_files:
            selected_result_files = st.multiselect(
                "Select results files to download", list(results_files.keys())
            )
            if selected_result_files:
                if len(selected_result_files) == 1:
                    selected_result_file = results_files[selected_result_files[0]]
                    new_file_name = f"{selected_model}_{video_name_org}_{selected_result_files[0]}"
                    with open(selected_result_file, "rb") as file:
                        st.download_button(
                            label="Download File",
                            data=file,
                            file_name=new_file_name
                        )
                else:
                    st.warning(
                        "If you select more than one file, they will be downloaded "
                        "together as a ZIP folder"
                    )
                    # Create a zip file
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for result_file_name in selected_result_files:
                            result_file_path = results_files[result_file_name]
                            new_file_name = f"{selected_model}_{video_name_org}_{result_file_name}"
                            with open(result_file_path, "rb") as file:
                                zip_file.writestr(new_file_name, file.read())
                    zip_buffer.seek(0)
                    zip_file_name = f"results_{selected_model}_{video_name_org}.zip"
                    st.download_button(
                        label="Download Files",
                        data=zip_buffer,
                        file_name=zip_file_name,
                        mime="application/zip"
                    )
        else:
            st.write("No results files available for this video.")
    else:
        st.write("No video to preview")
