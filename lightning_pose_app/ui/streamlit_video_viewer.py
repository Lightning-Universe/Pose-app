from lightning.app import CloudCompute, LightningFlow
from lightning.app.utilities.state import AppState
import os
import streamlit as st
import pandas as pd

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



## Add to all functions
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


# Function to list labeled MP4 files in a directory
def list_labeled_mp4_files(model_dir, selected_model):
    labeled_mp4_files = []
    model_path = os.path.join(model_dir, selected_model)
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith("_labeled.mp4"):
                labeled_mp4_files.append(os.path.join(root, file))
    return labeled_mp4_files

# Extracts the video name from a full file path
def extract_video_name(full_path):
    base_filename = os.path.basename(full_path)  # Get the base filename
    video_name, _ = os.path.splitext(base_filename)  # Split filename and extension
    return video_name

# Function to select columns from DataFrame based on text similarity
def select_columns_by_text(df, text_pattern):
    selected_columns = [df.columns[0]]  # Always keep the first column

    # Find columns where most of the text is similar to the pattern (case-insensitive)
    for col in df.columns[1:]:
        if isinstance(col, tuple):
            col_name = col[0]  # Extract the actual column name from the tuple
        else:
            col_name = col

        intersection_size = len(set(col_name.lower()) & set(text_pattern.lower()))  # Calculate intersection size
        union_size = max(len(set(col_name.lower())), len(set(text_pattern.lower())))  # Calculate union size
        similarity = intersection_size / union_size  # Calculate Jaccard similarity

        if similarity >= 0.8:  # Threshold for similarity (adjust as needed)
            selected_columns.append(col)

    return df[selected_columns]


def _render_streamlit_fn(state: AppState):

    ## in real dir need to append /models to path 
    proj_dir = state.proj_dir  # "/teamspace/studios/this_studio/Pose-app/data/full_multiModels"

    if proj_dir is None:
        return

    model_dir = abspath(os.path.join(proj_dir, MODELS_DIR)) 
    #/teamspace/studios/this_studio/Pose-app/data/full_multiModels/models/2024-02-05/15-45-00

    # Streamlit UI 
    st.header("Labeled Video")

    with st.sidebar:
        st.title("Visulise Model Predictions")

        st.write("Select the model and then the associated video you wish to inspect")

        st.markdown("### Existing Models")
        trained_models = get_models(model_dir) # create a list of all models 
        selected_model = st.selectbox("Browse", trained_models)
        
        st.divider()
        
        # Labeled videos  
        st.markdown("### Existing Videos")
        mp4_files = list_labeled_mp4_files(model_dir, selected_model)
        selected_video = st.selectbox("Select a video", mp4_files,format_func=extract_video_name)

    if selected_video:
        # read and show the predictions labeled video
        video_file = open(selected_video, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    
