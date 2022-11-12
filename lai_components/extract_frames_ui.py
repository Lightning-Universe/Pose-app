from lightning import LightningFlow
from lightning.app.utilities.state import AppState
import numpy as np
import os
import streamlit as st
import yaml

from lai_components.vsc_streamlit import StreamlitFrontend


class ExtractFramesUI(LightningFlow):
    """UI to set up project."""

    def __init__(
        self,
        *args,
        script_dir,
        script_name,
        script_args,
        proj_dir,
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
        self.proj_dir = proj_dir

        # output from the UI
        self.st_submits = 0
        self.st_video_files = None
        self.st_n_frames_per_video = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    # ----------------------------------------------------
    # landing
    # ----------------------------------------------------

    st.markdown(
        """
        ## Extract frames for labeling
        """
    )

    # upload video files
    st_videos = st.text_input("Select video files")
    st_videos = [st_videos]

    # select number of frames to label per video
    n_frames_per_video = st.text_input("Frames to label per video", 20)
    st_n_frames_per_video = int(n_frames_per_video)

    st_submit_button = st.button(
        "Extract frames",
        disabled=(st_n_frames_per_video == 0) or len(st_videos) == 0 or state.run_script
    )
    if state.run_script:
        st.warning(f"waiting for existing extraction to finish")

    if state.st_submits > 0 and not state.run_script:
        proceed_str = "Please proceed to the next tab to label frames."
        proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
        st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

    # Lightning way of returning the parameters
    if st_submit_button:

        state.st_submits += 1

        state.st_video_files = st_videos
        state.st_n_frames_per_video = st_n_frames_per_video

        state.run_script = True  # must the last to prevent race condition
