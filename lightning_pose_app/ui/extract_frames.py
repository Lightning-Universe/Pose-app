from lightning import LightningFlow
from lightning.app.utilities.state import AppState
from lightning.app.storage.drive import Drive
import numpy as np
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from lightning_pose_app.utilities import StreamlitFrontend


class ExtractFramesUI(LightningFlow):
    """UI to set up project."""

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
        self.run_script = False

        # send info to user
        self.st_video_files_ = []
        self.st_extract_status = {}  # 'initialized' | 'active' | 'complete'

        # save parameters for later run
        self.proj_dir = None

        # output from the UI
        self.st_submits = 0
        self.st_n_frames_per_video = None

    @property
    def st_video_files(self):
        return np.unique(self.st_video_files_).tolist()

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    st.markdown(
        """
        ## Extract frames for labeling
        """
    )

    if state.run_script:
        # don't autorefresh during large file uploads, only during processing
        st_autorefresh(interval=5000, key="refresh_extract_frames_ui")

    # upload video files
    video_dir = os.path.join(state.proj_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # initialize the file uploader
    uploaded_files = st.file_uploader("Select video files", accept_multiple_files=True)

    # for each of the uploaded files
    st_videos = []
    for uploaded_file in uploaded_files:
        # read it
        bytes_data = uploaded_file.read()
        # name it (append "_"
        filename = uploaded_file.name.replace(" ", "_")
        ext = os.path.splitext(filename)[1]
        filename = filename.replace(ext, f"_tmp{ext}")
        filepath = os.path.join(video_dir, filename)
        st_videos.append(filepath)
        if not state.run_script:
            # write the content of the file to the path, but not while processing
            with open(filepath, "wb") as f:
                f.write(bytes_data)
            # push the data to the Drive
            state.drive.put(filepath)
            # clean up the local file
            # os.remove(filepath)

    # select number of frames to label per video
    n_frames_per_video = st.text_input("Frames to label per video", 20)
    st_n_frames_per_video = int(n_frames_per_video)

    st_submit_button = st.button(
        "Extract frames",
        disabled=(st_n_frames_per_video == 0) or len(st_videos) == 0 or state.run_script
    )

    if state.run_script:
        for vid, status in state.st_extract_status.items():
            if status == "initialized":
                p = 0.0
            elif status == "active":
                p = 50.0
            elif status == "complete":
                p = 100.0
            else:
                st.text(status)
            st.progress(p / 100.0, f"{vid} progress ({status})")
        st.warning(f"waiting for existing extraction to finish")

    if state.st_submits > 0 and not st_submit_button and not state.run_script:
        proceed_str = "Please proceed to the next tab to label frames."
        proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
        st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

    # Lightning way of returning the parameters
    if st_submit_button:

        state.st_submits += 1

        state.st_video_files_ = st_videos
        state.st_extract_status = {s: 'initialized' for s in st_videos}
        state.st_n_frames_per_video = st_n_frames_per_video
        st.text("Request submitted!")
        state.run_script = True  # must the last to prevent race condition

        st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")
