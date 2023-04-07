"""Landing page for app."""

import streamlit as st

from lightning_pose_app.vsc_streamlit import StreamlitFrontend

from lightning import LightningFlow
from lightning.app.utilities.state import AppState


class LandingUI(LightningFlow):
    """UI for landing page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # output from the UI
        self.st_mode = None
        self.st_action = None
        self.st_proceed_str = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    st.markdown("""
    
        <img src="https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png" alt="Wide Lightning Pose Logo" width="400"/>

        Lightning Pose is a software package for animal pose estimation implemented in 
        **Pytorch Lightning**, 
        supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.
        
        ##### A single application that integrates various components:
        * Label data
        * Train models locally or on the cloud
        * Monitor training
        * View predictions and diagnostics on labeled frames
        * View predictions and diagnostics on unlabeled videos
        
        """, unsafe_allow_html=True)

    proceed_str = "Please proceed to the next tab to {}."
    proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"

    st.markdown(
        """
        #### Run demo
        Train a baseline supervised model and a semi-supervised model on an example dataset.
        """
    )
    button_demo = st.button("Run demo")
    if button_demo:
        state.st_mode = "demo"
        state.st_action = "review and train several baseline models on an example dataset"
        state.st_proceed_str = proceed_str.format(state.st_action)
    if state.st_mode == "demo":
        st.markdown(proceed_fmt % state.st_proceed_str, unsafe_allow_html=True)

    st.markdown(
        """
        #### Manage project
        Start a new project or load an existing project.
        """
    )
    button_new = st.button("Launch project manager")
    if button_new:
        state.st_mode = "project"
        state.st_action = "manage your project"
        state.st_proceed_str = proceed_str.format(state.st_action)
    if state.st_mode == "project":
        st.markdown(proceed_fmt % state.st_proceed_str, unsafe_allow_html=True)

    # button_new = st.button("Load project")
    # if button_new:
    #     state.st_mode = "project"
    #     state.st_action = "manage your project"
    #     state.st_proceed_str = proceed_str.format(state.st_action)
    # if state.st_mode == "project":
    #     st.markdown(proceed_fmt % state.st_proceed_str, unsafe_allow_html=True)
