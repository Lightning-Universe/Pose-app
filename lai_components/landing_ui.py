"""Note: this replaces run_config_ui.py"""

import streamlit as st

from lai_components.vsc_streamlit import StreamlitFrontend

from lightning import LightningFlow
from lightning_app.utilities.state import AppState


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
    
        <img src="https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png" alt="Wide Lightning Pose Logo" width="200"/>

        Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, 
        supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.
        
        #### A single application with pre-integrated components
        * Train Models
        * Training Diagnostics
        * View Predictions on Images
        * Diagnostics on Labeled Images
        * Diagnostics on Unlabeled Videos
        
        """, unsafe_allow_html=True)

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
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "demo":
        st.markdown(
            "<p style='font-family:sans-serif; color:Green;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        #### Start new project
        Start a new project: upload videos, label frames, and train models.
        """
    )
    button_new = st.button("New project")
    if button_new:
        state.st_mode = "new"
        state.st_action = "start a new project"
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "new":
        st.markdown(
            "<p style='font-family:sans-serif; color:Green;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        #### Load existing project
        Train new models, label new frames, process new videos.
        """
    )
    button_load = st.button("Load project")
    if button_load:
        state.st_mode = "resume"
        state.st_action = "resume a previously initialized project"
        state.st_proceed_str = "Please proceed to the next tab to {}.".format(state.st_action)
    if state.st_mode == "resume":
        st.markdown(
            "<p style='font-family:sans-serif; color:Green;'>%s</p>" % state.st_proceed_str,
            unsafe_allow_html=True
        )
