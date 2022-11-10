"""Note: this replaces run_ui.py"""

from lightning import LightningFlow
from lightning.app.utilities.state import AppState
import streamlit as st

from lai_components.vsc_streamlit import StreamlitFrontend


class VideoUI(LightningFlow):
    """UI to enter training parameters for demo data."""

    def __init__(self, *args, video_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_file = video_file

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    # st.markdown(
    #     """
    #     ## Train models
    #     This tab presents options for training two models:
    #     * fully supervised model
    #     * semi-supervised model
    #
    #     """
    # )

    if state.video_file:
        # render video in streamlit
        st.text(f"Rendering video at {state.video_file}")
        st.video(state.video_file)
    else:
        st.text("Rendering labeled video, please wait...")
