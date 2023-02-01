"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here:
(lai) user@machine: lightning run app app.py

"""

import lightning.app as L
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
from lightning.app.utilities.state import AppState
import streamlit as st


class StreamlitFrontend(LitStreamlitFrontend):
    """VSC requires output to auto forward port"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_server(self, *args, **kwargs):
        super().start_server(*args, **kwargs)
        try:
            print(f"Running streamlit on http://{kwargs['host']}:{kwargs['port']}")
        except:
            # on the cloud, args[0] = host, args[1] = port
            pass


class LandingUI(L.LightningFlow):
    """UI for landing page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):
    st.markdown("""
        TEST
        """, unsafe_allow_html=True)


class LitPoseApp(L.LightningFlow):

    def __init__(self):
        super().__init__()
        self.landing_ui = LandingUI()

    def run(self):
        pass

    def configure_layout(self):
        return [{"name": "Lightning Pose", "content": self.landing_ui}]


app = L.LightningApp(LitPoseApp())
