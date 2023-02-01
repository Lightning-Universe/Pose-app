import sys
import inspect
import os
import subprocess

import lightning.app
from lightning.app.utilities.log import get_logfile
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend


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
