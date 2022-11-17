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


class StreamlitFrontendFileUploader(LitStreamlitFrontend):
    """From Luca, for uploading files into Drive."""

    def start_server(self, host: str, port: int) -> None:
        env = os.environ.copy()
        env["LIGHTNING_FLOW_NAME"] = self.flow.name
        env["LIGHTNING_RENDER_FUNCTION"] = self.render_fn.__name__
        env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(self.render_fn).__file__
        std_err_out = get_logfile("error.log")
        std_out_out = get_logfile("output.log")
        with open(std_err_out, "wb") as stderr, open(std_out_out, "wb") as stdout:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    os.path.join(
                        os.path.dirname(lightning.app.frontend.__file__), "streamlit_base.py"),
                    "--server.address",
                    str(host),
                    "--server.port",
                    str(port),
                    "--server.baseUrlPath",
                    self.flow.name,
                    "--server.headless",
                    "true",  # do not open the browser window when running locally
                    "--server.enableCORS",
                    "true",
                    "--server.enableXsrfProtection",
                    "false",
                    "--server.maxUploadSize",
                    "2000"
                ],
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
