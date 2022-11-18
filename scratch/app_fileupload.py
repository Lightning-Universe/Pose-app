import os

from lightning.app import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState
from lightning.app.storage import Drive

## TEMPORARY PATCH

import sys
import inspect
import subprocess

import lightning.app
from lightning.app.utilities.log import get_logfile


class StreamlitFrontend_(StreamlitFrontend):

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
                    os.path.join(os.path.dirname(lightning.app.frontend.__file__), "streamlit_base.py"),
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
                    "false"
                ],
                env=env,
                stdout=stdout,
                stderr=stderr,
            )

## PATCH ENDS


class StreamlitUI(LightningFlow):
    def __init__(self, drive):
        super().__init__()
        self.drive = drive

    def configure_layout(self):
        return StreamlitFrontend_(render_fn=render_fn)


def render_fn(state: AppState):
    import streamlit as st

    # initialize the file uploader
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    # for each of the uploaded files
    for uploaded_file in uploaded_files:
        # read it
        bytes_data = uploaded_file.read()
        # build a temporary file name, since we need to pass an actual path,
        # to the Drive, while uploaded_file is just a file-like object
        filename = uploaded_file.name.replace(" ", "_")
        filepath = os.path.join(state.drive.root_folder, filename)
        # write the content of the file to the path
        with open(filepath, "wb") as f:
            f.write(bytes_data)
        # push the data to the Drive, at this point the file has been
        # stored in S3, and can be accessed from other components using
        # the same lit:// path
        state.drive.put(filename)
        # clean up the local file
        os.remove(filepath)


class HelloWorld(LightningFlow):
    def __init__(self):
        super().__init__()
        # initialize a Drive with that path as that root folder
        self.mydrive = Drive("lit://mydrive")
        print("======== drive ========")
        print(self.mydrive.root_folder)
        # instantiate a flow and pass the Drive in
        # here we are assuming we are building a larger application
        # that will pass the drive around, but you can also instantiate
        # the drive in the child flow
        self.streamlit_ui = StreamlitUI(self.mydrive)

    def run(self):
        pass
        # self.streamlit_ui.run()

    def configure_layout(self):
        return [{"name": "StreamLitUI", "content": self.streamlit_ui}]


app = LightningApp(HelloWorld())
