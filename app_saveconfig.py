import os
import streamlit as st
import yaml

import lightning.app
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState
from lightning.app.storage import Drive


class StreamlitUI(LightningFlow):

    def __init__(self):
        super().__init__()

        self.update_file = False

        # output from the UI
        self.st_project_name = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    st_project_name = st.text_input("Enter project name", value="")
    st_submit_button = st.button("Update project", disabled=(len(st_project_name) == 0))
    if st_submit_button:
        state.st_project_name = st_project_name
        state.update_file = True


class DataIO(LightningWork):

    def __init__(self, *args, drive_name, config_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = Drive(drive_name)
        self.config_file = config_file

    def run(self, project_name):

        # make config
        config_dict = {"project_name": project_name}

        # save locally
        # config_file_abspath = os.path.join(self.drive.root_folder, config_file)
        print(os.path.abspath(self.config_file))
        yaml.dump(config_dict, open(self.config_file, "w"))

        # push to drive
        self.drive.put(self.config_file)

        # clean up the local file
        os.remove(self.config_file)


class Test(LightningFlow):

    def __init__(self):
        super().__init__()
        self.streamlit_ui = StreamlitUI()
        self.data_io = DataIO(drive_name="lit://lpa", config_file="config.yaml")

    def run(self):

        self.streamlit_ui.run()
        if self.streamlit_ui.update_file:
            self.data_io.run(self.streamlit_ui.st_project_name)
            self.streamlit_ui.update_file = False

    def configure_layout(self):
        return [{"name": "StreamLitUI", "content": self.streamlit_ui}]


app = LightningApp(Test())
