import lightning.app as la
from lightning.app.storage.drive import Drive
from lai_work.bashwork import LitBashWork

from pathlib import Path
from string import Template
import time
import os
from typing import Optional, Union, List

# dir where label studio python venv will be setup
label_studio_venv = "venv-label-studio"
# Lighting App Drive name to exchange dirs and files
label_studio_drive_name = "lit://label-studio"
# nginx conf template to remove x-frame-options
conf_file = "nginx-8080.conf"
new_conf_file = "nginx-new-8080.conf"


class LabelStudioBuildConfig(la.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        # added an install for label-studio-sdk to automatically launch the label studio server.
        return [
            "sudo apt-get update",
            "sudo apt-get install nginx",
            "sudo touch /run/nginx.pid",
            "sudo chown -R `whoami` /etc/nginx/ /var/log/nginx/",
            "sudo chown -R `whoami` /var/lib/nginx/",
            "sudo chown `whoami` /run/nginx.pid",
            f"virtualenv ~/{label_studio_venv}",
            f". ~/{label_studio_venv}/bin/activate; which python; "
            f"python -m pip install label-studio label-studio-sdk; deactivate",
        ]


class LitLabelStudio(la.LightningFlow):

    def __init__(
        self,
        *args,
        drive_name=label_studio_drive_name,
        data_dir=None,  # this should point to
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label_studio = LitBashWork(
            cloud_compute=la.CloudCompute("default"),
            cloud_build_config=LabelStudioBuildConfig(),
        )
        self.drive = Drive(drive_name)
        self.count = 0
        self.label_studio_url = "http://localhost:8080"
        self.label_studio_config_file = None
        self.project_name = None
        self.username = "matt@columbia.edu"
        self.password = "whiteway123"
        self.user_token = "whitenoise1"
        self.time = time.time()

        # these attributes get set by external app
        self.data_dir = data_dir
        self.keypoints = None

    def start_label_studio(self):

        # create config file
        self.label_studio.run(
            f"sed -e s/__port__/{self.label_studio.port}/g -e s/__host__/{self.label_studio.host}/"
            f" nginx-8080.conf > ~/{new_conf_file}",
            wait_for_exit=True,
        )

        # run reverse proxy on external port and remove x-frame-options
        self.label_studio.run(
            f"nginx -c ~/{new_conf_file}",
            wait_for_exit=True,
        )

        # start label-studio on the default port 8080
        self.label_studio.run(
            f"label-studio start --no-browser --internal-host $host",
            venv_name=label_studio_venv,
            wait_for_exit=False,
            env={
                'LABEL_STUDIO_USERNAME': self.username,
                'LABEL_STUDIO_PASSWORD': self.password,
                'LABEL_STUDIO_USER_TOKEN': self.user_token,
                'USE_ENFORCE_CSRF_CHECKS': 'false',
                'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED': 'true',
                'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT': os.path.abspath(os.getcwd()),
                'LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK': 'true',
            },
        )

        self.count += 1

    def build_labeling_task(self):
        # create labeling task
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "build_labeling_task.py")
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--data_dir {self.data_dir} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {self.project_name} " \
                        f"--label_config {self.label_studio_config_file} "
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
        )

    def check_labeling_task_and_export(self, time):
        # check labeling task
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "check_labeling_task_and_export.py")
        keypoints_list = ";".join(self.keypoints)
        run_command = f"python {script_path} " \
                      f"--label_studio_url {self.label_studio_url} " \
                      f"--data_dir {self.data_dir} " \
                      f"--api_key {self.user_token} " \
                      f"--keypoints_list {keypoints_list} "
        self.label_studio.run(
            run_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            timer=time
        )

    def create_labeling_config_xml(self, keypoints):
        self.keypoints = keypoints
        xml_str = build_xml(keypoints)
        filename = os.path.join(self.data_dir, "label_studio_config.xml")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        with open(filename, 'wt') as outfile:
            outfile.write(xml_str)
        self.label_studio_config_file = filename

    def run(self):

        if self.count == 0:
            self.start_label_studio()
            self.build_labeling_task()

        # check for new annotations every 15 seconds
        # if updates, export the data and convert to lightning pose format
        new_time = time.time()
        if (new_time - self.time) > 15:
            self.time = new_time
            self.check_labeling_task_and_export(time=new_time)


def build_xml(bodypart_names: List[str]) -> str:
    """Builds the XML file for Label Studio"""
    # 25 colors
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gray",
              "black", "white", "cyan", "magenta", "lime", "maroon", "olive", "navy", "teal",
              "aqua", "fuchsia", "silver", "gold", "indigo", "violet", "coral"]
    # replicate just to be safe
    colors = colors + colors + colors + colors
    colors_to_use = colors[:len(bodypart_names)]  # practically ignoring colors
    view_str = "<!--Basic keypoint image labeling configuration for multiple regions-->"
    view_str += "\n<View>"
    view_str += "\n<Header value=\"Select keypoint name with the cursor/number button, " \
                "then click on the image.\"/>"
    view_str += "\n<Text name=\"text1\" value=\"Important: Click Submit after you have labeled " \
                "all visible keypoints in this image.\"/>"
    view_str += "\n<Text name=\"text2\" value=\"Also useful: Press H for hand tool, " \
                "CTRL+ to zoom in and CTRL- to zoom out\"/>"
    view_str += "\n  <KeyPointLabels name=\"kp-1\" toName=\"img-1\" strokeWidth=\"3\">"  # indent 2
    for keypoint, color in zip(bodypart_names, colors_to_use):
        view_str += f"\n    <Label value=\"{keypoint}\" />"  # indent 4
    view_str += "\n  </KeyPointLabels>"  # indent 2
    view_str += "\n  <Image name=\"img-1\" value=\"$img\" />"  # indent 2
    view_str += "\n</View>"  # indent 0
    return view_str
