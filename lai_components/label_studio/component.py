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
            f". ~/{label_studio_venv}/bin/activate; which python; ",
            f"python -m pip install -e .; ",
            f"python -m pip install label-studio label-studio-sdk; deactivate",
        ]


class LitLabelStudio(la.LightningFlow):

    def __init__(
        self,
        *args,
        cloud_compute,
        drive_name,
        proj_dir=None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label_studio = LitBashWork(
            cloud_compute=cloud_compute,
            cloud_build_config=LabelStudioBuildConfig(),
            drive_name=drive_name,
            component_name="label_studio",
        )
        self.count = 0
        self.label_studio_url = "http://localhost:8080"
        self.label_studio_config_file = None
        self.project_name = None
        self.username = "matt@columbia.edu"
        self.password = "whiteway123"
        self.user_token = "whitenoise1"
        self.time = time.time()

        # these attributes get set by external app
        self.proj_dir = proj_dir
        self.keypoints = None

    def start_label_studio(self):

        if self.count > 0:
            return

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

    def create_new_project(self):

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "create_new_project.py")
        proj_dir = os.path.join(os.getcwd(), self.proj_dir)
        label_studio_config_file = os.path.join(os.getcwd(), self.label_studio_config_file)
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {proj_dir} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {self.project_name} " \
                        f"--label_config {label_studio_config_file} "

        # specify inputs to get from drive
        inputs = [
            self.label_studio_config_file,
            os.path.join(self.proj_dir, "labeled-data")
        ]

        # specify outputs to put to drive
        outputs = [os.path.join(self.proj_dir, "label_studio_metadata.yaml")]

        # run command to create new label studio project
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            inputs=inputs,
            outputs=outputs,
        )

    def update_tasks(self, videos=[]):

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "update_tasks.py")
        proj_dir = os.path.join(os.getcwd(), self.proj_dir)
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {proj_dir} " \
                        f"--api_key {self.user_token} "

        # specify inputs to get from drive
        inputs = [os.path.join(self.proj_dir, "labeled-data")]

        # specify outputs to put to drive
        outputs = []

        # run command to update label studio tasks
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            timer=videos,
            inputs=inputs,
            outputs=outputs,
        )

    def check_labeling_task_and_export(self, timer):

        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "check_labeling_task_and_export.py")

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # build script command
            keypoints_list = "/".join(self.keypoints)
            proj_dir = os.path.join(os.getcwd(), self.proj_dir)
            run_command = f"python {script_path} " \
                          f"--label_studio_url {self.label_studio_url} " \
                          f"--proj_dir {proj_dir} " \
                          f"--api_key {self.user_token} " \
                          f"--keypoints_list '{keypoints_list}' "

            # specify inputs to get from drive
            inputs = [os.path.join(self.proj_dir, "labeled-data")]

            # specify outputs to put to drive
            outputs = [
                os.path.join(self.proj_dir, "CollectedData.csv"),
                os.path.join(self.proj_dir, "label_studio_tasks.pkl"),
                os.path.join(self.proj_dir, "label_studio_metadata.yaml"),
            ]

            # run command to check labeling task
            self.label_studio.run(
                run_command,
                venv_name=label_studio_venv,
                wait_for_exit=True,
                timer=timer,
                inputs=inputs,
                outputs=outputs,
            )

    def create_labeling_config_xml(self, keypoints):
        self.keypoints = keypoints
        xml_str = build_xml(keypoints)
        filename = os.path.join(self.proj_dir, "label_studio_config.xml")
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
        with open(filename, 'wt') as outfile:
            outfile.write(xml_str)
        self.label_studio_config_file = filename

        # put xml file on drive
        self.label_studio.run(
            args="",
            input_output_only=True,
            inputs=[],
            outputs=[filename]
        )

    def run(self, action=None, **kwargs):

        if action == "start_label_studio":
            self.start_label_studio()
        elif action == "create_labeling_config_xml":
            self.create_labeling_config_xml(**kwargs)
        elif action == "create_new_project":
            self.create_new_project()
        elif action == "update_tasks":
            self.update_tasks(**kwargs)
        elif action == "check_labeling_task_and_export":
            # check for new annotations every n seconds
            # if updates, export the data and convert to lightning pose format
            n = 15
            new_time = kwargs["timer"]
            if (new_time - self.time) > n:
                self.time = new_time
                self.check_labeling_task_and_export(timer=new_time)


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
