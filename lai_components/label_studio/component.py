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

    def build_commands(self) -> List[str]:
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
    def __init__(self, *args, drive_name=label_studio_drive_name, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.label_studio = LitBashWork(
            cloud_compute=la.CloudCompute("default"),
            cloud_build_config=LabelStudioBuildConfig(),
        )
        self.drive = Drive(drive_name)
        self.count = 0
        self.label_studio_url = 'http://localhost:8080'
        self.data_dir = os.path.join(os.getcwd(), "images_to_label")  # could have subdirs
        self.username = "matt@columbia.edu"
        self.password = "whiteway123"
        self.user_token = "whitenoise1"  # '4949affb1e0883c20552b123a7aded4e6c76760b'

    def start_label_studio(self):
        # create config file
        self.label_studio.run(
            f"sed -e s/__port__/{self.label_studio.port}/g -e s/__host__/{self.label_studio.host}/ nginx-8080.conf > ~/{new_conf_file}",
            wait_for_exit=True,
        )

        # run reverse proxy on external port and remove x-frame-options
        self.label_studio.run(
            f"nginx -c ~/{new_conf_file}",
            wait_for_exit=True,
        )

        # start label-studio on the default port 8080
        # added start, make sure it doesn't break
        # TODO: we need to take in username, password from users. add tokens ourselves so that we can sync data.
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
        # TODO: add args
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "build_labeling_task.py")
        assert os.path.exists(script_path), f"script path does not exist: {script_path}"
        project_name = "test_locally_with_args"
        # TODO: project name as an arg
        # TODO: label_config as a file
        label_config = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "label_config.txt")
        assert os.path.exists(label_config), f"label config does not exist: {label_config}"
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--data_dir {self.data_dir} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {project_name} " \
                        f"--label_config {label_config}"

        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
        )

    def check_labeling_task_and_export(self, time):
        # check labeling task
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "check_labeling_task_and_export.py")
        run_command = f"python {script_path} " \
                      f"--label_studio_url {self.label_studio_url} " \
                      f"--data_dir {self.data_dir} " \
                      f"--api_key {self.user_token}"
        self.label_studio.run(
            run_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            timer=time
        )

    def run(self):

        if self.count == 0:
            self.start_label_studio()
            self.build_labeling_task()

        # execute another command on self.label_studio that checks for updates in the annotation
        # files. if so, we export the data and convert to lightning pose format.
        time.sleep(15)  # TODO: make this programmatic check for 15 seconds difference
        self.check_labeling_task_and_export(time=time.time())
