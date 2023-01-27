from lightning import LightningFlow
from lightning.app import BuildConfig
import os
from typing import List

from lai_work.bashwork import LitBashWork


# dir where label studio python venv will be setup
label_studio_venv = "venv-label-studio"
# nginx conf template to remove x-frame-options
conf_file = "nginx-8080.conf"
new_conf_file = "nginx-new-8080.conf"


class LabelStudioBuildConfig(BuildConfig):

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


class LitLabelStudio(LightningFlow):

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
        self.project_name = None
        self.username = "matt@columbia.edu"
        self.password = "whiteway123"
        self.user_token = "whitenoise1"
        self.check_labels = False
        self.time = 0.0

        # paths to relevant data; set when putting/getting from Drive
        self.filenames = {
            "label_studio_config": "",
            "label_studio_metadata": "",
            "labeled_data_dir": "",
            "collected_data": "",
        }

        # these attributes get set by external app
        self.proj_dir = proj_dir
        self.keypoints = None

    @property
    def proj_dir_local(self):
        return os.path.join(os.getcwd(), self.proj_dir)

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

    def create_new_project(self, labeled_data_dir):

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "create_new_project.py")
        label_studio_config_file = os.path.join(os.getcwd(), self.filenames["label_studio_config"])
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.proj_dir_local} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {self.project_name} " \
                        f"--label_config {label_studio_config_file} "

        # cannot use: os.path.join(self.proj_dir, "labeled-data")
        # component cannot find this data. need Path here?
        self.filenames["labeled_data_dir"] = labeled_data_dir

        # specify outputs to put to drive
        self.filenames["label_studio_metadata"] = os.path.join(
            self.proj_dir, "label_studio_metadata.yaml")

        # run command to create new label studio project
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            inputs=[
                self.filenames["label_studio_config"],
                self.filenames["labeled_data_dir"],
            ],
            outputs=[
                self.filenames["label_studio_metadata"],
            ],
        )

    def update_tasks(self, videos=[]):

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "update_tasks.py")
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.proj_dir_local} " \
                        f"--api_key {self.user_token} "

        # run command to update label studio tasks
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            timer=videos,
            inputs=[
                self.filenames["labeled_data_dir"],
                self.filenames["label_studio_metadata"],
            ],
            outputs=[],
        )

    def check_labeling_task_and_export(self, timer):

        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "check_labeling_task_and_export.py")

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # build script command
            keypoints_list = "/".join(self.keypoints)
            run_command = f"python {script_path} " \
                          f"--label_studio_url {self.label_studio_url} " \
                          f"--proj_dir {self.proj_dir_local} " \
                          f"--api_key {self.user_token} " \
                          f"--keypoints_list '{keypoints_list}' "

            self.filenames["collected_data"] = os.path.join(
                self.proj_dir, "CollectedData.csv")
            self.filenames["label_studio_tasks"] = os.path.join(
                self.proj_dir, "label_studio_tasks.pkl")

            # run command to check labeling task
            self.label_studio.run(
                run_command,
                venv_name=label_studio_venv,
                wait_for_exit=True,
                timer=timer,
                inputs=[
                    self.filenames["labeled_data_dir"],
                    self.filenames["label_studio_metadata"],
                ],
                outputs=[
                    self.filenames["collected_data"],
                    self.filenames["label_studio_tasks"],
                    self.filenames["label_studio_metadata"],
                ],
            )

        self.check_labels = True

    def create_labeling_config_xml(self, keypoints):

        self.keypoints = keypoints

        filename = "label_studio_config.xml"
        self.filenames["label_studio_config"] = os.path.join(self.proj_dir, filename)

        # build script command
        keypoints_list = "/".join(keypoints)
        script_path = os.path.join(
            os.getcwd(), "lai_components", "label_studio", "create_labeling_config.py")
        build_command = f"python {script_path} " \
                        f"--proj_dir {self.proj_dir_local} " \
                        f"--filename {filename} " \
                        f"--keypoints_list {keypoints_list} "

        # run command to save out xml
        # NOTE: we cannot just save out the xml from this function, since we are inside a Flow. We
        # need a Work to save the xml so that the Work has local access to that file; the Work can
        # then put that file to a Drive
        self.label_studio.run(
            build_command,
            # venv_name=label_studio_venv,
            wait_for_exit=True,
            timer=keypoints,
            inputs=[],
            outputs=[self.filenames["label_studio_config"]],
        )

    def run(self, action=None, **kwargs):

        if action == "start_label_studio":
            self.start_label_studio()
        elif action == "create_labeling_config_xml":
            self.create_labeling_config_xml(**kwargs)
        elif action == "create_new_project":
            self.create_new_project(**kwargs)
        elif action == "update_tasks":
            self.update_tasks(**kwargs)
        elif action == "check_labeling_task_and_export":
            self.check_labeling_task_and_export(timer=kwargs["timer"])
