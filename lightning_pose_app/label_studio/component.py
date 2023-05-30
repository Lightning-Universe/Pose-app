from lightning import LightningFlow
import os

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LabelStudioBuildConfig, label_studio_venv


class LitLabelStudio(LightningFlow):

    def __init__(
        self,
        *args,
        cloud_compute,
        drive_name,
        database_dir="data",
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
        self.counts = {
            "import_database": 0,
            "start_label_studio": 0,
            "create_new_project": 0,
        }
        self.label_studio_url = f"http://localhost:{self.label_studio.port}"
        self.username = "user@localhost"
        self.password = "pw"
        self.user_token = "whitenoise"
        self.check_labels = False
        self.time = 0.0

        # location of label studio sqlite database relative to current working directory
        self.database_dir = database_dir

        # paths to relevant data; set when putting/getting from Drive
        self.filenames = {
            "label_studio_config": "",
            "label_studio_metadata": "",
            "labeled_data_dir": "",
            "collected_data": "",
            "label_studio_tasks": "",
        }

        # these attributes get set by external app
        self.proj_dir = proj_dir
        self.proj_name = None
        self.keypoints = None

    @property
    def proj_dir_local(self):
        return os.path.join(os.getcwd(), self.proj_dir)

    def import_database(self):
        # pull database from Drive if it exists
        # NOTE: db must be imported _after_ LabelStudio is started, otherwise some nginx error
        if self.counts["import_database"] > 0:
            return

        self.label_studio.run(
            "null command",
            venv_name=label_studio_venv,
            cwd=os.getcwd(),
            input_output_only=True,
            inputs=[self.database_dir],
            wait_for_exit=True,
        )

        self.counts["import_database"] += 1

    def start_label_studio(self):

        if self.counts["start_label_studio"] > 0:
            return

        # start label-studio on the default port 8080
        self.label_studio.run(
            f"label-studio start --no-browser --internal-host $host --port $port",
            venv_name=label_studio_venv,
            wait_for_exit=False,
            env={
                "LABEL_STUDIO_USERNAME": self.username,
                "LABEL_STUDIO_PASSWORD": self.password,
                "LABEL_STUDIO_USER_TOKEN": self.user_token,
                "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
                "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": os.path.abspath(os.getcwd()),
                "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
                "LABEL_STUDIO_BASE_DATA_DIR": os.path.join(os.getcwd(), self.database_dir),
                "LABEL_STUDIO_SESSION_COOKIE_SAMESITE": "Lax",
                "LABEL_STUDIO_CSRF_COOKIE_SAMESITE": "Lax",
                "LABEL_STUDIO_SESSION_COOKIE_SECURE": "1",
                "LABEL_STUDIO_USE_ENFORCE_CSRF_CHECKS": "0",
            },
        )

        self.counts["start_label_studio"] += 1

    def update_paths(self, proj_dir, proj_name):

        self.proj_dir = proj_dir
        self.proj_name = proj_name

        self.filenames["label_studio_config"] = os.path.join(
            self.proj_dir, "label_studio_config.xml")

        self.filenames["label_studio_metadata"] = os.path.join(
            self.proj_dir, "label_studio_metadata.yaml")

        self.filenames["labeled_data_dir"] = os.path.join(self.proj_dir, "labeled-data")

        self.filenames["collected_data"] = os.path.join(self.proj_dir, "CollectedData.csv")

        self.filenames["label_studio_tasks"] = os.path.join(
            self.proj_dir, "label_studio_tasks.pkl")

    def create_new_project(self):

        if self.counts["create_new_project"] > 0:
            return

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "create_new_project.py")
        label_studio_config_file = os.path.join(os.getcwd(), self.filenames["label_studio_config"])
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.proj_dir_local} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {self.proj_name} " \
                        f"--label_config {label_studio_config_file} "

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

        self.counts["create_new_project"] += 1

    def update_tasks(self, videos=[]):
        """Update tasks after new video frames have been extracted."""

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "update_tasks.py")
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
        """Check for new labels, export to lightning pose format, export database to Drive"""

        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "check_labeling_task_and_export.py")

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # build script command
            keypoints_list = "/".join(self.keypoints)
            run_command = f"python {script_path} " \
                          f"--label_studio_url {self.label_studio_url} " \
                          f"--proj_dir {self.proj_dir_local} " \
                          f"--api_key {self.user_token} " \
                          f"--keypoints_list '{keypoints_list}' "

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
                    self.database_dir,  # sqlite database
                ],
            )

        self.check_labels = True

    def create_labeling_config_xml(self, keypoints):

        self.keypoints = keypoints

        # build script command
        keypoints_list = "/".join(keypoints)
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "create_labeling_config.py")
        build_command = f"python {script_path} " \
                        f"--proj_dir {self.proj_dir_local} " \
                        f"--filename {os.path.basename(self.filenames['label_studio_config'])} " \
                        f"--keypoints_list {keypoints_list} "

        # run command to save out xml
        # NOTE: we cannot just save out the xml from this function, since we are inside a Flow. We
        # need a Work to save the xml so that the Work has local access to that file; the Work can
        # then put that file to a Drive
        self.label_studio.run(
            build_command,
            wait_for_exit=True,
            timer=keypoints,
            inputs=[],
            outputs=[self.filenames["label_studio_config"]],
        )

    def run(self, action=None, **kwargs):

        if action == "import_database":
            self.import_database()
        elif action == "start_label_studio":
            self.start_label_studio()
        elif action == "create_labeling_config_xml":
            self.create_labeling_config_xml(**kwargs)
        elif action == "create_new_project":
            self.create_new_project()
        elif action == "update_tasks":
            self.update_tasks(**kwargs)
        elif action == "check_labeling_task_and_export":
            self.check_labeling_task_and_export(timer=kwargs["timer"])
        elif action == "update_paths":
            self.update_paths(**kwargs)
