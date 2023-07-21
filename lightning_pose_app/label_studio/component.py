from lightning import CloudCompute, LightningFlow
import os

from lightning_pose_app import (
    LABELED_DATA_DIR,
    LABELSTUDIO_CONFIG_FILENAME,
    LABELSTUDIO_METADATA_FILENAME,
    LABELSTUDIO_TASKS_FILENAME,
    COLLECTED_DATA_FILENAME,
)
from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LabelStudioBuildConfig, label_studio_venv


class LitLabelStudio(LightningFlow):

    def __init__(self, *args, database_dir="/data", proj_dir=None, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.label_studio = LitBashWork(
            name="labelstudio",
            cloud_compute=CloudCompute("default"),
            cloud_build_config=LabelStudioBuildConfig(),
        )
        self.counts = {
            "import_database": 0,
            "start_label_studio": 0,
            "create_new_project": 0,
            "import_existing_annotations": 0,
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
            "label_studio_tasks": "",
            "labeled_data_dir": "",
            "collected_data": "",
            "config_file": "",
        }

        # these attributes get set by external app
        self.proj_dir = proj_dir
        self.proj_name = None
        self.keypoints = None

    @staticmethod
    def abspath(path):
        if path[0] == "/":
            path_ = path[1:]
        else:
            path_ = path
        return os.path.abspath(path_)

    def _import_database(self):
        # pull database from FileSystem if it exists
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

    def _start_label_studio(self):

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
                "LABEL_STUDIO_BASE_DATA_DIR": self.abspath(self.database_dir),
                "LABEL_STUDIO_SESSION_COOKIE_SAMESITE": "Lax",
                "LABEL_STUDIO_CSRF_COOKIE_SAMESITE": "Lax",
                "LABEL_STUDIO_SESSION_COOKIE_SECURE": "1",
                "LABEL_STUDIO_USE_ENFORCE_CSRF_CHECKS": "0",
            },
        )

        self.counts["start_label_studio"] += 1

    def _update_paths(self, proj_dir, proj_name):

        self.proj_dir = proj_dir
        self.proj_name = proj_name

        self.filenames["label_studio_config"] = os.path.join(
            self.proj_dir, LABELSTUDIO_CONFIG_FILENAME)

        self.filenames["label_studio_metadata"] = os.path.join(
            self.proj_dir, LABELSTUDIO_METADATA_FILENAME)

        self.filenames["label_studio_tasks"] = os.path.join(
            self.proj_dir, LABELSTUDIO_TASKS_FILENAME)

        self.filenames["labeled_data_dir"] = os.path.join(self.proj_dir, LABELED_DATA_DIR)

        self.filenames["collected_data"] = os.path.join(self.proj_dir, COLLECTED_DATA_FILENAME)

        self.filenames["config_file"] = os.path.join(
            self.proj_dir, f"model_config_{self.proj_name}.yaml")

    def _create_new_project(self):
        """Create a label studio project."""

        if self.counts["create_new_project"] > 0:
            return

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "create_new_project.py")
        label_studio_config_file = self.abspath(self.filenames["label_studio_config"])
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.abspath(self.proj_dir)} " \
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

    def _update_tasks(self, videos=[]):
        """Update tasks after new video frames have been extracted."""

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "update_tasks.py")
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.abspath(self.proj_dir)} " \
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

    def _check_labeling_task_and_export(self, timer):
        """Check for new labels, export to lightning pose format, export database to FileSystem."""

        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "check_labeling_task_and_export.py")

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # build script command
            keypoints_list = "/".join(self.keypoints)
            run_command = f"python {script_path} " \
                          f"--label_studio_url {self.label_studio_url} " \
                          f"--proj_dir {self.abspath(self.proj_dir)} " \
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

    def _create_labeling_config_xml(self, keypoints):
        """Create a label studio configuration xml file."""

        self.keypoints = keypoints

        # build script command
        keypoints_list = "/".join(keypoints)
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "create_labeling_config.py")
        build_command = f"python {script_path} " \
                        f"--proj_dir {self.abspath(self.proj_dir)} " \
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

    def _import_existing_annotations(self, **kwargs):
        """Import annotations into an existing, empty label studio project."""

        if self.counts["import_existing_annotations"] > 0:
            return

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "update_tasks.py")
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {self.abspath(self.proj_dir)} " \
                        f"--api_key {self.user_token} " \
                        f"--config_file {self.abspath(self.filenames['config_file'])} " \
                        f"--update_from_csv "

        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            inputs=[
                self.filenames["labeled_data_dir"],
                self.filenames["label_studio_metadata"],
                self.filenames["collected_data"],
                self.filenames["config_file"],
            ],
            outputs=[]
        )

        self.counts["import_existing_annotations"] += 1

    def run(self, action=None, **kwargs):

        if action == "import_database":
            self._import_database()
        elif action == "start_label_studio":
            self._start_label_studio()
        elif action == "create_labeling_config_xml":
            self._create_labeling_config_xml(**kwargs)
        elif action == "create_new_project":
            self._create_new_project()
        elif action == "update_tasks":
            self._update_tasks(**kwargs)
        elif action == "check_labeling_task_and_export":
            self._check_labeling_task_and_export(timer=kwargs["timer"])
        elif action == "update_paths":
            self._update_paths(**kwargs)
        elif action == "import_existing_annotations":
            self._import_existing_annotations(**kwargs)

    def on_exit(self):
        # final save to drive
        print("SAVING DATA ONE LAST TIME")
        self._check_labeling_task_and_export(timer=0.0)
