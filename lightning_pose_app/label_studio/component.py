import logging
import os

from lightning.app import CloudCompute, LightningFlow

from lightning_pose_app import (
    COLLECTED_DATA_FILENAME,
    LABELED_DATA_DIR,
    LABELSTUDIO_CONFIG_FILENAME,
    LABELSTUDIO_METADATA_FILENAME,
    LABELSTUDIO_TASKS_FILENAME,
)
from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.utilities import abspath

_logger = logging.getLogger('APP.LABELSTUDIO')
log_level = "ERROR"  # log level sent to label studio sdk

label_studio_venv = None


class LitLabelStudio(LightningFlow):

    def __init__(self, *args, database_dir="/data", proj_dir=None, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.label_studio = LitBashWork(
            cloud_compute=CloudCompute("default"),
        )
        self.counts = {
            "start_label_studio": 0,
            "create_new_project": 0,
            "import_existing_annotations": 0,
        }
        self.label_studio_url = None
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

    def _start_label_studio(self):

        if self.counts["start_label_studio"] > 0:
            return

        # assign label studio url here; note that, in a lightning studio, if you "share" the port
        # display it will increment the port info. therefore, you must start label studio in the
        # same window that you will be using it in
        self.label_studio_url = f"http://localhost:{self.label_studio.port}"

        # start label-studio
        self.label_studio.run(
            "label-studio start --no-browser --internal-host $host --port $port --log-level ERROR",
            venv_name=label_studio_venv,
            wait_for_exit=False,
            env={
                "LOG_LEVEL": log_level,
                "LABEL_STUDIO_USERNAME": self.username,
                "LABEL_STUDIO_PASSWORD": self.password,
                "LABEL_STUDIO_USER_TOKEN": self.user_token,
                "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
                "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": os.path.abspath(os.getcwd()),
                "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
                "LABEL_STUDIO_BASE_DATA_DIR": abspath(self.database_dir),
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

        if self.proj_dir is None or self.proj_name is None:
            for key in self.filenames.keys():
                self.filenames[key] = ""
        else:
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
        label_studio_config_file = abspath(self.filenames["label_studio_config"])
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {abspath(self.proj_dir)} " \
                        f"--api_key {self.user_token} " \
                        f"--project_name {self.proj_name} " \
                        f"--label_config {label_studio_config_file} "

        # put this here to make sure `self.label_studio.run()` is only called once
        self.counts["create_new_project"] += 1

        # run command to create new label studio project
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            env={"LOG_LEVEL": log_level},
        )

    def _update_tasks(self, videos=[]):
        """Update tasks after new video frames have been extracted."""

        # build script command
        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "update_tasks.py")
        build_command = f"python {script_path} " \
                        f"--label_studio_url {self.label_studio_url} " \
                        f"--proj_dir {abspath(self.proj_dir)} " \
                        f"--api_key {self.user_token} "

        # run command to update label studio tasks
        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            env={"LOG_LEVEL": log_level},
            timer=videos,
        )

    def _check_labeling_task_and_export(self, timer):
        """Check for new labels, export to lightning pose format."""

        script_path = os.path.join(
            os.getcwd(), "lightning_pose_app", "label_studio", "check_labeling_task_and_export.py")

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # build script command
            keypoints_list = "/".join(self.keypoints)
            run_command = f"python {script_path} " \
                          f"--label_studio_url {self.label_studio_url} " \
                          f"--proj_dir {abspath(self.proj_dir)} " \
                          f"--api_key {self.user_token} " \
                          f"--keypoints_list '{keypoints_list}' "

            # run command to check labeling task
            self.label_studio.run(
                run_command,
                venv_name=label_studio_venv,
                wait_for_exit=True,
                env={"LOG_LEVEL": log_level},
                timer=timer,
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
                        f"--proj_dir {abspath(self.proj_dir)} " \
                        f"--filename {os.path.basename(self.filenames['label_studio_config'])} " \
                        f"--keypoints_list {keypoints_list} "

        # run command to save out xml
        self.label_studio.run(
            build_command,
            wait_for_exit=True,
            env={"LOG_LEVEL": log_level},
            timer=keypoints,
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
                        f"--proj_dir {abspath(self.proj_dir)} " \
                        f"--api_key {self.user_token} " \
                        f"--config_file {abspath(self.filenames['config_file'])} " \
                        f"--update_from_csv "

        self.label_studio.run(
            build_command,
            venv_name=label_studio_venv,
            wait_for_exit=True,
            env={"LOG_LEVEL": log_level},
        )

        self.counts["import_existing_annotations"] += 1

    def _delete_project(self, **kwargs):
        """Delete a project from the label studio database."""

        # reset paths
        self.keypoints = None
        self._update_paths(proj_dir=None, proj_name=None)

        # NOTE:
        # the below will delete the project from the label studio database
        # this is commented out to force users to do this manually as an added safey measure

        # # build script command
        # script_path = os.path.join(
        #     os.getcwd(), "lightning_pose_app", "label_studio", "delete_project.py")
        # build_command = f"python {script_path} " \
        #                 f"--label_studio_url {self.label_studio_url} " \
        #                 f"--proj_dir {abspath(self.proj_dir)} " \
        #                 f"--api_key {self.user_token} "
        #
        # # run command to update label studio tasks
        # self.label_studio.run(
        #     build_command,
        #     venv_name=label_studio_venv,
        #     wait_for_exit=True,
        #     env={"LOG_LEVEL": log_level},
        #     timer=self.time,
        # )
        #
        # # reset paths
        # self.keypoints = None
        # self._update_paths(proj_dir=None, proj_name=None)

    def run(self, action=None, **kwargs):

        if action == "start_label_studio":
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
        elif action == "delete_project":
            self._delete_project(**kwargs)

    def on_exit(self):
        # final save
        _logger.info("SAVING DATA ONE LAST TIME")
        self._check_labeling_task_and_export(timer=0.0)
