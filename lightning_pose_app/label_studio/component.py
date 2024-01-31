import datetime
from lightning.app import CloudCompute, LightningFlow
import logging
import os
import pandas as pd
import subprocess
import yaml

from lightning_pose_app import (
    LABELED_DATA_DIR,
    LABELSTUDIO_CONFIG_FILENAME,
    LABELSTUDIO_METADATA_FILENAME,
    LABELSTUDIO_TASKS_FILENAME,
    COLLECTED_DATA_FILENAME,
)
from lightning_pose_app.bashwork import add_to_system_env
from lightning_pose_app.label_studio.utils import (
    build_xml,
    connect_to_label_studio,
    get_annotation,
)
from lightning_pose_app.utilities import WorkWithFileSystem


_logger = logging.getLogger('APP.LABELSTUDIO')
log_level = "ERROR"  # log level sent to label studio sdk


class LitLabelStudio(WorkWithFileSystem):

    def __init__(self, *args, database_dir="/data", proj_dir=None, **kwargs) -> None:

        super().__init__(*args, name="labelstudio", **kwargs)
        self.pid = None

        self.counts = {
            "import_database": 0,
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

    def _import_database(self):
        # pull database from FileSystem if it exists
        # NOTE: db must be imported _after_ LabelStudio is started, otherwise some nginx error
        if self.counts["import_database"] > 0:
            return
        self.get_from_drive([self.database_dir])
        self.counts["import_database"] += 1

    def _start_label_studio(self):

        if self.counts["start_label_studio"] > 0:
            return

        # assign label studio url here; note that, in a lightning studio, if you "share" the port
        # display it will increment the port info. therefore, you must start label studio in the
        # same window that you will be using it in
        self.label_studio_url = f"http://localhost:{self.port}"

        cmd = f"label-studio start --no-browser --internal-host {self.host} --port {self.port}"
        env = {
            "LOG_LEVEL": log_level,
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
        }

        kwargs = {'env': add_to_system_env(env=env)}
        _logger.info(cmd, kwargs)
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            close_fds=True,
            **kwargs
        )
        self.pid = proc.pid

        self.counts["start_label_studio"] += 1

    def _update_paths(self, proj_dir, proj_name):

        self.proj_dir = proj_dir
        self.proj_name = proj_name
        print(" ----------------- HERE 1 -------------------")
        self.filenames["label_studio_config"] = os.path.join(
            self.proj_dir, LABELSTUDIO_CONFIG_FILENAME)

        self.filenames["label_studio_metadata"] = os.path.join(
            self.proj_dir, LABELSTUDIO_METADATA_FILENAME)

        self.filenames["label_studio_tasks"] = os.path.join(
            self.proj_dir, LABELSTUDIO_TASKS_FILENAME)

        self.filenames["labeled_data_dir"] = os.path.join(
            self.proj_dir, LABELED_DATA_DIR)

        self.filenames["collected_data"] = os.path.join(
            self.proj_dir, COLLECTED_DATA_FILENAME)

        self.filenames["config_file"] = os.path.join(
            self.proj_dir, f"model_config_{self.proj_name}.yaml")
        print(" ----------------- HERE 2 -------------------")
        print(self.proj_dir)
        print(self.filenames)

    def _create_new_project(self):
        """Create a label studio project."""

        if not self.filenames['label_studio_config']:
            # do not execute if filenames have not been updated
            return

        if self.counts["create_new_project"] > 0:
            return

        # put this here to make sure `self.label_studio.run()` is only called once
        self.counts["create_new_project"] += 1

        # pull data from FileSystem
        inputs = [
            self.filenames["label_studio_config"],
            self.filenames["labeled_data_dir"],
        ]
        self.get_from_drive(inputs)

        # create new project
        create_new_project(
            label_studio_url=self.label_studio_url,
            proj_dir=self.abspath(self.proj_dir),
            api_key=self.user_token,
            project_name=self.proj_name,
            label_config_file=self.abspath(self.filenames["label_studio_config"]),
        )

        # put data to FileSystem
        self.put_to_drive([self.filenames["label_studio_metadata"]])

    def _update_tasks(self, videos=[]):
        """Update tasks after new video frames have been extracted."""

        # pull data from FileSystem
        inputs = [
            self.filenames["labeled_data_dir"],
            self.filenames["label_studio_metadata"],
        ]
        self.get_from_drive(inputs)

        # update tasks
        update_tasks(
            label_studio_url=self.label_studio_url, 
            proj_dir=self.abspath(self.proj_dir), 
            api_key=self.user_token, 
        )

    def _check_labeling_task_and_export(self, timer):
        """Check for new labels, export to lightning pose format, export database to FileSystem."""

        if self.keypoints is not None:
            # only check task if keypoints attribute has been populated

            # pull data from FileSystem
            inputs=[
                self.filenames["labeled_data_dir"],
                self.filenames["label_studio_metadata"],
            ]

            # check task
            check_labeling_task_and_export(
                label_studio_url=self.label_studio_url,
                proj_dir=self.abspath(self.proj_dir),
                api_key=self.user_token,
                keypoints=self.keypoints,
            )

            # put data to FileSystem
            outputs=[
                self.filenames["collected_data"],
                self.filenames["label_studio_tasks"],
                self.filenames["label_studio_metadata"],
                self.database_dir,  # sqlite database
            ]

        self.check_labels = True

    def _create_labeling_config_xml(self, keypoints):
        """Create a label studio configuration xml file."""

        self.keypoints = keypoints

        if not self.filenames['label_studio_config']:
            # do not execute if filenames have not been updated
            return

        # ---------------------------
        # create new project
        # ---------------------------
        _logger.info("Executing create_labeling_config")

        xml_str = build_xml(self.keypoints)

        proj_dir = self.abspath(self.proj_dir)
        config_file = os.path.join(proj_dir, os.path.basename(self.filenames["label_studio_config"]))
        os.makedirs(proj_dir, exist_ok=True)
        with open(config_file, "wt") as outfile:
            outfile.write(xml_str)

        # ---------------------------
        # put data to FileSystem
        # ---------------------------
        self.put_to_drive([self.filenames["label_studio_config"]])

    def _import_existing_annotations(self, **kwargs):
        """Import annotations into an existing, empty label studio project."""

        if self.counts["import_existing_annotations"] > 0:
            return

        # pull data from FileSystem
        inputs=[
            self.filenames["labeled_data_dir"],
            self.filenames["label_studio_metadata"],
            self.filenames["collected_data"],
            self.filenames["config_file"],
        ]
        self.get_from_drive(inputs)

        # update tasks
        update_tasks(
            label_studio_url=self.label_studio_url, 
            proj_dir=self.abspath(self.proj_dir), 
            api_key=self.user_token,
            config_file=self.abspath(self.filenames['config_file']),
            update_from_csv=True,
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
        _logger.info("SAVING DATA ONE LAST TIME")
        self._check_labeling_task_and_export(timer=0.0)


def create_new_project(
    label_studio_url: str,
    proj_dir: str,
    api_key: str,
    project_name: str,
    label_config_file: str,
) -> None:
    
    from lightning_pose_app.label_studio.utils import start_project
    from lightning_pose_app.label_studio.utils import create_data_source

    _logger.info("Executing create_new_project.py")

    _logger.debug("Connecting to LabelStudio at %s..." % label_studio_url)
    label_studio_client = connect_to_label_studio(url=label_studio_url, api_key=user_token)
    _logger.debug("Connected to LabelStudio at %s" % label_studio_url)

    _logger.info("Creating LabelStudio project...")
    try:
        with open(label_config_file, 'r') as f:
            label_config = f.read()
    except FileNotFoundError:
        _logger.warning(f"Cannot find label studio labeling config at {label_config_file}")
        exit()
    label_studio_project = start_project(
        label_studio_client=label_studio_client,
        title=project_name,
        label_config=label_config,
    )
    _logger.info("LabelStudio project created.")

    # save out project details
    proj_details = {
        "project_name": project_name,
        "id": label_studio_project.id,
        "created_at": str(datetime.datetime.now()),
        "api_key": api_key,
        "n_labeled_tasks": 0,
        "n_total_tasks": 0,
    }
    metadata_file = os.path.join(proj_dir, LABELSTUDIO_METADATA_FILENAME)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    yaml.safe_dump(proj_details, open(metadata_file, "w"))

    # connect label studio to local data source
    _logger.info("Creating LabelStudio data source...")
    json = {  # there are other args to json, but these are the only ones that are required
        "path": proj_dir,
        "project": label_studio_project.id
    }
    create_data_source(label_studio_project=label_studio_project, json=json)
    _logger.info("LabelStudio data source created.")


def update_tasks(
    label_studio_url: str, 
    proj_dir: str, 
    api_key: str, 
    config_file: str = "", 
    update_from_csv: bool = False,
) -> None:

    from lightning_pose_app.label_studio.utils import get_project
    from lightning_pose_app.label_studio.utils import get_rel_image_paths_from_idx_files

    _logger.info("Executing update_tasks.py")

    _logger.debug("Connecting to LabelStudio at %s..." % label_studio_url)
    label_studio_client = connect_to_label_studio(url=label_studio_url, api_key=api_key)
    _logger.debug("Connected to LabelStudio at %s" % label_studio_url)

    # get current project
    metadata_file = os.path.join(proj_dir, LABELSTUDIO_METADATA_FILENAME)
    try:
        metadata = yaml.safe_load(open(metadata_file, "r"))
    except FileNotFoundError:
        _logger.warning(f"Cannot find {metadata_file} in {proj_dir}")
        exit()
    label_studio_project = get_project(label_studio_client=label_studio_client, id=metadata["id"])
    _logger.debug("Fetched Project ID: %s, Project Title: %s" % (
        label_studio_project.id, label_studio_project.title))

    # get tasks that already exist
    existing_tasks = label_studio_project.get_tasks()
    if len(existing_tasks) > 0:
        existing_imgs = [t["data"]["img"] for t in existing_tasks]
    else:
        existing_imgs = []

    _logger.debug("Importing tasks...")
    basedir = os.path.relpath(proj_dir, os.getcwd())
    rel_images = get_rel_image_paths_from_idx_files(proj_dir)
    _logger.debug("relative image paths: {}".format(rel_images))
    label_studio_prefix = f"data/local-files?d={basedir}/"
    # loop over files and add them as dicts to the list, using label studio path format
    # ignore files that are already registered as tasks
    image_list = []
    for r, rel_img in enumerate(rel_images):
        ls_img_path = os.path.join(label_studio_prefix, rel_img)
        if ls_img_path not in existing_imgs:
            image_list.append({"img": ls_img_path})

    label_studio_project.import_tasks(image_list)
    _logger.debug("%i Tasks imported." % len(image_list))

    # add annotations to tasks when importing from another project (e.g. previous DLC project)
    if update_from_csv:

        csv_file = os.path.join(proj_dir, COLLECTED_DATA_FILENAME)
        df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        config = yaml.safe_load(open(config_file, "r"))

        tasks = label_studio_project.get_tasks()
        for task in tasks:
            if len(task["annotations"]) == 0:
                task_id = task["id"]
                rel_img_idx = task["data"]["img"].find("labeled-data")
                rel_img = task["data"]["img"][rel_img_idx:]
                annotation = get_annotation(
                    rel_path=rel_img,
                    labels=df.loc[rel_img].to_frame().reset_index(),
                    dims=config["data"]["image_orig_dims"],
                    task_id=task_id,
                    project_id=label_studio_project.id,
                )
                label_studio_project.create_annotation(task_id=task_id, **annotation)

        # update metadata so app has access to labeling project info
        metadata_file = os.path.join(proj_dir, LABELSTUDIO_METADATA_FILENAME)
        proj_details = yaml.safe_load(open(metadata_file, "r"))
        proj_details["n_labeled_tasks"] = len(image_list)
        proj_details["n_total_tasks"] = len(label_studio_project.get_tasks())
        yaml.safe_dump(proj_details, open(metadata_file, "w"))


def check_labeling_task_and_export(
    label_studio_url: str,
    proj_dir: str,
    api_key: str,
    keypoints: list,
) -> None:
    
    import pickle

    from lightning_pose_app.label_studio.utils import get_project
    from lightning_pose_app.label_studio.utils import LabelStudioJSONProcessor

    _logger.info("Executing check_labeling_task_and_export.py")

    # connect to label studio
    _logger.debug("Connecting to LabelStudio at %s..." % label_studio_url)
    label_studio_client = connect_to_label_studio(url=label_studio_url, api_key=api_key)
    _logger.debug("Connected to LabelStudio at %s" % label_studio_url)

    # get current project
    metadata_file = os.path.join(proj_dir, LABELSTUDIO_METADATA_FILENAME)
    try:
        metadata = yaml.safe_load(open(metadata_file, "r"))
    except FileNotFoundError:
        _logger.warning(f"Cannot find {metadata_file} in {proj_dir}")
        exit()
    label_studio_project = get_project(label_studio_client=label_studio_client, id=metadata["id"])
    print("Fetched Project ID: %s, Project Title: %s" % (
        label_studio_project.id, label_studio_project.title))

    # export the labeled tasks
    _logger.debug("Exporting labeled tasks...")
    exported_tasks = label_studio_project.export_tasks()
    _logger.debug("Exported %i tasks" % len(exported_tasks))

    # use our processor to convert into pandas dlc format
    if len(exported_tasks) > 0:
        # save to pickle for resuming projects
        _logger.debug("Saving tasks to pickle file")
        pickle.dump(exported_tasks, open(os.path.join(proj_dir, LABELSTUDIO_TASKS_FILENAME), "wb"))
        # save to csv for lightning pose models
        _logger.debug("Saving annotations to csv file")
        processor = LabelStudioJSONProcessor(
            label_studio_json_export=exported_tasks,
            data_dir=proj_dir,
            relative_image_dir="",
            keypoint_names=keypoints,
        )
        df = processor()
        df.to_csv(os.path.join(proj_dir, COLLECTED_DATA_FILENAME))

    # update metadata so app has access to labeling project info
    metadata_file = os.path.join(proj_dir, LABELSTUDIO_METADATA_FILENAME)
    proj_details = yaml.safe_load(open(metadata_file, "r"))
    proj_details["n_labeled_tasks"] = len(exported_tasks)
    proj_details["n_total_tasks"] = len(label_studio_project.get_tasks())
    yaml.safe_dump(proj_details, open(metadata_file, "w"))
