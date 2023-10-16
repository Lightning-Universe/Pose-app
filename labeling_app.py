"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app labeling_app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
from lightning.app.structures import Dict
import logging
import os
import time
import yaml

from lightning_pose_app import LABELSTUDIO_DB_DIR
from lightning_pose_app.label_studio.component import LitLabelStudio
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.ui.project import ProjectUI
from lightning_pose_app.build_configs import lightning_pose_dir


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('APP')


class LitPoseApp(LightningFlow):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # paths
        # -----------------------------
        self.data_dir = "/data"  # relative to FileSystem root

        # load default config and pass to project manager
        config_dir = os.path.join(lightning_pose_dir, "scripts", "configs")
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # project manager tab (flow)
        self.project_ui = ProjectUI(
            data_dir=self.data_dir,
            default_config_dict=default_config_dict,
            debug=False,  # if True, hard-code project details like n_views, keypoint_names, etc.
        )

        # extract frames tab (flow + work)
        self.extract_ui = ExtractFramesUI()

        # label studio (flow + work)
        self.label_studio = LitLabelStudio(
            database_dir=os.path.join(self.data_dir, LABELSTUDIO_DB_DIR),
        )

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # find previously initialized projects, expose to project UI
        self.project_ui.run(action="find_initialized_projects")

        # -------------------------------------------------------------
        # init label studio; this will only happen once
        # -------------------------------------------------------------
        self.label_studio.run(action="import_database")
        self.label_studio.run(action="start_label_studio")

        # -------------------------------------------------------------
        # update project data (user has clicked button in project UI)
        # -------------------------------------------------------------
        if self.project_ui.run_script:
            # update paths now that we know which project we're working with
            self.project_ui.run(action="update_paths")
            self.extract_ui.proj_dir = self.project_ui.proj_dir
            self.label_studio.run(
                action="update_paths",
                proj_dir=self.project_ui.proj_dir, proj_name=self.project_ui.st_project_name)

            # create/load/delete project
            if self.project_ui.st_create_new_project and self.project_ui.count == 0:
                # create project from scratch
                # load project defaults then overwrite certain fields with user input from app
                self.project_ui.run(action="update_project_config")
                if self.project_ui.st_keypoints:
                    # if statement here so that we only run "create_new_project" once we have data
                    self.label_studio.run(
                        action="create_labeling_config_xml",
                        keypoints=self.project_ui.st_keypoints)
                    self.label_studio.run(action="create_new_project")
                    # import existing project in another format
                    if self.project_ui.st_upload_existing_project:
                        self.project_ui.run(action="upload_existing_project")
                        self.label_studio.run(action="import_existing_annotations")
                        self.project_ui.st_upload_existing_project = False
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
            elif self.project_ui.st_delete_project:
                self.extract_ui.proj_dir = None  # stop tabs from opening
                self.project_ui.run(action="delete_project")
                self.project_ui.run_script = False
            else:
                # project already created
                if self.project_ui.count == 0:
                    # load project configuration from config file
                    self.project_ui.run(action="load_project_defaults")
                    # update label studio object
                    # self.label_studio.keypoints = self.project_ui.st_keypoints
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
                else:
                    # update project
                    self.project_ui.run(action="update_project_config")
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False

        # -------------------------------------------------------------
        # extract frames for labeling from uploaded videos
        # -------------------------------------------------------------
        if self.extract_ui.proj_dir and self.extract_ui.run_script:
            self.extract_ui.run(
                action="extract_frames",
                video_files=self.extract_ui.st_video_files,  # add arg for run caching purposes
            )
            # wait until frame extraction is complete, then update label studio tasks
            if self.extract_ui.work_is_done_extract_frames:
                self.project_ui.run(action="update_frame_shapes")
                self.extract_ui.run_script = False  # hack, app won't advance past ls run
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_video_files)
                self.extract_ui.run_script = False

        # -------------------------------------------------------------
        # periodically check labeling task and export new labels
        # -------------------------------------------------------------
        if self.project_ui.count > 0:
            t_elapsed = 15  # seconds
            t_elapsed_list = ",".join([str(v) for v in range(0, 60, t_elapsed)])
            if self.schedule(f"* * * * * {t_elapsed_list}"):
                # only true for a single flow execution every n seconds; capture event in state var
                self.label_studio.check_labels = True
                self.label_studio.time = time.time()
            if self.label_studio.check_labels:
                self.label_studio.run(
                    action="check_labeling_task_and_export", timer=self.label_studio.time)
                self.project_ui.run(
                    action="compute_labeled_frame_fraction", timer=self.label_studio.time)
                self.label_studio.check_labels = False

    def configure_layout(self):

        project_tab = {"name": "Manage Project", "content": self.project_ui}
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        annotate_tab = {"name": "Label Frames", "content": self.label_studio.label_studio}

        if self.extract_ui.proj_dir:
            return [
                project_tab,
                extract_tab,
                annotate_tab,
            ]
        else:
            return [
                project_tab,
            ]


app = LightningApp(LitPoseApp())
