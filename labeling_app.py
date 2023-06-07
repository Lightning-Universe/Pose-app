"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app labeling_app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
from lightning.app.structures import Dict
import os
import time
import yaml

from lightning_pose_app.label_studio.component import LitLabelStudio
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.ui.project import ProjectUI, ProjectDataIO
from lightning_pose_app.build_configs import lightning_pose_dir


# TODO
# - update label studio xml and CollectedData.csv when user inputs new keypoint in project ui


class LitPoseApp(LightningFlow):

    def __init__(self):

        super().__init__()

        # shared data for apps; NOTE: this is hard-coded in the run_inference method below too
        drive_name = "lit://lpa"

        # -----------------------------
        # paths
        # -----------------------------
        config_dir = os.path.join(lightning_pose_dir, "scripts", "configs")
        self.data_dir = "data"  # relative to self.drive

        # load default config and pass to project manager
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # project manager (work) and tab (flow)
        self.project_io = ProjectDataIO(
            drive_name=drive_name,
            data_dir=self.data_dir,
            default_config_dict=default_config_dict,
        )
        self.project_ui = ProjectUI(
            data_dir=self.data_dir,
            debug=True,  # if True, hard-code project details like n_views, keypoint_names, etc.
        )
        for key, val in self.project_io.proj_defaults.items():
            setattr(self.project_ui, key, val)

        # extract frames tab (flow)
        self.extract_ui = ExtractFramesUI(drive_name=drive_name)

        # label studio (flow + work)
        self.label_studio = LitLabelStudio(
            cloud_compute=CloudCompute("default"),
            drive_name=drive_name,
            database_dir=os.path.join(self.data_dir, "labelstudio_db"),
        )

    def extract_frames(self, video_files, proj_dir, n_frames_per_video):
        for video_file in video_files:
            status = self.extract_ui.st_extract_status[video_file]
            if status == "initialized" or status == "active":
                self.extract_ui.st_extract_status[video_file] = "active"
                print(self.extract_ui.work.progress)
                self.extract_ui.work.run(
                    action="extract_frames",
                    video_file=video_file,
                    proj_dir=proj_dir,
                    n_frames_per_video=n_frames_per_video,
                )
                self.extract_ui.st_extract_status[video_file] = "complete"
                self.extract_ui.work.progress = 0.0

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # find previously initialized projects, expose to project UI
        self.project_io.run(action="find_initialized_projects")
        self.project_ui.initialized_projects = self.project_io.initialized_projects

        # -------------------------------------------------------------
        # init label studio
        # -------------------------------------------------------------
        self.label_studio.run(action="import_database")
        self.label_studio.run(action="start_label_studio")

        # -------------------------------------------------------------
        # update project data (user has clicked button in project UI)
        # -------------------------------------------------------------
        if self.project_ui.run_script:
            # update paths now that we know which project we're working with
            self.project_io.run(
                action="update_paths", project_name=self.project_ui.st_project_name)
            self.extract_ui.proj_dir = self.project_io.proj_dir
            self.label_studio.run(
                action="update_paths",
                proj_dir=self.project_io.proj_dir, proj_name=self.project_ui.st_project_name)

            # create/load project
            if self.project_ui.create_new_project and self.project_ui.count == 0:
                # create project from scratch
                self.project_io.run(
                    action="update_project_config", new_vals_dict=self.project_ui.st_new_vals)
                if self.project_ui.st_keypoints:
                    # if statement here so that we only run "create_new_project" once we have data
                    self.label_studio.run(
                        action="create_labeling_config_xml",
                        keypoints=self.project_ui.st_keypoints)
                    self.label_studio.run(action="create_new_project")
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
            else:
                # project already created
                if self.project_ui.count == 0:
                    # load project configuration from config file
                    self.project_io.run(
                        action="load_project_defaults", new_vals_dict=self.project_ui.st_new_vals)
                    # copy project defaults into UI
                    for key, val in self.project_io.proj_defaults.items():
                        setattr(self.project_ui, key, val)
                    # update label studio object
                    self.label_studio.keypoints = self.project_ui.st_keypoints
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
                else:
                    # update project
                    self.project_io.run(
                        action="update_project_config", new_vals_dict=self.project_ui.st_new_vals)
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False

        # -------------------------------------------------------------
        # extract frames for labeling from uploaded videos
        # -------------------------------------------------------------
        if self.extract_ui.proj_dir and self.extract_ui.run_script:
            self.extract_frames(
                video_files=self.extract_ui.st_video_files,
                proj_dir=self.extract_ui.proj_dir,
                n_frames_per_video=self.extract_ui.st_n_frames_per_video,
            )
            # wait until litpose is done extracting frames, then update tasks
            if self.extract_ui.work.work_is_done_extract_frames:
                self.project_io.run(action="update_frame_shapes")
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
                self.project_io.run(
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
