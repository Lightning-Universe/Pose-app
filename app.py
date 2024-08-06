"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app app.py

"""

import json
from lightning.app import CloudCompute, LightningApp, LightningFlow
import logging
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time
import yaml

from lightning_pose_app import (
    COLLECTED_DATA_FILENAME,
    LABELED_DATA_DIR,
    LABELSTUDIO_DB_DIR,
    LIGHTNING_POSE_DIR,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
)
from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.label_studio.component import LitLabelStudio
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.ui.project import ProjectUI
from lightning_pose_app.ui.streamlit import StreamlitAppLightningPose
from lightning_pose_app.ui.streamlit_video_viewer import StreamlitVideoViewer
from lightning_pose_app.ui.train_infer import TrainUI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger('APP')


# TODO: HIGH PRIORITY
# - `abort` button next to training/inference progress bars so user doesn't have to kill app

# TODO: LOW PRIORITY
# - launch training in parallel (get this working with `extract_frames` standalone app first)
# - update label studio xml and CollectedData.csv when user inputs new keypoint in project ui


class LitPoseApp(LightningFlow):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # paths
        # -----------------------------
        self.data_dir = "/data"  # # relative to Pose-app root

        # load default config and pass to project manager
        config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # project manager (flow)
        self.project_ui = ProjectUI(
            data_dir=self.data_dir,
            default_config_dict=default_config_dict,
            debug=False,  # if True, hard-code project details like n_views, keypoint_names, etc.
        )

        # extract frames tab (flow + work)
        self.extract_ui = ExtractFramesUI()

        # training tab (flow + work)
        self.train_ui = TrainUI()

        # fiftyone tab (work)
        self.fiftyone = LitBashWork(
            cloud_compute=CloudCompute("default"),
        )

        # streamlit tabs (flow + work)
        self.streamlit_frame = StreamlitAppLightningPose(app_type="frame")
        self.streamlit_video = StreamlitAppLightningPose(app_type="video")
        self.streamlit_video_player = StreamlitVideoViewer()

        # tensorboard tab (work)
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
        )

        # label studio (flow + work)
        self.label_studio = LitLabelStudio(
            database_dir=os.path.join(self.data_dir, LABELSTUDIO_DB_DIR),
        )

        self.import_demo_count = 0

    def import_demo_dataset(self, src_dir_abs, dst_dir_abs):

        """NOTE
        This is an ugly solution. Previously this function was called from the app constructor,
        which required label studio to be started inside the constructor as well. This led to
        issues with ports. Therefore this import needs to happen in the app's run method.
        However, this means that various parts of this function will execute several times before
        it is finished. Furthermore, this function runs *every* time the app is called.
        """

        if self.import_demo_count > 0:
            return

        proj_dir_abs = dst_dir_abs
        project_name = os.path.basename(dst_dir_abs)

        # check to see if the demo dataset has already been imported
        label_studio_exports = os.path.join(
            os.path.dirname(dst_dir_abs), LABELSTUDIO_DB_DIR, "export",
        )
        projects = {}
        if os.path.isdir(label_studio_exports):
            files = os.listdir(label_studio_exports)
            for f in files:
                if f.endswith("info.json"):
                    try:
                        json_file = os.path.join(label_studio_exports, f)
                        d = json.load(open(json_file, "r"))
                        project_name_curr = d["project"]["title"]
                        n_labels_curr = d["project"]["task_number"]
                        projects[project_name_curr] = n_labels_curr
                    except Exception:
                        # sometimes there is a json read error, not sure why
                        continue

        if project_name in projects.keys() and projects[project_name] >= 90:
            self.import_demo_count += 1
            return

        _logger.info("Importing demo dataset; this will only take a minute")

        # -------------------------------
        # copy data
        # -------------------------------
        # copy full example data directory over
        if not os.path.isdir(proj_dir_abs):
            shutil.copytree(src_dir_abs, proj_dir_abs)

        # copy config file
        config_file_dst = os.path.join(proj_dir_abs, f"model_config_{project_name}.yaml")
        if not os.path.isfile(config_file_dst):
            shutil.copyfile(
                os.path.join(
                    LIGHTNING_POSE_DIR, "scripts", "configs", f"config_{project_name}.yaml"
                ),
                config_file_dst,
            )

        # make csv file for label studio
        csv_file_ls = os.path.join(proj_dir_abs, LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME)
        if not os.path.isfile(csv_file_ls):
            n_frames = len(os.listdir(os.path.join(proj_dir_abs, LABELED_DATA_DIR)))
            idxs_selected = np.arange(1, n_frames - 2)  # we've stored mock context frames
            n_digits = 2
            extension = "png"
            frames_to_label = np.sort(np.array(
                ["img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected]
            ))
            np.savetxt(csv_file_ls, frames_to_label, delimiter=",", fmt="%s")

        # make models dir
        os.makedirs(os.path.join(proj_dir_abs, MODELS_DIR), exist_ok=True)

        # -------------------------------
        # remove obstacle keypoints
        # -------------------------------
        config_dict = yaml.safe_load(open(config_file_dst))
        config_dict["data"]["keypoint_names"] = [
            "paw1LH_top",
            "paw2LF_top",
            "paw3RF_top",
            "paw4RH_top",
            "tailBase_top",
            "tailMid_top",
            "nose_top",
            "paw1LH_bot",
            "paw2LF_bot",
            "paw3RF_bot",
            "paw4RH_bot",
            "tailBase_bot",
            "tailMid_bot",
            "nose_bot",
        ]
        config_dict["data"]["columns_for_singleview_pca"] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        ]
        config_dict["data"]["mirrored_column_matches"] = [
            [0, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13],
        ]
        config_dict["losses"]["temporal"]["epsilon"] = 10
        yaml.dump(config_dict, open(config_file_dst, "w"))

        csv_file = os.path.join(proj_dir_abs, COLLECTED_DATA_FILENAME)
        df = pd.read_csv(csv_file, index_col=0, header=[0, 1, 2])
        if 'obs_top' in df.columns.get_level_values(1):
            df.drop('obs_top', axis=1, level=1, inplace=True)
        if 'obsHigh_bot' in df.columns.get_level_values(1):
            df.drop('obsHigh_bot', axis=1, level=1, inplace=True)
        if 'obsLow_bot' in df.columns.get_level_values(1):
            df.drop('obsLow_bot', axis=1, level=1, inplace=True)
        df.to_csv(csv_file)

        # -------------------------------
        # import project to labelstudio
        # -------------------------------
        # create project flow to help with upload
        project_ui_demo = ProjectUI(
            data_dir=self.data_dir,
            default_config_dict=self.project_ui.default_config_dict,
            debug=False,  # if True, hard-code project details like n_views, keypoint_names, etc.
        )
        # update paths
        project_ui_demo.run(action="update_paths", project_name=project_name)
        # load project defaults
        project_ui_demo.run(action="update_project_config")
        # make keypoints field
        keypoint_names = project_ui_demo.config_dict["data"]["keypoint_names"]
        project_ui_demo.run(
            action="update_project_config",
            new_vals_dict={
                "data": {"keypoints": keypoint_names, "num_keypoints": len(keypoint_names)}
            },
        )

        # import to labelstudio
        self.label_studio.run(
            action="update_paths",
            proj_dir=project_ui_demo.proj_dir,
            proj_name=project_name,
        )
        self.label_studio.run(
            action="create_labeling_config_xml",
            keypoints=project_ui_demo.config_dict["data"]["keypoints"],
        )
        self.label_studio.run(action="create_new_project")
        self.label_studio.run(action="import_existing_annotations")

        # -------------------------------
        # cleanup - reset labelstudio
        # -------------------------------
        self.label_studio.proj_dir = None
        self.label_studio.proj_name = None
        self.label_studio.keypoints = None
        for key, val in self.label_studio.filenames.items():
            self.label_studio.filenames[key] = ""
        self.label_studio.counts["create_new_project"] = 0
        self.label_studio.counts["import_existing_annotations"] = 0

        del project_ui_demo

        self.import_demo_count += 1

    def start_tensorboard(self, logdir):
        """run tensorboard"""
        cmd = f"tensorboard --logdir {logdir} --host $host --port $port --reload_interval 30"
        self.tensorboard.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def start_fiftyone(self):
        """run fiftyone"""
        cmd = "fiftyone app launch --address $host --port $port --remote --wait -1"
        self.fiftyone.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def update_trained_models_list(self, timer):
        self.project_ui.run(action="update_trained_models_list", timer=timer)
        if self.project_ui.trained_models:
            self.train_ui.trained_models = self.project_ui.trained_models

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # don't interfere /w train; since all Works use the same filesystem when running locally,
        # one Work updating the filesystem which is also used by the trainer can corrupt data, etc.
        run_while_training = True
        if self.train_ui.run_script_train:
            run_while_training = False

        # don't interfere w/ inference
        run_while_inferring = True
        if self.train_ui.run_script_infer:
            run_while_inferring = False

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # find previously initialized projects, expose to project UI
        self.project_ui.run(action="find_initialized_projects")

        # -------------------------------------------------------------
        # start background services (run only once)
        # -------------------------------------------------------------
        self.label_studio.run(action="start_label_studio")
        self.start_fiftyone()
        if self.project_ui.model_dir is not None:
            # find previously trained models for project, expose to training and diagnostics UIs
            # timer to force later runs
            self.update_trained_models_list(timer=self.train_ui.submit_count_train)
            # only launch once we know which project we're working on, and we're not deleting
            if self.extract_ui.proj_dir and not self.project_ui.st_delete_project:
                self.start_tensorboard(logdir=self.project_ui.model_dir[1:])
                self.streamlit_frame.run(action="initialize")
                self.streamlit_video.run(action="initialize")

        # import mirror-mouse-example dataset
        if not os.environ.get("TESTING_LAI"):
            self.import_demo_dataset(
                src_dir_abs=os.path.join(
                    os.path.dirname(__file__), LIGHTNING_POSE_DIR, "data", "mirror-mouse-example"),
                dst_dir_abs=os.path.join(
                    os.path.dirname(__file__), self.data_dir[1:], "mirror-mouse-example"),
            )

        # -------------------------------------------------------------
        # update project data (user has clicked button in project UI)
        # -------------------------------------------------------------
        if self.project_ui.run_script and run_while_training and run_while_inferring:
            # update paths now that we know which project we're working with
            self.project_ui.run(action="update_paths")
            self.extract_ui.proj_dir = self.project_ui.proj_dir
            self.train_ui.proj_dir = self.project_ui.proj_dir
            self.streamlit_frame.proj_dir = self.project_ui.proj_dir
            self.streamlit_video.proj_dir = self.project_ui.proj_dir
            self.streamlit_video_player.proj_dir = self.project_ui.proj_dir
            self.label_studio.run(
                action="update_paths",
                proj_dir=self.project_ui.proj_dir,
                proj_name=self.project_ui.st_project_name,
            )

            # create/load/delete project
            if self.project_ui.st_create_new_project and self.project_ui.count == 0:
                # create project from scratch
                # load project defaults then overwrite certain fields with user input from app
                self.project_ui.run(action="update_project_config")
                # send params to train ui
                self.train_ui.config_dict = self.project_ui.config_dict
                if self.project_ui.st_keypoints:
                    # if statement here so that we only run "create_new_project" once we have data
                    self.label_studio.run(
                        action="create_labeling_config_xml",
                        keypoints=self.project_ui.st_keypoints)
                    self.label_studio.run(action="create_new_project")
                    # import existing project in another format
                    if self.project_ui.st_upload_existing_project:
                        self.project_ui.run(action="upload_existing_project")
                        self.train_ui.run(action="determine_dataset_type")
                        self.label_studio.run(action="import_existing_annotations")
                        self.project_ui.st_upload_existing_project = False
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
            elif self.project_ui.st_delete_project:
                self.extract_ui.proj_dir = None  # stop tabs from opening
                self.train_ui.proj_dir = None
                self.streamlit_frame.proj_dir = None
                self.streamlit_video.proj_dir = None
                self.streamlit_video_player.proj_dir = None
                self.label_studio.run(action="delete_project")
                self.project_ui.run(action="delete_project")
                self.project_ui.run_script = False
            else:
                # project already created
                # figure out if this is a context dataset; won't expose option to user otherwise
                self.train_ui.run(action="determine_dataset_type")
                if self.project_ui.count == 0:
                    # load project configuration from config file
                    self.project_ui.run(action="load_project_defaults")
                    self.train_ui.config_dict = self.project_ui.config_dict
                    # update label studio object
                    self.label_studio.keypoints = self.project_ui.st_keypoints
                    # count labeled frames
                    self.project_ui.run(
                        action="compute_labeled_frame_fraction", timer=self.label_studio.time)
                    self.train_ui.n_labeled_frames = self.project_ui.n_labeled_frames
                    self.train_ui.n_total_frames = self.project_ui.n_total_frames
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
                else:
                    # update project
                    self.project_ui.run(action="update_project_config")
                    # send params to train ui
                    self.train_ui.config_dict = self.project_ui.config_dict
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False

        # -------------------------------------------------------------
        # extract frames for labeling
        # -------------------------------------------------------------
        if self.extract_ui.proj_dir and self.extract_ui.run_script_video_random:
            self.extract_ui.run(
                action="extract_frames",
                video_files=self.extract_ui.st_video_files,  # add arg for run caching purposes
            )
            # wait until frame extraction is complete, then update label studio tasks
            if self.extract_ui.work_is_done_extract_frames:
                self.project_ui.run(action="update_frame_shapes")
                # hack; for some reason the app won't advance past the ls run
                self.extract_ui.run_script_video_random = False
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_video_files)
                self.extract_ui.run_script_video_random = False

        if self.extract_ui.proj_dir and self.extract_ui.run_script_zipped_frames:
            self.extract_ui.run(
                action="unzip_frames",
                video_files=self.extract_ui.st_frame_files,  # add arg for run caching purposes
            )
            # wait until frame extraction is complete, then update label studio tasks
            if self.extract_ui.work_is_done_extract_frames:
                self.project_ui.run(action="update_frame_shapes")
                # hack; for some reason the app won't advance past the ls run
                self.extract_ui.run_script_zipped_frames = False
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_frame_files)
                self.extract_ui.run_script_zipped_frames = False

        if self.extract_ui.proj_dir and self.extract_ui.run_script_video_model:
            self.extract_ui.run(
                action="extract_frames_using_model",
                video_files=self.extract_ui.st_video_files,  # add arg for run caching purposes
            )
            # wait until frame extraction is complete, then update label studio tasks
            if self.extract_ui.work_is_done_extract_frames:
                self.project_ui.run(action="update_frame_shapes")
                # hack; for some reason the app won't advance past the ls run
                self.extract_ui.run_script_video_model = False
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_video_files)
                self.extract_ui.run_script_video_model = False

        if self.extract_ui.proj_dir and self.extract_ui.run_script_check_labels:
            self.extract_ui.run(
                action="save_annotated_frames",
                selected_body_parts=self.extract_ui.selected_body_parts
            )
            self.extract_ui.run_script_check_labels = False

        # -------------------------------------------------------------
        # periodically check labeling task and export new labels
        # -------------------------------------------------------------
        if self.project_ui.count > 0 and run_while_training and run_while_inferring:
            # with large datasets, the check_labeling_task_and_export can take >15 s
            if self.project_ui.n_labeled_frames is None or self.project_ui.n_labeled_frames < 1000:
                if self.project_ui.n_labeled_frames is None or self.project_ui.n_total_frames < 500:
                    t_elapsed = 15  # seconds
                else:
                    t_elapsed = 30
                t_elapsed_list = ",".join([str(v) for v in range(0, 60, t_elapsed)])
                if self.schedule(f"* * * * * {t_elapsed_list}"):
                    # only true for a single flow execution every n seconds; capture event in state var
                    self.label_studio.check_labels = True
                    self.label_studio.time = time.time()
            else:  # assume 5k frames/minute for label studio updates (conservative, ~15k/minute in reality)
                t_elapsed = int(np.ceil(self.project_ui.n_total_frames / 5000))
                if self.schedule(f"*/{t_elapsed} * * * *"):
                    # only true for a single flow execution every n seconds; capture event in state var
                    self.label_studio.check_labels = True
                    self.label_studio.time = time.time()

            if self.label_studio.check_labels:
                self.label_studio.run(
                    action="check_labeling_task_and_export", timer=self.label_studio.time)
                self.project_ui.run(
                    action="compute_labeled_frame_fraction", timer=self.label_studio.time)
                self.train_ui.n_labeled_frames = self.project_ui.n_labeled_frames
                self.train_ui.n_total_frames = self.project_ui.n_total_frames
                self.label_studio.check_labels = False

        # -------------------------------------------------------------
        # train models on ui button press
        # -------------------------------------------------------------
        if self.train_ui.run_script_train and run_while_inferring:
            self.train_ui.run(action="train", config_filename=self.project_ui.config_name)
            self.project_ui.update_models = True
            self.train_ui.run_script_train = False

        # set the new outputs for UIs
        if self.project_ui.update_models:
            self.project_ui.update_models = False
            self.update_trained_models_list(timer=self.train_ui.submit_count_train)

        # -------------------------------------------------------------
        # run inference on ui button press (single model, multiple vids)
        # -------------------------------------------------------------
        if self.train_ui.run_script_infer and run_while_training:
            self.train_ui.run(
                action="run_inference",
                video_files=self.train_ui.st_inference_videos,  # add arg for run caching purposes
            )
            self.train_ui.run_script_infer = False

    def configure_layout(self):

        # init tabs
        project_tab = {"name": "Manage Project", "content": self.project_ui}
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        annotate_tab = {"name": "Label Frames", "content": self.label_studio.label_studio}

        # training tabs
        train_tab = {"name": "Train/Infer", "content": self.train_ui}
        train_status_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        st_frame_tab = {"name": "Image Diagnostics", "content": self.streamlit_frame.work}
        st_video_tab = {"name": "Video Diagnostics", "content": self.streamlit_video.work}
        st_video_player_tab = {"name": "Video Player", "content": self.streamlit_video_player}
        fo_tab = {"name": "Image Viewer", "content": self.fiftyone}

        if self.extract_ui.proj_dir and not self.project_ui.st_delete_project:
            return [
                project_tab,
                extract_tab,
                annotate_tab,
                train_tab,
                train_status_tab,
                st_frame_tab,
                fo_tab,
                st_video_tab,
                st_video_player_tab,
            ]
        else:
            return [
                project_tab,
            ]


app = LightningApp(LitPoseApp())
