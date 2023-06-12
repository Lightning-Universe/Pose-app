"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
import os
import time
import yaml

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.ui.fifty_one import FiftyoneConfigUI
from lightning_pose_app.label_studio.component import LitLabelStudio
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.ui.project import ProjectUI
from lightning_pose_app.ui.streamlit import StreamlitAppLightningPose
from lightning_pose_app.ui.train_infer import TrainUI, LitPose
from lightning_pose_app.build_configs import TensorboardBuildConfig, LitPoseBuildConfig
from lightning_pose_app.build_configs import lightning_pose_dir


class LitPoseApp(LightningFlow):

    def __init__(self):

        # raise Exception("app.py needs to be updated with new components!")
        
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

        # fiftyone tab (flow + work)
        self.fiftyone_ui = FiftyoneConfigUI()

        # streamlit tabs (flow + work)
        self.streamlit_frame = StreamlitAppLightningPose(app_type="frame")
        self.streamlit_video = StreamlitAppLightningPose(app_type="video")

        # tensorboard tab (work)
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
        )

        # label studio (flow + work)
        self.label_studio = LitLabelStudio(
            database_dir=os.path.join(self.data_dir, "labelstudio_db"),
        )

        # works for inference
        self.inference = Dict()

    # @property
    # def ready(self) -> bool:
    #     """Return true once all works have an assigned url"""
    #     return all([
    #         self.fiftyone_ui.work.url != "",
    #         self.streamlit_frame.work.url != "",
    #         self.streamlit_video.work.url != "",
    #         self.train_ui.work.url != "",
    #         self.label_studio.label_studio.url != ""
    #     ])

    def start_tensorboard(self, logdir):
        """run tensorboard"""
        cmd = f"tensorboard --logdir {logdir} --host $host --port $port --reload_interval 30"
        self.tensorboard.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def train_models(self):

        # check to see if we're in demo mode or not
        base_dir = os.path.join(os.getcwd(), self.project_ui.proj_dir[1:])
        cfg_overrides = {
            "data": {
                "data_dir": base_dir,
                "video_dir": os.path.join(base_dir, "videos"),
            },
            "eval": {
                "test_videos_directory": os.path.join(base_dir, "videos"),
                "predict_vids_after_training": True,
            },
            "training": {
                "imgaug": "dlc",
                "max_epochs": self.train_ui.st_max_epochs,
            }
        }

        # list files needed from FileSystem
        inputs = [
            os.path.join(self.project_ui.proj_dir, self.project_ui.config_name),
            os.path.join(self.project_ui.proj_dir, "barObstacleScaling1"),
            os.path.join(self.project_ui.proj_dir, "unlabeled_videos"),
            os.path.join(self.project_ui.proj_dir, "CollectedData_.csv"),
        ]

        # train models
        for m in ["super", "semisuper"]:
            status = self.train_ui.st_train_status[m]
            if status == "initialized" or status == "active":
                self.train_ui.st_train_status[m] = "active"
                outputs = [os.path.join(self.project_ui.model_dir, self.train_ui.st_datetimes[m], "")]
                cfg_overrides["model"] = {"losses_to_use": self.train_ui.st_losses[m]}
                self.train_ui.work.run(
                    action="train", inputs=inputs, outputs=outputs, cfg_overrides=cfg_overrides,
                    results_dir=os.path.join(base_dir, "models", self.train_ui.st_datetimes[m])
                )
                self.train_ui.st_train_status[m] = "complete"
                self.train_ui.work.progress = 0.0
                self.train_ui.progress = 0.0

        self.train_ui.count += 1

    def run_inference(self, model, videos):

        print("--------------------")
        print("RUN INFERENCE HERE!!")
        print(f"model: {model}")
        print("--------------------")
        # launch works
        for video in videos:
            self.inference[video] = LitPose(
                cloud_compute=CloudCompute("gpu"),
                cloud_build_config=LitPoseBuildConfig(),
                parallel=is_running_in_cloud(),
            )
            self.train_ui.run(action="push_video", video_file=video)
            self.inference[video].run('run_inference', model=model, video=video)

        # clean up works
        while len(self.inference) > 0:
            for video in list(self.inference):
                if video in self.inference.keys() and self.inference[video].work_is_done_inference:
                    # kill work
                    print(f"killing work from video {video}")
                    self.inference[video].stop()
                    del self.inference[video]

        print("--------------------")
        print("END OF INFERENCE")
        print("--------------------")

    def update_trained_models_list(self, timer):
        self.project_ui.run(action="update_trained_models_list", timer=timer)
        if self.project_ui.trained_models:
            self.train_ui.trained_models = self.project_ui.trained_models
            self.fiftyone_ui.trained_models = self.project_ui.trained_models

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # don't interfere /w train; since all Works use the same filesystem when running locally,
        # one Work updating the filesystem which is also used by the trainer can corrupt data, etc.
        run_while_training = True
        if not is_running_in_cloud() and self.train_ui.run_script_train:
            run_while_training = False

        # -------------------------------------------------------------
        # update project data
        # -------------------------------------------------------------
        # find previously initialized projects, expose to project UI
        self.project_ui.run(action="find_initialized_projects")

        # find previously constructed fiftyone datasets
        self.fiftyone_ui.run(action="find_fiftyone_datasets")

        # -------------------------------------------------------------
        # start background services (run only once)
        # -------------------------------------------------------------
        self.label_studio.run(action="import_database")
        self.label_studio.run(action="start_label_studio")
        self.fiftyone_ui.run(action="start_fiftyone")
        if self.project_ui.model_dir is not None:
            # find previously trained models for project, expose to training and diagnostics UIs
            self.update_trained_models_list(timer=self.train_ui.count)  # timer to force later runs
            # only launch once we know which project we're working on
            self.start_tensorboard(logdir=self.project_ui.model_dir[1:])
            self.streamlit_frame.run(action="initialize")
            self.streamlit_video.run(action="initialize")

        # -------------------------------------------------------------
        # update project data (user has clicked button in project UI)
        # -------------------------------------------------------------
        if self.project_ui.run_script and run_while_training:
            # update paths now that we know which project we're working with
            self.project_ui.run(action="update_paths")
            self.extract_ui.proj_dir = self.project_ui.proj_dir
            self.train_ui.proj_dir = self.project_ui.proj_dir
            self.streamlit_frame.proj_dir = self.project_ui.proj_dir
            self.streamlit_video.proj_dir = self.project_ui.proj_dir
            self.fiftyone_ui.proj_dir = self.project_ui.proj_dir
            self.fiftyone_ui.config_name = self.project_ui.config_name
            self.label_studio.run(
                action="update_paths",
                proj_dir=self.project_ui.proj_dir, proj_name=self.project_ui.st_project_name)

            # create/load project
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
                    # allow app to advance
                    self.project_ui.count += 1
                    self.project_ui.run_script = False
            else:
                # project already created
                if self.project_ui.count == 0:
                    # load project configuration from config file
                    self.project_ui.run(action="load_project_defaults")
                    # update label studio object
                    self.label_studio.keypoints = self.project_ui.st_keypoints
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
        if self.extract_ui.proj_dir and self.extract_ui.run_script and run_while_training:
            self.extract_ui.run(
                action="extract_frames",
                video_files=self.extract_ui.st_video_files,  # add arg for run caching purposes
            )
            # wait until litpose is done extracting frames, then update tasks
            if len(self.extract_ui.works_dict) == 0:
                self.project_ui.run(action="update_frame_shapes")
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_video_files)
                self.extract_ui.run_script = False

        # -------------------------------------------------------------
        # periodically check labeling task and export new labels
        # -------------------------------------------------------------
        if self.project_ui.count > 0 and run_while_training:
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
                self.train_ui.n_labeled_frames = self.project_ui.n_labeled_frames
                self.train_ui.n_total_frames = self.project_ui.n_total_frames
                self.label_studio.check_labels = False

        # -------------------------------------------------------------
        # train models on ui button press
        # -------------------------------------------------------------
        if self.train_ui.run_script_train:
            self.train_models()
            inputs = [self.project_ui.model_dir]
            # have tensorboard pull the new data
            self.tensorboard.run(
                "null command",
                cwd=os.getcwd(),
                input_output_only=True,  # pull inputs from Drive, but do not run commands
                inputs=inputs,
            )
            # have streamlit pull the new data
            self.streamlit_frame.run(action="pull_models", inputs=inputs)
            self.streamlit_video.run(action="pull_models", inputs=inputs)
            self.project_ui.update_models = True
            self.train_ui.run_script_train = False

        # set the new outputs for UIs
        if self.project_ui.update_models:
            self.project_ui.update_models = False
            self.update_trained_models_list(timer=self.train_ui.count)

        # -------------------------------------------------------------
        # run inference on ui button press (single model, multiple vids)
        # -------------------------------------------------------------
        if self.train_ui.run_script_infer and run_while_training:
            self.run_inference(
                model=self.train_ui.st_inference_model,
                videos=self.train_ui.st_inference_videos,
            )
            self.train_ui.run_script_infer = False

        # -------------------------------------------------------------
        # build fiftyone dataset on button press from FiftyoneUI
        # -------------------------------------------------------------
        if self.fiftyone_ui.run_script:
            self.fiftyone_ui.run(action="build_fiftyone_dataset")
            self.fiftyone_ui.run_script = False

    def configure_layout(self):

        # init tabs
        project_tab = {"name": "Manage Project", "content": self.project_ui}
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        annotate_tab = {"name": "Label Frames", "content": self.label_studio.label_studio}

        # training tabs
        train_tab = {"name": "Train/Infer", "content": self.train_ui}
        train_status_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        st_frame_tab = {"name": "Labeled Diagnostics", "content": self.streamlit_frame.work}
        st_video_tab = {"name": "Video Diagnostics", "content": self.streamlit_video.work}
        fo_prep_tab = {"name": "Prepare Fiftyone", "content": self.fiftyone_ui}
        fo_tab = {"name": "Fiftyone", "content": self.fiftyone_ui.work}

        if self.extract_ui.proj_dir:
            return [
                project_tab,
                extract_tab,
                annotate_tab,
                train_tab,
                train_status_tab,
                st_frame_tab,
                st_video_tab,
                fo_prep_tab,
                fo_tab,
            ]
        else:
            return [
                project_tab,
            ]


app = LightningApp(LitPoseApp())
