"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
import os
import time
import yaml

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.litpose import LitPose, lightning_pose_dir
from lightning_pose_app.label_studio.component import LitLabelStudio
from lightning_pose_app.ui.diagnostics import DiagnosticsUI
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.ui.landing import LandingUI
from lightning_pose_app.ui.project import ProjectUI, ProjectDataIO
from lightning_pose_app.ui.train import TrainDemoUI
from lightning_pose_app.utils.build_configs import TensorboardBuildConfig

ON_CLOUD = False  # set False when debugging locally, weird things can happen w/ shared filesystem


class LitPoseApp(LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps
        drive_name = "lit://lpa"

        # -----------------------------
        # paths
        # -----------------------------
        config_dir = os.path.join(lightning_pose_dir, "scripts", "configs")
        data_dir = "data"  # relative to self.drive

        # load default config and pass to project manager
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # UIs
        # -----------------------------
        # landing tab
        self.landing_ui = LandingUI()

        # project manager tab
        self.project_io = ProjectDataIO(
            drive_name=drive_name,
            data_dir=data_dir,
            default_config_dict=default_config_dict,
        )
        self.project_ui = ProjectUI(data_dir=data_dir)

        # extract frames tab
        self.extract_ui = ExtractFramesUI(
            drive_name=drive_name,
            script_dir=lightning_pose_dir,
            script_name="scripts/extract_frames.py",
            script_args="",
        )

        # training tab; this will default to the toy dataset, unless a project is created/loaded
        self.train_ui = TrainDemoUI(
            script_dir=lightning_pose_dir,
            script_name="scripts/train_hydra.py",
            script_args="",
            script_env="HYDRA_FULL_ERROR=1",
            max_epochs=200,
        )

        # diagnostics tab
        # self.diagnostics_ui = DiagnosticsUI(
        #     script_dir=lightning_pose_dir,
        #     script_name="scripts/create_fiftyone_dataset.py",
        #     script_env="HYDRA_FULL_ERROR=1",
        #     script_args="""
        #         eval.fiftyone.dataset_name=test1
        #         eval.fiftyone.model_display_names=["test1"]
        #         eval.fiftyone.dataset_to_create="images"
        #         eval.fiftyone.build_speed="fast"
        #         eval.fiftyone.launch_app_from_script=False
        #     """
        # )

        # -----------------------------
        # workers
        # -----------------------------
        # tensorboard
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
            drive_name=drive_name,
            component_name="tensorboard",
        )

        # lightning pose: frame extraction and model training
        self.litpose = LitPose(
            cloud_compute=CloudCompute("gpu"),
            drive_name=drive_name,
        )

        # label studio
        self.label_studio = LitLabelStudio(
            cloud_compute=CloudCompute("default"),
            drive_name=drive_name,
            database_dir=os.path.join(data_dir, "labelstudio_db"),
        )

        # streamlit labeled
        # self.my_streamlit_frame = LitBashWork(
        #     cloud_compute=CloudCompute("default"),
        #     cloud_build_config=StreamlitBuildConfig(),  # this may not be necessary
        #     drive_name=drive_name,
        # )

        # streamlit video
        # self.my_streamlit_video = LitBashWork(
        #     cloud_compute=CloudCompute("default"),
        #     cloud_build_config=StreamlitBuildConfig(),  # this may not be necessary
        #     drive_name=drive_name,
        # )

    # @property
    # def ready(self) -> bool:
    #     """Return true once all works have an assigned url"""
    #     return all([
    #         self.my_tb.url != "",
    #         self.my_work.url != "",
    #         self.label_studio.label_studio.url != ""])

    def start_tensorboard(self, logdir):
        """run tensorboard"""
        cmd = f"tensorboard --logdir {logdir} --host $host --port $port --reload_interval 30"
        self.tensorboard.run(cmd, wait_for_exit=False, cwd=os.getcwd())

    def train_models(self):

        # check to see if we're in demo mode or not
        base_dir = os.path.join(os.getcwd(), self.project_io.proj_dir)
        if self.project_io.config_name is not None:
            config_cmd = f" --config-path={base_dir}" \
                         f" --config-name={self.project_io.config_name}" \
                         f" data.data_dir={base_dir}" \
                         f" data.video_dir={os.path.join(base_dir, 'videos')}" \
                         f" eval.test_videos_directory={os.path.join(base_dir, 'videos')}"
        else:
            config_cmd = \
                " eval.test_videos_directory=toy_datasets/toymouseRunningData/unlabeled_videos"

        # list files needed from Drive
        inputs = [
            os.path.join(self.project_io.proj_dir, self.project_io.config_name),
            os.path.join(self.project_io.proj_dir, "labeled-data"),
            os.path.join(self.project_io.proj_dir, "CollectedData.csv"),
        ]
        outputs = [os.path.join(self.project_io.proj_dir, "models")]

        # train supervised model
        if self.train_ui.st_train_super \
                and not self.train_ui.st_train_complete_flag["super"]:
            hydra_srun = os.path.join(
                base_dir, "models", self.train_ui.st_datetimes["super"])
            hydra_mrun = os.path.join(
                base_dir, "models/multirun", self.train_ui.st_datetimes["super"])
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["super"] \
                  + f" hydra.run.dir={hydra_srun}" \
                  + f" hydra.sweep.dir={hydra_mrun}"
            self.litpose.work.run(
                cmd,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                inputs=inputs,
                outputs=outputs,
            )
            self.train_ui.st_train_complete_flag["super"] = True

        # train semi-supervised model
        if self.train_ui.st_train_semisuper \
                and not self.train_ui.st_train_complete_flag["semisuper"]:
            hydra_srun = os.path.join(
                base_dir, "models", self.train_ui.st_datetimes["semisuper"])
            hydra_mrun = os.path.join(
                base_dir, "models/multirun", self.train_ui.st_datetimes["semisuper"])
            cmd = "python" \
                  + " " + self.train_ui.st_script_name \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["semisuper"] \
                  + f" hydra.run.dir={hydra_srun}" \
                  + f" hydra.sweep.dir={hydra_mrun}"
            self.litpose.work.run(
                cmd,
                env=self.train_ui.st_script_env,
                cwd=self.train_ui.st_script_dir,
                inputs=inputs,
                outputs=outputs,
            )
            self.train_ui.st_train_complete_flag["semisuper"] = True

        # have TB pull the new data
        # input_output_only=True means that we'll pull inputs from drive, but not run commands
        cmd = "null command"  # make this unique
        self.tensorboard.run(
            cmd,
            cwd=os.getcwd(),
            input_output_only=True,
            inputs=outputs,
        )

        # set the new outputs for UIs
        self.litpose.run(action="init_lp_outputs_to_uis", search_dir=outputs[0])

    def start_fiftyone_dataset_creation(self):

        cmd = "python" \
              + " " + self.diagnostics_ui.st_script_name \
              + " " + self.diagnostics_ui.st_script_args \
              + " " + self.diagnostics_ui.script_args_append \
              + " " + "eval.fiftyone.dataset_to_create=images"
        self.litpose.work.run(
            cmd,
            env=self.diagnostics_ui.st_script_env,
            cwd=self.diagnostics_ui.st_script_dir,
          )

        # add dataset name to list for user to see
        self.diagnostics_ui.add_fo_dataset(self.diagnostics_ui.st_dataset_name)

    def start_st_frame(self):
        """run streamlit for labeled frames"""

        # set labeled csv
        # TODO: extract this directly from hydra
        csv_file = os.path.join(
            lightning_pose_dir, "toy_datasets/toymouseRunningData/CollectedData_.csv")
        labeled_csv_args = f"--labels_csv={csv_file}"

        # set prediction files (hard code some paths for now)
        prediction_file_args = ""
        for model_dir in self.diagnostics_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.diagnostics_ui.outputs_dir, model_dir, "predictions.csv")
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.diagnostics_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.diagnostics_ui.outputs_dir, self.diagnostics_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/labeled_frame_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + labeled_csv_args \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_frame.run(cmd, wait_for_exit=False, cwd=".")

    def start_st_video(self):
        """run streamlit for videos"""

        # set prediction files (hard code some paths for now)
        # select random video
        prediction_file_args = ""
        video_file = "test_vid.csv"  # TODO: find random vid in predictions directory
        for model_dir in self.diagnostics_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.diagnostics_ui.outputs_dir, model_dir, "video_preds", video_file)
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.diagnostics_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.diagnostics_ui.outputs_dir, self.diagnostics_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/video_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_video.run(cmd, wait_for_exit=False, cwd=".")

    def run(self):

        # for unit testing purposes
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Pose App! ⚡")

        # don't interfere /w train; since all Works use the same filesystem when running locally,
        # one Work updating the filesystem which is also used by the trainer can corrupt data, etc.
        run_while_training = True
        if not ON_CLOUD and self.train_ui.run_script:
            run_while_training = False

        # -----------------------------
        # init UIs (find prev artifacts)
        # -----------------------------
        # find previously initialized projects, expose to project UI
        self.project_io.run(action="find_initialized_projects")
        self.project_ui.initialized_projects = self.project_io.initialized_projects

        # find previously trained models, expose to training and diagnostics UIs
        self.litpose.run(action="init_lp_outputs_to_uis", search_dir=self.project_io.model_dir)
        if self.litpose.lp_outputs:
            self.train_ui.set_hydra_outputs(self.litpose.lp_outputs)
            # self.diagnostics_ui.set_hydra_outputs(self.litpose.lp_outputs)

        # find previously constructed fiftyone datasets, expose to fiftyone UI
        # self.litpose.run(action="init_fiftyone_outputs_to_ui")
        # if self.litpose.fiftyone_datasets:
        #     self.diagnostics_ui.set_fo_dataset(self.litpose.fiftyone_datasets)

        # -----------------------------
        # init background services once
        # -----------------------------
        self.label_studio.run(action="import_database")
        self.label_studio.run(action="start_label_studio")
        if self.project_io.model_dir is not None:
            # only launch once we know which project we're working on
            self.start_tensorboard(logdir=self.project_io.model_dir)
        # self.litpose.run(action="start_fiftyone")

        # -----------------------------
        # run works
        # -----------------------------
        # update paths if we know which project we're working with
        self.project_io.run(
            action="update_paths", project_name=self.project_ui.st_project_name)
        # load project configuration from defaults
        self.project_io.run(
            action="load_project_defaults", new_vals_dict=self.project_ui.st_new_vals)
        # copy project defaults into UI
        for key, val in self.project_io.proj_defaults.items():
            setattr(self.project_ui, key, val)

        # update project configuration when user clicks button in project UI
        if self.project_ui.run_script and run_while_training:
            # update user-supplied parameters in config yaml file
            self.project_io.run(
                action="update_project_config", new_vals_dict=self.project_ui.st_new_vals)

            # update data dir for frame extraction object
            self.extract_ui.proj_dir = self.project_io.proj_dir
            # update label studio object paths
            self.label_studio.run(
                action="update_paths",
                proj_dir=self.project_io.proj_dir, proj_name=self.project_ui.st_project_name)

            # create label studio xml file, create/load project
            if self.project_ui.create_new_project and self.project_ui.count == 0:
                self.label_studio.run(
                    action="create_labeling_config_xml", keypoints=self.project_ui.keypoints)
                self.label_studio.run(action="create_new_project")
            else:
                # project already created; update label studio object
                self.label_studio.keypoints = self.project_ui.keypoints

            self.project_ui.count += 1
            self.project_ui.run_script = False

        # extract frames for labeling from uploaded videos
        if self.extract_ui.proj_dir and self.extract_ui.run_script and run_while_training:
            self.litpose.run(
                action="start_extract_frames",
                video_files=self.extract_ui.st_video_files,
                proj_dir=self.extract_ui.proj_dir,
                script_name=self.extract_ui.script_name,
                n_frames_per_video=self.extract_ui.st_n_frames_per_video,
            )
            # wait until litpose is done extracting frames, then update tasks
            if self.litpose.work_is_done_extract_frames:
                self.project_io.run(action="update_frame_shapes")
                self.label_studio.run(action="update_tasks", videos=self.extract_ui.st_video_files)
                self.litpose.work_is_done_extract_frames = False
                self.extract_ui.run_script = False

        # check labeling task and export new labels
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
                self.project_io.run(
                    action="compute_labeled_frame_fraction", timer=self.label_studio.time)
                self.train_ui.n_labeled_frames = self.project_io.n_labeled_frames
                self.train_ui.n_total_frames = self.project_io.n_total_frames
                self.label_studio.check_labels = False

        # train on ui button press
        if self.train_ui.run_script:
            self.train_models()
            self.train_ui.run_script = False

        # initialize diagnostics on button press
        # if self.fo_ui.run_script:
        #     self.fo_ui.run_script = False
        #     self.start_fiftyone_dataset_creation()
        #     # self.start_st_frame()
        #     # self.start_st_video()

    def configure_layout(self):

        # init tabs
        landing_tab = {"name": "Home", "content": self.landing_ui}
        project_tab = {"name": "Manage Project", "content": self.project_ui}
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        annotate_tab = {"name": "Label Frames", "content": self.label_studio.label_studio}

        # training tabs
        train_demo_tab = {"name": "Train", "content": self.train_ui}
        train_diag_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        # fo_prep_tab = {"name": "Prepare Diagnostics", "content": self.fo_ui}
        fo_tab = {"name": "Labeled Preds", "content": self.litpose.work}
        # st_frame_tab = {"name": "Labeled Diagnostics", "content": self.my_streamlit_frame}
        # st_video_tab = {"name": "Video Diagnostics", "content": self.my_streamlit_video}

        if self.landing_ui.st_mode == "demo":
            return [
                landing_tab,
                train_demo_tab,
                train_diag_tab,
                # fo_prep_tab,
                fo_tab,
                # st_frame_tab,
                # st_video_tab,
            ]

        elif self.landing_ui.st_mode == "project":
            if not self.extract_ui.proj_dir:
                # need to create/load new project before moving on to other tabs
                return [
                    landing_tab,
                    project_tab,
                ]
            else:
                # show all tabs
                return [
                    landing_tab,
                    project_tab,
                    extract_tab,
                    annotate_tab,
                    train_demo_tab,
                    train_diag_tab,
                    # fo_prep_tab,
                    # fo_tab,
                ]

        else:
            return [landing_tab]


app = LightningApp(LitPoseApp())
