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
from lightning_pose_app.diagnostics import DiagnosticsUI
from lightning_pose_app.litpose import LitPose
from lightning_pose_app.ui.landing import LandingUI
from lightning_pose_app.ui.project import ProjectDataIO
from lightning_pose_app.ui.train_infer import TrainUI
from lightning_pose_app.build_configs import TensorboardBuildConfig, lightning_pose_dir


# TODO
# - project_io: simplify `load_project_defaults`, set defaults in __init__
# - launch training in parallel (get this working with `extract_frames` standalone app first)
# - figure out what to do about inference
# - figure out what to do with landing tab/current markdown


class LitPoseApp(LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps; NOTE: this is hard-coded in the run_inference method below too
        drive_name = "lit://lpa"

        # -----------------------------
        # paths
        # -----------------------------
        config_dir = os.path.join(lightning_pose_dir, "scripts", "configs")
        data_dir = "data"  # relative to self.drive

        # load default config and pass to project manager
        default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

        # -----------------------------
        # flows and works
        # -----------------------------
        # landing tab (flow)
        # self.landing_ui = LandingUI()

        # project manager (work) and tab (flow)
        self.project_io = ProjectDataIO(
            drive_name=drive_name,
            data_dir=data_dir,
            default_config_dict=default_config_dict,
        )
        # self.project_ui = ProjectUI(data_dir=data_dir)

        # training tab (flow)
        self.train_ui = TrainUI(drive_name=drive_name)

        # diagnostics tab (flow + work)
        self.diagnostics_ui = DiagnosticsUI(drive_name=drive_name)

        # tensorboard tab (work)
        self.tensorboard = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=TensorboardBuildConfig(),
            drive_name=drive_name,
            component_name="tensorboard",
        )

        # lightning pose: work for frame extraction and model training
        self.litpose = LitPose(
            cloud_compute=CloudCompute("gpu"),
            drive_name=drive_name,
        )

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
        outputs = [self.project_io.model_dir]

        # train supervised model
        if self.train_ui.st_train_super \
                and not self.train_ui.st_train_complete_flag["super"]:
            hydra_srun = os.path.join(
                base_dir, "models", self.train_ui.st_datetimes["super"])
            hydra_mrun = os.path.join(
                base_dir, "models/multirun", self.train_ui.st_datetimes["super"])
            cmd = "python scripts/train_hydra.py" \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["super"] \
                  + f" hydra.run.dir={hydra_srun}" \
                  + f" hydra.sweep.dir={hydra_mrun}"
            self.litpose.work.run(
                cmd,
                cwd=lightning_pose_dir,
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
            cmd = "python scripts/train_hydra.py" \
                  + config_cmd \
                  + " " + self.train_ui.st_script_args["semisuper"] \
                  + f" hydra.run.dir={hydra_srun}" \
                  + f" hydra.sweep.dir={hydra_mrun}"
            self.litpose.work.run(
                cmd,
                cwd=lightning_pose_dir,
                inputs=inputs,
                outputs=outputs,
            )
            self.train_ui.st_train_complete_flag["semisuper"] = True

        self.train_ui.count += 1

    def update_trained_models_list(self, timer):
        self.project_io.run(action="update_trained_models_list", timer=timer)
        if self.project_io.trained_models:
            self.train_ui.trained_models = self.project_io.trained_models
            self.diagnostics_ui.trained_models = self.project_io.trained_models

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
        # init UIs (find prev artifacts)
        # -------------------------------------------------------------
        # find previously trained models for project, expose to training and diagnostics UIs
        self.update_trained_models_list(timer=self.train_ui.count)  # timer is to force later runs

        # find previously constructed fiftyone datasets
        self.diagnostics_ui.run(action="find_fiftyone_datasets")

        # -------------------------------------------------------------
        # init background services once
        # -------------------------------------------------------------
        self.diagnostics_ui.run(action="start_fiftyone")
        if self.project_io.model_dir is not None:
            # only launch once we know which project we're working on
            self.start_tensorboard(logdir=self.project_io.model_dir)

        # -------------------------------------------------------------
        # update project data from toy dataset
        # -------------------------------------------------------------
        # update paths if we know which project we're working with
        self.project_io.run(action="update_paths", project_name="demo")

        # here we copy the toy dataset config file that comes packaged with the lightning-pose repo and
        # move it to a new directory that is consistent with the project structure the app expects
        # we then write that newly copied config file to the Drive so other Works have access
        toy_config_file_src = os.path.join(os.getcwd(), lightning_pose_dir, "scripts/configs/config_toy-dataset.yaml")
        toy_config_file_dst = os.path.join(os.getcwd(), "data", "demo", "model_config_demo.yaml")  # TODO: don't hard-code "data"
        self.project_io._copy_file(toy_config_file_src, toy_config_file_dst)
        # self.project_io._put_to_drive_remove_local(toy_config_file_dst, remove_local=False)
        self.project_io.run(action="put_file_to_drive", file_or_dir=toy_config_file_dst, remove_local=False)
        # load project configuration from defaults
        self.project_io.run(action="load_project_defaults")

        # -------------------------------------------------------------
        # update project data (user has clicked button in project UI)
        # -------------------------------------------------------------
        self.train_ui.proj_dir = self.project_io.proj_dir
        self.diagnostics_ui.proj_dir = self.project_io.proj_dir
        self.diagnostics_ui.config_name = self.project_io.config_name

        # -------------------------------------------------------------
        # train models on ui button press
        # -------------------------------------------------------------
        if self.train_ui.run_script_train:
            self.train_models()
            # have tensorboard pull the new data
            self.tensorboard.run(
                "null command",
                cwd=os.getcwd(),
                input_output_only=True,  # pull inputs from Drive, but do not run commands
                inputs=[self.project_io.model_dir],
            )
            self.project_io.update_models = True
            self.train_ui.run_script_train = False

        # set the new outputs for UIs
        if self.project_io.update_models:
            self.project_io.update_models = False
            self.update_trained_models_list(timer=self.train_ui.count)

        # -------------------------------------------------------------
        # run inference on ui button press (single model, multiple vids)
        # -------------------------------------------------------------
        if self.train_ui.run_script_infer and run_while_training:
            print("Cannot run inference in demo right now")
            # self.run_inference(
            #     model=self.train_ui.st_inference_model,
            #     videos=self.train_ui.st_inference_videos,
            # )
            self.train_ui.run_script_infer = False

        # -------------------------------------------------------------
        # initialize diagnostics on button press
        # -------------------------------------------------------------
        if self.diagnostics_ui.run_script:
            self.diagnostics_ui.run(action="build_fiftyone_dataset")
            # self.diagnostics_ui.run(action="start_st_frame")
            # self.diagnostics_ui.run(action="start_st_video")
            self.diagnostics_ui.run_script = False

    def configure_layout(self):

        # training tabs
        train_tab = {"name": "Train/Infer", "content": self.train_ui}
        train_status_tab = {"name": "Train Status", "content": self.tensorboard}

        # diagnostics tabs
        diagnostics_prep_tab = {"name": "Prepare Diagnostics", "content": self.diagnostics_ui}
        fo_tab = {"name": "Labeled Preds", "content": self.diagnostics_ui.fiftyone}
        # st_frame_tab = {"name": "Labeled Diagnostics", "content": self.diagnostics_ui.st_frame}
        # st_video_tab = {"name": "Video Diagnostics", "content": self.diagnostics_ui.st_video}

        return [
            train_tab,
            # train_status_tab,
            # diagnostics_prep_tab,
            # fo_tab,
            # st_frame_tab,
            # st_video_tab,
        ]


app = LightningApp(LitPoseApp())
