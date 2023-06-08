from lightning import CloudCompute, LightningFlow
import os

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LitPoseBuildConfig, lightning_pose_dir


class StreamlitAppLightningPose(LightningFlow):
    """UI to run Streamlit labeled frame app."""

    def __init__(self, *args, app_type, **kwargs):

        super().__init__(*args, **kwargs)

        self.work = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=LitPoseBuildConfig(),  # this may not be necessary
        )

        # choose labeled frame or video option
        if app_type == "frame":
            script_name = "labeled_frame_diagnostics.py"
        elif app_type == "video":
            script_name = "video_diagnostics.py"
        else:
            raise ValueError(f"'app_type' argument must be in ['frame', 'video']; not {app_type}")

        self.app_type = app_type
        self.script_name = script_name
        self.is_initialized = False

        # params updated externally by top-level flow
        self.proj_dir = None

    def initialize(self, **kwargs):

        if not self.is_initialized:

            self.is_initialized = True

            if kwargs.get("model_dir", None):
                model_dir = kwargs["model_dir"]
            else:
                model_dir = os.path.join(os.getcwd(), self.proj_dir[1:], "models")

            model_dir_args = f" --model_dir={model_dir} --make_dir"

            cmd = f"streamlit run lightning_pose/apps/{self.script_name}" \
                + " --server.address $host --server.port $port --server.headless true" \
                + " -- " \
                + " " + model_dir_args

            self.work.run(cmd, cwd=lightning_pose_dir, wait_for_exit=False)

    def pull_models(self, **kwargs):
        inputs = kwargs.get("inputs", None)
        if inputs:
            self.work.run(
                "null command",
                cwd=os.getcwd(),
                input_output_only=True,  # pull inputs from Drive, but do not run commands
                inputs=inputs,
            )

    def run(self, action, **kwargs):

        if action == "initialize":
            self.initialize(**kwargs)
        elif action == "pull_models":
            self.pull_models(**kwargs)
