from lightning.app import CloudCompute, LightningFlow
import os

from lightning_pose_app import MODELS_DIR, LIGHTNING_POSE_DIR
from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LitPoseBuildConfig


class StreamlitAppLightningPose(LightningFlow):
    """UI to run Streamlit labeled frame app."""

    def __init__(self, *args, app_type, **kwargs):

        super().__init__(*args, **kwargs)

        self.work = LitBashWork(
            cloud_compute=CloudCompute("default"),
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
                model_dir = os.path.join(os.getcwd(), self.proj_dir[1:], MODELS_DIR)

            model_dir_args = f" --model_dir={model_dir} --make_dir"

            cmd = f"streamlit run lightning_pose/apps/{self.script_name}" \
                + " --server.address $host --server.port $port --server.headless true" \
                + " -- " \
                + " " + model_dir_args

            self.work.run(cmd, cwd=LIGHTNING_POSE_DIR, wait_for_exit=False)

    def run(self, action, **kwargs):

        if action == "initialize":
            self.initialize(**kwargs)
