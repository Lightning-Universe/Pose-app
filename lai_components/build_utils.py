import lightning.app as L
from typing import Optional, Union, List

# sub dirs
lightning_pose_dir = "lightning-pose"
tracking_diag_dir = "tracking-diagnostics"

# virtualenv names located in ~
lightning_pose_venv = "venv-lightning-pose"
tensorboard_venv = "venv-tensorboard"


class TensorboardBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            f"virtualenv ~/{tensorboard_venv}",
            f". ~/{tensorboard_venv}/bin/activate; ",
            f"python -m pip install tensorflow tensorboard;",
            f"deactivate",
        ]


class FiftyOneBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            f"virtualenv ~/{lightning_pose_venv}",
            f". ~/{lightning_pose_venv}/bin/activate;python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102; deactivate",
            f". ~/{lightning_pose_venv}/bin/activate;python -m pip install -e {lightning_pose_dir}; deactivate",
        ]


class StreamlitBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            f"virtualenv ~/{lightning_pose_venv}",
            f". ~/{lightning_pose_venv}/bin/activate; ",
            f"cd tracking-diagnostics; ",
            f"which python; ",
            f"python -m pip install -e .; ",
            f"deactivate",
        ]
