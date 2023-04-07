import lightning.app as L
from typing import Optional, Union, List

# sub dirs
lightning_pose_dir = "lightning-pose"

# virtualenv names located in ~
tensorboard_venv = "venv-tensorboard"


class TensorboardBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "python -m pip install tensorboard",
        ]


class LitPoseBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            "python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120",
            f"python -m pip install -e {lightning_pose_dir}",
        ]


class StreamlitBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "python -m pip install -e ."
        ]
