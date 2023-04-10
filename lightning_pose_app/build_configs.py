from lightning.app import BuildConfig
from typing import List

# dir where lightning pose package lives
lightning_pose_dir = "lightning-pose"

# dir where label studio python venv will be setup
label_studio_venv = "venv-label-studio"


class LitPoseBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            "python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120",
            f"python -m pip install -e {lightning_pose_dir}",
        ]


class LitPoseNoGpuBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            f"python -m pip install -e {lightning_pose_dir}",
        ]


class LabelStudioBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        # keep virtualenv because of local package clash with google-oauth
        return [
            "sudo apt-get update",
            "sudo apt-get install libpq-dev",
            f"virtualenv ~/{label_studio_venv}",
            f". ~/{label_studio_venv}/bin/activate; which python; ",
            f"python -m pip install -e .; ",  # install lightning app to have access to packages
            f"python -m pip install label-studio label-studio-sdk; deactivate",
        ]


class TensorboardBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "python -m pip install tensorboard",
        ]


class StreamlitBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "python -m pip install -e ."
        ]