from lightning.app import BuildConfig
from typing import List

# dir where lightning pose package lives
lightning_pose_dir = "lightning-pose"

# dir where label studio python venv will be set up
label_studio_venv = None


class LitPoseBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            f"pip install -e {lightning_pose_dir}",
        ]


class LabelStudioBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "pip install -e .",  # install lightning app to have access to packages
        ]


class TensorboardBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "pip install tensorboard",
        ]
