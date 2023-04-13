from lightning.app import BuildConfig
from typing import List

# dir where lightning pose package lives
lightning_pose_dir = "lightning-pose"


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
        # keep virtualenv because of local package clash with google-oauth
        return [
            "sudo apt-get update",
            "sudo apt-get install libpq-dev",
            "conda install libffi==3.3",
            "pip install -e .; ",  # install lightning app to have access to packages
            "pip install label-studio label-studio-sdk; deactivate",
        ]


class TensorboardBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "pip install tensorboard",
        ]


class StreamlitBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "pip install -e ."
        ]
