import lightning_app as L
from typing import Optional, Union, List

# sub dirs
lightning_pose_dir  = "lightning-pose"
label_studio_dir    = "label-studio"
tracking_diag_dir   = "tracking-diagnostics"

# virtualenv names located in ~
lightning_pose_venv = "venv-lightning-pose"
label_studio_venv   = "venv-label-studio"
tensorboard_venv    = "venv-tensorboard"


class TensorboardBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands(self) -> List[str]:
        return [
            f"virtualenv ~/{tensorboard_venv}",
            f". ~/{tensorboard_venv}/bin/activate; ",
            f"python -m pip install tensorflow tensorboard;",
            f"deactivate",
        ]


class LabelStudioBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands(self) -> List[str]:
        return [
          f"virtualenv ~/{label_studio_venv}",
          "git clone https://github.com/robert-s-lee/label-studio",
          "cd label-studio; git checkout x-frame-options; cd ..",
          # source is not available, so using ".",
          f". ~/{label_studio_venv}/bin/activate; ",
          f"cd label-studio; ",
          f"which python; ",
          f"python -m pip install -e .;",
          f"deactivate",
          # TODO: after PR is merged,
          # f". ~/{label_studio_venv}/bin/activate; which python; python -m pip install label-studio",
        ]


class FiftyOneBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands(self) -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            f"virtualenv ~/{lightning_pose_venv}",
            f". ~/{lightning_pose_venv}/bin/activate;python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102; deactivate",
            f". ~/{lightning_pose_venv}/bin/activate;python -m pip install -e {lightning_pose_dir}; deactivate",
        ]


class StreamlitBuildConfig(L.BuildConfig):

    @staticmethod
    def build_commands(self) -> List[str]:
        return [
            f"virtualenv ~/{lightning_pose_venv}",
            f". ~/{lightning_pose_venv}/bin/activate; ",
            f"cd tracking-diagnostics; ",
            f"which python; ",
            f"python -m pip install -e .; ",
            f"deactivate",
        ]
