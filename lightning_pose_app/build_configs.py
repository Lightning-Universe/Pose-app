from lightning.app import BuildConfig
from typing import List

from lightning_pose_app import LIGHTNING_POSE_DIR


class LitPoseBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            f"pip install -e {LIGHTNING_POSE_DIR}",
        ]
