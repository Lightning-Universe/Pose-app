from lightning.app import BuildConfig
from typing import List


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
