"""Main loop for lightning pose app

To run from the command line (inside the conda environment named "lai" here):
(lai) user@machine: lightning run app app.py

"""

from lightning import CloudCompute, LightningApp, LightningFlow
from lightning.app.utilities.cloud import is_running_in_cloud
import os
import time
import yaml

from lightning_pose_app.litpose import LitPose
from lightning_pose_app.ui.extract_frames import ExtractFramesUI
from lightning_pose_app.build_configs import lightning_pose_dir


class LitPoseApp(LightningFlow):

    def __init__(self):
        super().__init__()

        # shared data for apps; NOTE: this is hard-coded in the run_inference method below too
        drive_name = "lit://lpa"

        # extract frames tab (flow)
        self.extract_ui = ExtractFramesUI(drive_name=drive_name)
        self.extract_ui.proj_dir = "data/TEST"

        # works for inference
        self.inference = {}

    def extract_frames(self, video_files, proj_dir, n_frames_per_video=20):

        print("--------------------")
        print("RUN EXTRACTION")
        print("--------------------")
        
        # launch works
        for v, video in enumerate(video_files):
            self.inference[video] = LitPose(
                cloud_compute=CloudCompute("gpu"),
                drive_name="lit://lpa",
                component_name=f"extractor_{v}",
                parallel=is_running_in_cloud(),
            )
            self.inference[video].run(
                action="start_extract_frames",
                video_files=[video],
                proj_dir=proj_dir,
                n_frames_per_video=n_frames_per_video,
            )

        # clean up works
        while len(self.inference) > 0:
            for video in list(self.inference):
                if (video in self.inference.keys()) \
                        and self.inference[video].work_is_done_extract_frames:
                    # kill work
                    print(f"killing work from video {video}")
                    del self.inference[video]

        print("--------------------")
        print("END OF EXTRACTION")
        print("--------------------")

    def run(self):

        if self.extract_ui.run_script:
            self.extract_frames(
                video_files=self.extract_ui.st_video_files,
                proj_dir=self.extract_ui.proj_dir,
                n_frames_per_video=self.extract_ui.st_n_frames_per_video,
            )
            self.extract_ui.run_script = False

    def configure_layout(self):

        # init tabs
        extract_tab = {"name": "Extract Frames", "content": self.extract_ui}
        return [extract_tab]


app = LightningApp(LitPoseApp())
