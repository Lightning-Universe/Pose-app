from lightning import LightningFlow
import os

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LitPoseBuildConfig, lightning_pose_dir


class LitPose(LightningFlow):

    def __init__(
        self,
        *args,
        cloud_compute,
        drive_name,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.work = LitBashWork(
            cloud_compute=cloud_compute,
            cloud_build_config=LitPoseBuildConfig(),  # this is where Lightning Pose is installed
            drive_name=drive_name,
            component_name="litpose",
            wait_seconds_after_run=1,
        )

        self.work_is_done_extract_frames = False
        self.work_is_done_training = True

    def start_extract_frames(self, video_files=None, proj_dir=None, n_frames_per_video=20):

        # set videos to select frames from
        vid_file_args = ""
        for vid_file in video_files:
            vid_file_ = os.path.join(os.getcwd(), vid_file)
            vid_file_args += f" --video_files={vid_file_}"

        data_dir = os.path.join(os.getcwd(), proj_dir, "labeled-data")

        cmd = "python" \
              + " scripts/extract_frames.py" \
              + vid_file_args \
              + f" --data_dir={data_dir}" \
              + f" --n_frames_per_video={n_frames_per_video}" \
              + f" --context_frames=2" \
              + f" --export_idxs_as_csv"
        self.work.run(
            cmd,
            wait_for_exit=True,
            cwd=lightning_pose_dir,
            inputs=video_files,
            outputs=[os.path.join(proj_dir, "labeled-data")],
        )
        self.work_is_done_extract_frames = True

    def run_inference(self, model, video):
        pass

    def run(self, action=None, **kwargs):

        if action == "start_extract_frames":
            self.start_extract_frames(**kwargs)
        elif action == "run_inference":
            self.run_inference(**kwargs)
