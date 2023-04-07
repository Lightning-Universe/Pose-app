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
        proj_dir=None,
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

        # these attributes get set by external app
        self.trained_models = None

    def find_trained_models(self, search_dir=None):

        # get existing model directories that contain predictions.csv file
        # (created upon completion of model training)
        if not search_dir:
            return

        cmd = f"find {search_dir} -maxdepth 4 -type f -name predictions.csv"
        self.work.run(cmd, cwd=os.getcwd(), save_stdout=True)
        if self.work.last_args() == cmd:
            outputs = process_stdout(self.work.last_stdout())
            self.trained_models = outputs
            self.work.reset_last_args()

    def start_extract_frames(
            self, video_files=None, proj_dir=None, script_name=None, n_frames_per_video=20):

        # set videos to select frames from
        vid_file_args = ""
        for vid_file in video_files:
            vid_file_ = os.path.join(os.getcwd(), vid_file)
            vid_file_args += f" --video_files={vid_file_}"

        data_dir = os.path.join(os.getcwd(), proj_dir, "labeled-data")

        cmd = "python" \
              + " " + script_name \
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

    def run(self, action=None, **kwargs):

        if action == "find_trained_models":
            self.find_trained_models(**kwargs)
        elif action == "start_extract_frames":
            self.start_extract_frames(**kwargs)


def process_stdout(lines) -> dict:
    """example: outputs/2022-07-04/17-28-54/test_vid_heatmap.csv"""
    outputs = {}
    if lines and lines[0][:4] != "find":  # check command not returned
        for l in lines:
            l_split = l.strip().split("/")
            value = l_split[-1]
            key = "/".join(l_split[-3:-1])
            outputs[key] = value
    return outputs
