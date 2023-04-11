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
        component_name="litpose",
        parallel=False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.work = LitBashWork(
            cloud_compute=cloud_compute,
            cloud_build_config=LitPoseBuildConfig(),  # this is where Lightning Pose is installed
            drive_name=drive_name,
            component_name=component_name,
            wait_seconds_after_run=1,
            parallel=parallel,
        )

        self.work_is_done_extract_frames = True
        self.work_is_done_training = True
        self.work_is_done_inference = True
        self.count = 0

        # these attributes get set by external app
        self.trained_models = {}

    def update_trained_models_dict(self, search_dir=None):

        # get existing model directories that contain predictions.csv file
        # (created upon completion of model training)
        if not search_dir:
            return

        # pull models from Drive on first call
        if self.count == 0:
            self.work.run(
                "null command",
                cwd=os.getcwd(),
                input_output_only=True,  # pull inputs from Drive, but do not run commands
                inputs=[search_dir],
            )
            self.count += 1

        cmd = f"find {search_dir} -maxdepth 4 -type f -name predictions.csv"
        self.work.run(cmd, cwd=os.getcwd(), save_stdout=True)
        if self.work.last_args() == cmd:
            outputs = process_stdout(self.work.last_stdout())
            self.trained_models = outputs
            self.work.reset_last_args()
        self.count += 1

    def start_extract_frames(self, video_files=None, proj_dir=None, n_frames_per_video=20):

        print(f"launching extraction for video {video_files[0]}")
        self.work_is_done_extract_frames = False

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
        import time
        self.work_is_done_inference = False
        print(f"launching inference for video {video} using model {model}")
        time.sleep(5)
        self.work_is_done_inference = True

    def run(self, action=None, **kwargs):

        if action == "update_trained_models_dict":
            self.update_trained_models_dict(**kwargs)
        elif action == "start_extract_frames":
            self.start_extract_frames(**kwargs)
        elif action == "run_inference":
            self.run_inference(**kwargs)


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
