from lightning import LightningFlow
from lightning.app import BuildConfig
import os
from typing import List

from lightning_pose_app.bashwork import LitBashWork

# sub dirs
lightning_pose_dir = "lightning-pose"


class LitPoseBuildConfig(BuildConfig):

    @staticmethod
    def build_commands() -> List[str]:
        return [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg libsm6 libxext6",
            "python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120",
            f"python -m pip install -e {lightning_pose_dir}",
        ]


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
        self.lp_outputs = None
        self.fiftyone_datasets = []

    def init_lp_outputs_to_uis(self, search_dir=None):

        # get existing model directories that contain predictions.csv file
        # (created upon completion of model training)
        if not search_dir:
            return

        cmd = f"find {search_dir} -maxdepth 4 -type f -name predictions.csv"
        self.work.run(cmd, cwd=os.getcwd(), save_stdout=True)
        if self.work.last_args() == cmd:
            outputs = process_stdout(self.work.last_stdout())
            self.lp_outputs = outputs
            self.work.reset_last_args()

    # TODO: where is the fiftyone db stored?
    def init_fiftyone_outputs_to_ui(self):
        # get existing fiftyone datasets
        cmd = "fiftyone datasets list"
        self.work.run(cmd)
        if self.work.last_args() == cmd:
            names = []
            for x in self.work.stdout:
                if x.endswith("No datasets found"):
                    continue
                if x.startswith("Migrating database"):
                    continue
                if x.endswith("python"):
                    continue
                names.append(x)
        else:
            pass

    def start_fiftyone(self):
        """run fiftyone"""
        # TODO:
        #   right after fiftyone, the previous find command is triggered should not be the case.
        cmd = "fiftyone app launch --address $host --port $port"
        self.work.run(cmd, wait_for_exit=False, cwd=lightning_pose_dir)

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

        if action == "init_lp_outputs_to_uis":
            self.init_lp_outputs_to_uis(**kwargs)
        elif action == "init_fiftyone_outputs_to_ui":
            self.init_fiftyone_outputs_to_ui()
        elif action == "start_fiftyone":
            self.start_fiftyone()
        elif action == "start_extract_frames":
            self.start_extract_frames(**kwargs)


def process_stdout(lines) -> dict:
    """outputs/2022-07-04/17-28-54/test_vid_heatmap.csv"""
    outputs = {}
    if len(lines) == 1 and lines[0][:4] != "find":  # check command not returned
        for l in lines:
            l_split = l.strip().split("/")
            value = l_split[-1]
            key = "/".join(l_split[-3:-1])
            outputs[key] = value
    return outputs
