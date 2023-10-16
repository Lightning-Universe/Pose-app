import cv2
import glob
from lightning.app import LightningWork
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
from lightning.app.storage import FileSystem
import logging
import numpy as np
import os
import pandas as pd
import shlex
import shutil
import subprocess


_logger = logging.getLogger('APP.UTILS')


def args_to_dict(script_args: str) -> dict:
    """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
    script_args_dict = {}
    for x in shlex.split(script_args, posix=False):
        try:
            k, v = x.split("=", 1)
        except:
            k = x
            v = None
        script_args_dict[k] = v
    return script_args_dict


def dict_to_args(script_args_dict: dict) -> str:
    """convert dict {'A':1, 'B':2} to str A=1 B=2 to """
    script_args_array = []
    for k,v in script_args_dict.items():
        script_args_array.append(f"{k}={v}")
    # return as a text
    return " \n".join(script_args_array)


class StreamlitFrontend(LitStreamlitFrontend):
    """Provide helpful print statements for where streamlit tabs are forwarded."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_server(self, *args, **kwargs):
        super().start_server(*args, **kwargs)
        try:
            _logging.info(f"Running streamlit on http://{kwargs['host']}:{kwargs['port']}")
        except:
            # on the cloud, args[0] = host, args[1] = port
            pass


class WorkWithFileSystem(LightningWork):

    def __init__(self, *args, name, **kwargs):

        super().__init__(*args, **kwargs)

        # uniquely identify prints
        self.work_name = name

        # initialize shared storage system
        self._drive = FileSystem()

    def get_from_drive(self, inputs, overwrite=True):
        for i in inputs:
            _logger.debug(f"{self.work_name.upper()} get {i}")
            try:  # file may not be ready
                src = i  # shared
                dst = self.abspath(i)  # local
                self._drive.get(src, dst, overwrite=overwrite)
                _logger.debug(f"{self.work_name.upper()} data saved at {dst}")
            except Exception as e:
                _logger.debug(f"{self.work_name.upper()} did not load {i} from FileSystem: {e}")
                continue

    def put_to_drive(self, outputs):
        for o in outputs:
            _logger.debug(f"{self.work_name.upper()} drive try put {o}")
            src = self.abspath(o)  # local
            dst = o  # shared
            # make sure dir ends with / so that put works correctly
            if os.path.isdir(src):
                src = os.path.join(src, "")
                dst = os.path.join(dst, "")
            # check to make sure file exists locally
            if not os.path.exists(src):
                continue
            self._drive.put(src, dst)
            _logger.debug(f"{self.work_name.upper()} drive success put {dst}")

    @staticmethod
    def abspath(path):
        if path[0] == "/":
            path_ = path[1:]
        else:
            path_ = path
        return os.path.abspath(path_)


def reencode_video(input_file: str, output_file: str) -> None:
    """reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new video

    """
    # check input file exists
    assert os.path.isfile(input_file), "input video does not exist."
    # check directory for saving outputs exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
    subprocess.run(ffmpeg_cmd, shell=True)


def check_codec_format(input_file: str) -> bool:
    """Run FFprobe command to get video codec and pixel format."""

    ffmpeg_cmd = f'ffmpeg -i {input_file}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
    output_str = output_str.stderr

    # search for correct codec (h264) and pixel format (yuv420p)
    if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
        # print('Video uses H.264 codec')
        is_codec = True
    else:
        # print('Video does not use H.264 codec')
        is_codec = False
    return is_codec


def copy_and_reformat_video_directory(src_dir: str, dst_dir: str) -> None:
    """Copy a directory of videos, reencoding to be DALI compatible if necessary."""

    os.makedirs(dst_dir, exist_ok=True)
    video_dir_contents = os.listdir(src_dir)
    for file_or_dir in video_dir_contents:
        src = os.path.join(src_dir, file_or_dir)
        dst = os.path.join(dst_dir, file_or_dir)
        if os.path.isdir(src):
            # don't copy subdirectories in video directory
            continue
        else:
            if src.endswith(".mp4") or src.endswith(".avi"):
                video_file_correct_codec = check_codec_format(src)
                if not video_file_correct_codec:
                    _logger.info(
                        f"re-encoding {src} to be compatable with Lightning Pose video reader")
                    reencode_video(src, dst.replace(".avi", ".mp4"))
                else:
                    # copy already-formatted video
                    shutil.copyfile(src, dst)
            else:
                # copy non-video files
                shutil.copyfile(src, dst)


def get_frames_from_idxs(cap, idxs) -> np.ndarray:
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype="uint8")
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            _logger.debug(
                "warning! reached end of video; returning blank frames for remainder of "
                + "requested indices"
            )
            break
    return frames


def collect_dlc_labels(dlc_dir: str) -> pd.DataFrame:
    """Collect video-specific labels from DLC project and save in a single pandas dataframe."""

    dirs = os.listdir(os.path.join(dlc_dir, "labeled-data"))
    dirs.sort()
    dfs = []
    for d in dirs:
        try:
            csv_file = glob.glob(os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.csv"))[0]
            df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            if len(df_tmp.index.unique()) != df_tmp.shape[0]:
                # new DLC labeling scheme that splits video/image in different cells
                vids = df_tmp.loc[
                       :, ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2")]
                imgs = df_tmp.loc[
                       :, ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2")]
                new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                df_tmp1 = df_tmp.drop(
                    ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2"), axis=1,
                )
                df_tmp2 = df_tmp1.drop(
                    ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2"), axis=1,
                )
                df_tmp2.index = new_col
                df_tmp = df_tmp2
        except IndexError:
            try:
                h5_file = glob.glob(
                    os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.h5")
                )[0]
                df_tmp = pd.read_hdf(h5_file)
                if type(df_tmp.index) == pd.core.indexes.multi.MultiIndex:
                    # new DLC labeling scheme that splits video/image in different cells
                    imgs = [i[2] for i in df_tmp.index]
                    vids = [df_tmp.index[0][1] for _ in imgs]
                    new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                    df_tmp1 = df_tmp.reset_index().drop(
                        columns="level_0").drop(columns="level_1").drop(columns="level_2")
                    df_tmp1.index = new_col
                    df_tmp = df_tmp1
            except IndexError:
                _logger.error(f"Could not find labels for {d}; skipping")
                continue

        dfs.append(df_tmp)
    df_all = pd.concat(dfs)

    return df_all
