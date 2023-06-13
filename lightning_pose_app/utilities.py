import cv2
from lightning import LightningWork
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
from lightning.app.storage import FileSystem
import numpy as np
import os
import shlex
import subprocess


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
            print(f"Running streamlit on http://{kwargs['host']}:{kwargs['port']}")
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
            print(f"{self.work_name.upper()} get {i}")
            try:  # file may not be ready
                src = i  # shared
                dst = self.abspath(i)  # local
                self._drive.get(src, dst, overwrite=overwrite)
                print(f"{self.work_name.upper()} data saved at {dst}")
            except Exception as e:
                print(f"{self.work_name.upper()} did not load {i} from FileSystem: {e}")
                continue

    def put_to_drive(self, outputs):
        for o in outputs:
            print(f"{self.work_name.upper()} drive try put {o}")
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
            print(f"{self.work_name.upper()} drive success put {dst}")

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


def check_codec_format(input_file: str):
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


def get_frames_from_idxs(cap, idxs):
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
            print(
                "warning! reached end of video; returning blank frames for remainder of "
                + "requested indices"
            )
            break
    return frames
