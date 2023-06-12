import os
import subprocess

from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
import shlex


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


def reencode_video(input_file: str, output_file: str) -> None:
    """reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new video

    """
    # check input file exists
    assert os.path.isfile(input_file), "input video does not exist."
    # check directory for saving outputs exists
    assert os.path.isdir(
        os.path.dirname(output_file)), \
        f"saving folder {os.path.dirname(output_file)} does not exist."
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
