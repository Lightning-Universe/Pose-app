import cv2
import glob
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
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


def copy_and_reformat_video(video_file: str, dst_dir: str, remove_old=True) -> str:
    """Copy a single video, reformatting if necessary, and delete the original."""

    src = video_file

    # make sure copied vid has mp4 extension
    dst = os.path.join(dst_dir, os.path.basename(video_file).replace(".avi", ".mp4"))

    # check 0: do we even need to reformat?
    if os.path.isfile(dst):
        return dst

    # check 1: does file exist?
    if not os.path.exists(src):
        _logger.info(f"{src} does not exist! skipping")
        return None

    # check 2: is file in the correct format for DALI?
    video_file_correct_codec = check_codec_format(src)

    # reencode/rename
    if not video_file_correct_codec:
        _logger.info(f"re-encoding {src} to be compatable with Lightning Pose video reader")
        reencode_video(src, dst)
        # remove old video
        if remove_old:
            os.remove(src)
    else:
        # make dir to write into
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # rename
        if remove_old:
            os.rename(src, dst)
        else:
            shutil.copyfile(src, dst)

    return dst


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


def make_video_snippet(
        video_file: str,
        preds_file: str,
        clip_length: int = 30,
        likelihood_thresh: float = 0.9
) -> str:

    # save videos with csv file
    save_dir = os.path.dirname(preds_file)

    # load pose predictions
    df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # how large is the clip window?
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    win_len = int(fps * clip_length)

    # make a `clip_length` second video clip that contains the highest keypoint motion energy
    src = video_file
    dst = os.path.join(save_dir, os.path.basename(video_file).replace(".mp4", ".short.mp4"))
    if win_len >= df.shape[0]:
        # short video, no need to shorten further. just copy existing video
        shutil.copyfile(src, dst)
    else:
        # compute motion energy (averaged over keypoints)
        kps_and_conf = df.to_numpy().reshape(df.shape[0], -1, 3)
        kps = kps_and_conf[:, :, :2]
        conf = kps_and_conf[:, :, -1]
        conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)
        kps[conf2 < likelihood_thresh] = np.nan
        me = np.nanmean(np.linalg.norm(kps[1:] - kps[:1], axis=2), axis=-1)

        # find window
        df_me = pd.DataFrame({"me": np.concatenate([[0], me])})
        df_me_win = df_me.rolling(window=win_len, center=False).mean()
        # rolling places results in right edge of window, need to subtract this
        clip_start_idx = df_me_win.me.argmax() - win_len
        # convert to seconds
        clip_start_sec = int(clip_start_idx / fps)
        # if all predictions are bad, make sure we still create a valid snippet video
        if np.isnan(clip_start_sec) or clip_start_sec < 0:
            clip_start_sec = 0

        # make clip
        ffmpeg_cmd = f"ffmpeg -ss {clip_start_sec} -i {src} -t {clip_length} {dst}"
        subprocess.run(ffmpeg_cmd, shell=True)

    return dst


def get_frame_number(basename: str) -> tuple:
    """img0000234.png -> (234, "img", ".png")"""
    ext = basename.split(".")[-1]  # get base name
    split_idx = None
    for c_idx, c in enumerate(basename):
        try:
            int(c)
            split_idx = c_idx
            break
        except ValueError:
            continue
    # remove prefix
    prefix = basename[:split_idx]
    idx = basename[split_idx:]
    # remove file extension
    idx = idx.replace(f".{ext}", "")
    return int(idx), prefix, ext


def is_context_dataset(labeled_data_dir: str, selected_frames_filename: str) -> bool:
    """Starting from labeled data directory, determine if this is a context dataset or not."""
    # loop over all labeled frames, break as soon as single frame fails
    is_context = True
    n_frames = 0
    if os.path.isdir(labeled_data_dir):
        for d in os.listdir(labeled_data_dir):
            frames_in_dir_file = os.path.join(labeled_data_dir, d, selected_frames_filename)
            if not os.path.exists(frames_in_dir_file):
                continue
            frames_in_dir = np.genfromtxt(frames_in_dir_file, delimiter=",", dtype=str)
            print(frames_in_dir)
            for frame in frames_in_dir:
                idx_img, prefix, ext = get_frame_number(frame.split("/")[-1])
                # get the frames -> t-2, t-1, t, t+1, t + 2
                list_idx = [idx_img - 2, idx_img - 1, idx_img, idx_img + 1, idx_img + 2]
                print(list_idx)
                for fr_num in list_idx:
                    # replace frame number with 0 if we're at the beginning of the video
                    fr_num = max(0, fr_num)
                    # split name into pieces
                    img_pieces = frame.split("/")
                    # figure out length of integer
                    int_len = len(img_pieces[-1].replace(f".{ext}", "").replace(prefix, ""))
                    # replace original frame number with context frame number
                    img_pieces[-1] = f"{prefix}{str(fr_num).zfill(int_len)}.{ext}"
                    img_name = "/".join(img_pieces)
                    if not os.path.exists(os.path.join(labeled_data_dir, d, img_name)):
                        is_context = False
                        break
                    else:
                        n_frames += 1
    # set to False if we didn't find any frames
    if n_frames == 0:
        is_context = False
    return is_context


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
                levels1 = ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2")
                vids = df_tmp.loc[:, levels1]
                levels2 = ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2")
                imgs = df_tmp.loc[:, levels2]
                new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                df_tmp1 = df_tmp.drop(levels1, axis=1)
                df_tmp2 = df_tmp1.drop(levels2, axis=1)
                df_tmp2.index = new_col
                df_tmp = df_tmp2
        except IndexError:
            try:
                h5_file = glob.glob(
                    os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.h5")
                )[0]
                df_tmp = pd.read_hdf(h5_file)
                if isinstance(df_tmp.index, pd.core.indexes.multi.MultiIndex):
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


def abspath(path):
    if path[0] == "/":
        path_ = path[1:]
    else:
        path_ = path
    return os.path.abspath(path_)
