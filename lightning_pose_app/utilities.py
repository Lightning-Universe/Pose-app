import logging
import math
import os
import shlex

import numpy as np
from lightning.app.frontend import StreamlitFrontend as LitStreamlitFrontend
from sklearn.cluster import KMeans

_logger = logging.getLogger('APP.UTILS')


def args_to_dict(script_args: str) -> dict:
    """convert str to dict A=1 B=2 to {'A':1, 'B':2}"""
    script_args_dict = {}
    for x in shlex.split(script_args, posix=False):
        try:
            k, v = x.split("=", 1)
        except Exception:
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
            _logger.info(f"Running streamlit on http://{kwargs['host']}:{kwargs['port']}")
        except Exception:
            # on the cloud, args[0] = host, args[1] = port
            pass


def get_frame_number(image_path: str) -> tuple:
    base_name = os.path.basename(image_path)
    frame_number = int(''.join(filter(str.isdigit, base_name)))
    prefix = ''.join(filter(str.isalpha, base_name.split('.')[0]))
    extension = base_name.split('.')[-1]
    return frame_number, prefix, extension


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
            for frame in frames_in_dir:
                idx_img, prefix, ext = get_frame_number(frame.split("/")[-1])
                # get the frames -> t-2, t-1, t, t+1, t + 2
                list_idx = [idx_img - 2, idx_img - 1, idx_img, idx_img + 1, idx_img + 2]
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


def compute_resize_dims(pixels: int) -> int:
    """Return value in {128, 256, 384} that is closest but not greater than pixel size."""
    return min(max(2 ** (math.floor(math.log(pixels, 2))), 128), 384)


def compute_batch_sizes(height: int, width: int) -> tuple:
    """These are hard-coded for values that max out a 16GB GPU (T4) with the example datasets."""
    if height * width >= 1024 * 1024:
        # resize dims likely 384 x 384
        train_batch_size = 6
        dali_base_seq_len = 8
        dali_cxt_seq_len = 8
    elif height * width >= 512 * 512:
        # resize dims likely 384 x 384
        train_batch_size = 8
        dali_base_seq_len = 8
        dali_cxt_seq_len = 8
    else:
        # resize dims likely 256 x 256 or smaller
        train_batch_size = 16
        dali_base_seq_len = 16
        dali_cxt_seq_len = 16
    return train_batch_size, dali_base_seq_len, dali_cxt_seq_len


def update_config(config_dict: dict, new_vals_dict: dict) -> dict:
    # update config using new_vals_dict; assume this is a dict of dicts
    # new_vals_dict = {
    #     "data": new_data_dict,
    #     "eval": new_eval_dict,
    #     ...
    # }
    for sconfig_name, sconfig_dict in new_vals_dict.items():
        for key1, val1 in sconfig_dict.items():
            if isinstance(val1, dict):
                # update config file up to depth 2
                for key2, val2 in val1.items():
                    config_dict[sconfig_name][key1][key2] = val2
            else:
                config_dict[sconfig_name][key1] = val1
    return config_dict


def run_kmeans(X: np.ndarray, n_clusters: int) -> tuple:
    kmeans_obj = KMeans(n_clusters, n_init="auto")
    kmeans_obj.fit(X)
    cluster_labels = kmeans_obj.labels_
    cluster_centers = kmeans_obj.cluster_centers_
    return cluster_labels, cluster_centers


def abspath(path: str) -> str:
    if path[0] == "/":
        path_ = path[1:]
    else:
        path_ = path
    return os.path.abspath(path_)
