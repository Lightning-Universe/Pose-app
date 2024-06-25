import logging
import os
import shutil
import zipfile
import time
from io import BytesIO
import io
import glob
import cv2
import numpy as np
import streamlit as st
from lightning.app import CloudCompute, LightningFlow, LightningWork
from lightning.app.structures import Dict
from lightning.app.utilities.state import AppState
from streamlit_autorefresh import st_autorefresh
from typing import Optional

import torch
import yaml
from scipy.stats import zscore
from omegaconf import DictConfig

from lightning_pose.utils.scripts import get_imgaug_transform, get_dataset, get_data_module
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose_app.utilities import abspath

import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

from lightning_pose_app import (
    COLLECTED_DATA_FILENAME,
    LABELED_DATA_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_INFER_DIR,
    VIDEOS_TMP_DIR,
    ZIPPED_TMP_DIR,
)
from lightning_pose_app.backend.video import copy_and_reformat_video
from lightning_pose_app.backend.extract_frames import (
    export_frames,
    select_frame_idxs_kmeans,
    select_frame_idxs_model,
    find_contextual_frames,
    get_frame_number,
    get_frame_paths,
    convert_csv_to_dict,
    annotate_frames,
    zip_annotated_images,
)
from lightning_pose_app.utilities import (
    StreamlitFrontend,
    abspath,
)


_logger = logging.getLogger('APP.EXTRACT_FRAMES')

# options for handling frame extraction
VIDEO_RANDOM_STR = "Upload videos and automatically extract random frames"
ZIPPED_FRAMES_STR = "Upload zipped files of frames"
VIDEO_MODEL_STR = "Automatically extract frames using a given model"

# Options for loading videos
VIDEO_SELECT_NEW = "Upload video(s)"
VIDEO_SELECT_UPLOADED = "Select previously uploaded video(s)"

# options for process message in extract frames tab
PROCEED_STR = "Please proceed to the next tab to label frames."
PROCEED_FMT = "<p style='font-family:sans-serif; color:Green;'>%s</p>"


class ExtractFramesWork(LightningWork):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # record progress of computationally-intensive steps (like reading video frames)
        self.progress = 0.0

        # flag to communicate state of work to parent flow
        self.work_is_done = False

    def _extract_frames(
        self,
        method: str,
        video_file: str,
        proj_dir: str,
        n_frames_per_video: int,
        frame_range: list = [0, 1],
        model_dir: str = "None",         # for "active" strategy
        likelihood_thresh: float = 0.0,  # for "active" strategy
        thresh_metric_z: float = 3.0,    # for "active" strategy
    ) -> None:

        _logger.info(f"============== extracting frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done = False

        data_dir_rel = os.path.join(proj_dir, LABELED_DATA_DIR)
        if not os.path.exists(data_dir_rel):
            data_dir = abspath(data_dir_rel)
        else:
            data_dir = data_dir_rel
        n_digits = 8
        extension = "png"
        context_frames = 2

        # check: does file exist?
        if not os.path.exists(video_file):
            video_file_abs = abspath(video_file)
        else:
            video_file_abs = video_file
        video_file_exists = os.path.exists(video_file_abs)
        _logger.info(f"video file exists? {video_file_exists}")
        if not video_file_exists:
            _logger.info("skipping frame extraction")
            return

        # create folder to save images
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        save_dir = os.path.join(data_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)  # need this for the np.savetxt below

        # select indices for labeling
        if method == "random":
            # reduce image size, even more if there are many frames
            cap = cv2.VideoCapture(video_file_abs)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if n_frames > 1e5:
                resize_dims = 32
            else:
                resize_dims = 64
            idxs_selected = select_frame_idxs_kmeans(
                video_file=video_file_abs,
                resize_dims=resize_dims,
                n_frames_to_select=n_frames_per_video,
                frame_range=frame_range,
                work=self,
            )
        elif method == "active":
            idxs_selected = select_frame_idxs_model(
                video_file=video_file_abs,
                model_dir=model_dir,
                n_frames_to_select=n_frames_per_video,
                frame_range=frame_range,
                likelihood_thresh=likelihood_thresh,
                thresh_metric_z=thresh_metric_z,
            )
        else:
            raise NotImplementedError

        # save csv file inside same output directory
        frames_to_label = np.array([
            "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected])
        np.savetxt(
            os.path.join(save_dir, SELECTED_FRAMES_FILENAME),
            np.sort(frames_to_label),
            delimiter=",",
            fmt="%s"
        )

        # save frames
        export_frames(
            video_file=video_file_abs,
            save_dir=save_dir,
            frame_idxs=idxs_selected,
            format=extension,
            n_digits=n_digits,
            context_frames=context_frames,
        )

        # set flag for parent app
        self.work_is_done = True

    def _unzip_frames(
        self,
        video_file: str,
        proj_dir: str,
    ) -> None:

        _logger.info(f"============== unzipping frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done = False

        data_dir_rel = os.path.join(proj_dir, LABELED_DATA_DIR)
        if not os.path.exists(data_dir_rel):
            data_dir = abspath(data_dir_rel)
        else:
            data_dir = data_dir_rel

        # check: does file exist?
        if not os.path.exists(video_file):
            video_file_abs = abspath(video_file)
        else:
            video_file_abs = video_file
        video_file_exists = os.path.exists(video_file_abs)
        _logger.info(f"zipped file exists? {video_file_exists}")
        if not video_file_exists:
            _logger.info("skipping frame extraction")
            return

        # create folder to save images
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        save_dir = os.path.join(data_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)

        # unzip file in tmp directory
        with zipfile.ZipFile(video_file_abs) as z:
            unzipped_dir = video_file_abs.replace(".zip", "")
            z.extractall(path=unzipped_dir)

            # create a list all images in folder
            filenames = [
                os.path.basename(file_info.filename)
                for file_info in z.infolist()
                if file_info.filename.endswith('.png')
            ]

        # handle nested directories by moving all files to the base unzipped_dir
        for root, dirs, files in os.walk(unzipped_dir):
            for file in files:
                if file.endswith('.png') or file.endswith(SELECTED_FRAMES_FILENAME):
                    file_path = os.path.join(root, file)
                    if root != unzipped_dir:  # if the file is in a subfolder
                        shutil.move(file_path, unzipped_dir)

        # optionally clean up empty directories
        for root, dirs, files in os.walk(unzipped_dir, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if len(os.listdir(dir_path)) == 0:  # directory is empty
                    os.rmdir(dir_path)

        # process and rename filenames
        correct_imgnames = []
        for filename in filenames:
            frame_number = get_frame_number(filename)[0]
            new_filename = f"img{frame_number:08d}.png"
            src_path = os.path.join(unzipped_dir, filename)
            dst_path = os.path.join(unzipped_dir, new_filename)
            if filename != new_filename:
                os.rename(src_path, dst_path)
                _logger.info(f"Renamed '{filename}' to '{new_filename}'")
            correct_imgnames.append(new_filename)

        if len(correct_imgnames) == 0:
            _logger.error("No valid frame files found. Aborting frame extraction.")
            return

        # process filenames with get_frame_number and handle contexts
        frame_details = [get_frame_number(filename) for filename in correct_imgnames]
        frame_numbers = [details[0] for details in frame_details]
        csv_exists = SELECTED_FRAMES_FILENAME in os.listdir(unzipped_dir)
        frames, is_context = find_contextual_frames(frame_numbers)

        if not csv_exists:
            if not is_context:
                frames_to_label = np.array([
                    f"{prefix}{num:08d}.{ext}" for num, prefix, ext in frame_details
                ])
            else:
                frames_to_label = np.array([
                    f"{prefix}{num:08d}.{ext}"
                    for num, prefix, ext in frame_details
                    if num in frames
                ])
            np.savetxt(
                os.path.join(unzipped_dir, SELECTED_FRAMES_FILENAME),
                frames_to_label,
                delimiter=",",
                fmt="%s"
            )

        # save all contents to data directory
        # don't use copytree as the destination dir may already exist
        files = os.listdir(unzipped_dir)
        for file in files:
            src = os.path.join(unzipped_dir, file)
            dst = os.path.join(save_dir, file)
            shutil.copyfile(src, dst)

        # set flag for parent app
        self.work_is_done = True

    def _save_annotated_frames(
        self,
        proj_dir: str,
        selected_body_parts: list = None
    ) -> None:
        
        _logger.info(f"============== checking frames ================")
        self.work_is_done = False
        
        if not os.path.exists(proj_dir):
            proj_dir = abspath(proj_dir)

        labeled_data_path = os.path.join(proj_dir, LABELED_DATA_DIR)
        # Create a new folder for the annotated frames
        labeled_data_check_path = os.path.join(proj_dir, 'labeled-data-check')
        # Convert CSV to dictionary
        collected_data_file_path = os.path.join(proj_dir, COLLECTED_DATA_FILENAME)
        annotations_dict = convert_csv_to_dict(collected_data_file_path, selected_body_parts)

        
        # Copy and annotate frames
        for frame_rel_path, data in annotations_dict.items():
            frame_full_path = data['frame_full_path']
            video = data['video']
            frame_annotations = data['bodyparts']

            _logger.info(f"Processing frame: {frame_full_path} for video: {video} with annotations: {frame_annotations}")

            video_folder_path = os.path.join(labeled_data_check_path, video)
            project_name = os.path.basename(proj_dir)
            config_file_path = os.path.join(proj_dir, "model_config_" + project_name + ".yaml")
            
            annotate_frames(frame_full_path, frame_annotations, video_folder_path, config_file_path)

        self.work_is_done = True
        _logger.info(f"============== completed saving annotated frames ================")

    def run(self, action: str, **kwargs) -> None:
        if action == "extract_frames_using_kmeans":
            new_vid_file = copy_and_reformat_video(
                video_file=abspath(kwargs["video_file"]),
                dst_dir=abspath(os.path.join(kwargs["proj_dir"], VIDEOS_DIR)),
            )
            # save relative rather than absolute path
            kwargs["video_file"] = '/'.join(new_vid_file.split('/')[-4:])
            self._extract_frames(method="random", **kwargs)
        elif action == "extract_frames_using_model":
            # note: we do not need to copy and reformat video because we assume the user has
            # already run inference on this video(s)
            self._extract_frames(method="active", **kwargs)
        elif action == "unzip_frames":
            self._unzip_frames(**kwargs)
        elif action == "save_annotated_frames":
            self._save_annotated_frames(**kwargs)
        else:
            pass


class ExtractFramesUI(LightningFlow):
    """UI to manage projects - create, load, modify."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # updated externally by parent app
        self.proj_dir = None

        # works will be allocated once videos are uploaded
        self.works_dict = Dict()
        self.work_is_done_extract_frames = False
        self.work_is_done_check_labels = False  # set this flag for the check labels flow
        # flag; used internally and externally
        self.run_script_video_random = False
        self.run_script_zipped_frames = False
        self.run_script_video_model = False
        self.run_script_check_labels = False # flag for enable the streamlit button 

        # output from the UI
        self.st_extract_status = {}  # 'initialized' | 'active' | 'complete'
        self.st_check_status = "none"
        self.st_video_files_ = []  # list of uploaded video files
        self.st_frame_files_ = []  # list of uploaded zipped frame files
        self.st_submits = 0
        self.st_frame_range = [0, 1]  # limits for frame selection
        self.st_n_frames_per_video = None
        self.model_dir = None  # this will be used for extracting frames given a model
        self.last_execution_time = time.time()
        self.selected_body_parts = ["All"]

    @property
    def st_video_files(self):
        return np.unique(self.st_video_files_).tolist()

    @property
    def st_frame_files(self):
        return np.unique(self.st_frame_files_).tolist()

    def _launch_works(
        self,
        action: str,
        video_files: Optional[list] = None,
        work_kwargs: dict = None,
        testing: bool = False
    ) -> None:

        # launch works (sequentially for now)
        for video_file in video_files:
            video_key = video_file.replace(".", "_")  # keys cannot contain "."
            if video_key not in self.works_dict.keys():
                self.works_dict[video_key] = ExtractFramesWork(
                    cloud_compute=CloudCompute("default"),
                )
            status = self.st_extract_status[video_file]
            if status == "initialized" or status == "active":
                self.st_extract_status[video_file] = "active"
                # extract frames for labeling (automatically reformats video for DALI)
                self.works_dict[video_key].run(
                    action=action,
                    video_file="/" + video_file,
                    **work_kwargs,
                )
                self.st_extract_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done:
                    # kill work
                    _logger.info(f"killing work from video {video_key}")
                    if not testing:  # cannot run stop() from tests for some reason
                        self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

    def _extract_frames_using_kmeans(
        self,
        video_files: Optional[list] = None,
        n_frames_per_video: Optional[int] = None,
        testing: bool = False
    ) -> None:

        self.work_is_done_extract_frames = False

        if not video_files:
            video_files = self.st_video_files
        if not n_frames_per_video:
            n_frames_per_video = self.st_n_frames_per_video

        work_kwargs = {
            'proj_dir': self.proj_dir,
            'n_frames_per_video': n_frames_per_video,
            'frame_range': self.st_frame_range,
        }
        self._launch_works(
            action="extract_frames_using_kmeans",
            video_files=video_files,
            work_kwargs=work_kwargs,
            testing=testing,
        )

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _extract_frames_using_model(
        self,
        video_files: Optional[list] = None,
        n_frames_per_video: Optional[int] = None,
        testing: bool = False,
    ) -> None:

        self.work_is_done_extract_frames = False

        # NOTE: this could lead to problems if self.st_video_files is from uploading raw videos
        # instead of choosing videos that inference has been run on
        # if not video_files:
        #     video_files = self.st_video_files
        if not n_frames_per_video:
            n_frames_per_video = self.st_n_frames_per_video

        work_kwargs = {
            'proj_dir': self.proj_dir,
            'model_dir': self.model_dir,
            'n_frames_per_video': n_frames_per_video,
            'frame_range': self.st_frame_range,
        }
        self._launch_works(
            action="extract_frames_using_model",
            video_files=video_files,
            work_kwargs=work_kwargs,
            testing=testing,
        )

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _unzip_frames(
        self,
        video_files: Optional[list] = None,
        testing: bool = False
    ) -> None:

        self.work_is_done_extract_frames = False

        if not video_files:
            video_files = self.st_frame_files

        work_kwargs = {
            'proj_dir': self.proj_dir,
        }
        self._launch_works(
            action="unzip_frames",
            video_files=video_files,
            work_kwargs=work_kwargs,
            testing=testing,
        )

        self.work_is_done_extract_frames = True

    def _save_annotated_frames(
        self,
        selected_body_parts: list = None,
        testing: bool = False
    ) -> None:
        _logger.info(f"============== triggering save annotated frames ================")
        self.work_is_done_check_labels = False

        # Set prokect status and flag the worker when moving frames
        video_key = "check_labels"  # keys cannot contain "."
        if video_key not in self.works_dict.keys():
            self.works_dict[video_key] = ExtractFramesWork(
                cloud_compute=CloudCompute("default"),
            )
        status = self.st_check_status
        if status == "initialized" or status == "active":
            self.st_check_status = "active"
            # extract frames for labeling (automatically reformats video for DALI)
            self.works_dict[video_key].run(
                action="save_annotated_frames",
                proj_dir=self.proj_dir,
                selected_body_parts=self.selected_body_parts
            )
            self.st_check_status = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done:
                    # kill work
                    _logger.info(f"killing work from {video_key}")
                    if not testing:  # cannot run stop() from tests for some reason
                        self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        self.work_is_done_check_labels = True
        _logger.info(f"============== completed save annotated frames ================")

    def run(self, action: str, **kwargs) -> None:
        if action == "extract_frames":
            self._extract_frames_using_kmeans(**kwargs)
        elif action == "extract_frames_using_model":
            self._extract_frames_using_model(**kwargs)
        elif action == "unzip_frames":
            self._unzip_frames(**kwargs)
        elif action == "save_annotated_frames":
            _logger.info("Starting save_annotated_frames process.")
            self._save_annotated_frames(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)

def find_models(model_dir):
    trained_models = []
    # this returns a list of model training days
    dirs_day = os.listdir(model_dir)
    # loop over days and find HH-MM-SS
    for dir_day in dirs_day:
        fullpath1 = os.path.join(model_dir, dir_day)
        dirs_time = os.listdir(fullpath1)
        for dir_time in dirs_time:
            fullpath2 = os.path.join(fullpath1, dir_time)
            trained_models.append('/'.join(fullpath2.split('/')[-2:]))
    return trained_models


def _render_streamlit_fn(state: AppState):

    st.markdown(
        """
        ## Extract frames for labeling
        """
    )

    if state.run_script_video_random or state.run_script_zipped_frames \
            or state.run_script_video_model:
        # don't autorefresh during large file uploads, only during processing
        st_autorefresh(interval=5000, key="refresh_extract_frames_ui")

    model_dir = os.path.join(state.proj_dir[1:], MODELS_DIR)

    if os.path.exists(model_dir):
        models_list = find_models(os.path.join(state.proj_dir[1:], MODELS_DIR))
    else:
        models_list = []

    if len(models_list) == 0:
        options = [VIDEO_RANDOM_STR, ZIPPED_FRAMES_STR]
    else:
        options = [VIDEO_RANDOM_STR, ZIPPED_FRAMES_STR, VIDEO_MODEL_STR]

    st_mode = st.radio(
        "Select data upload option",
        options=options,
        # disabled=state.st_project_loaded,
        index=0,
    )

    if st_mode == VIDEO_RANDOM_STR:

        # upload video files to temporary directory
        video_dir = os.path.join(state.proj_dir[1:], VIDEOS_TMP_DIR)
        os.makedirs(video_dir, exist_ok=True)

        # allow user to select video through uploading or already-uploaded video
        video_select_option = st.radio(
            "Video selection",
            options=[
                VIDEO_SELECT_NEW,
                VIDEO_SELECT_UPLOADED,
            ],
        )

        if video_select_option == VIDEO_SELECT_NEW:
            # initialize the file uploader
            uploaded_files = st.file_uploader(
                "Select video files",
                type=["mp4", "avi"],
                accept_multiple_files=True,
            )

            if len(uploaded_files) > 0:
                col1, col2, col3 = st.columns(spec=3, gap="medium")
                col1.markdown("**Video Name**")
                col2.markdown("**Video Duration**")
                col3.markdown("**Number of Frames**")

            # for each of the uploaded files
            st_videos = []
            for uploaded_file in uploaded_files:
                # read it
                bytes_data = uploaded_file.read()
                # name it
                filename = uploaded_file.name.replace(" ", "_")
                filepath = os.path.join(video_dir, filename)
                st_videos.append(filepath)
                if not state.run_script_video_random:
                    # write the content of the file to the path, but not while processing
                    with open(filepath, "wb") as f:
                        f.write(bytes_data)

                    # calculate video duration and frame count
                    cap = cv2.VideoCapture(filepath)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = float(frame_count) / float(fps)

                    col1.write(uploaded_file.name)
                    col2.write(f"{duration:.2f} seconds")
                    col3.write(str(frame_count))

                    # relese video
                    cap.release()

        elif video_select_option == VIDEO_SELECT_UPLOADED:

            uploaded_video_dir_train = os.path.join(state.proj_dir[1:], VIDEOS_DIR)
            list_train = []
            if os.path.isdir(uploaded_video_dir_train):
                list_train = [
                    os.path.join(uploaded_video_dir_train, vid)
                    for vid in os.listdir(uploaded_video_dir_train)
                ]

            st_videos = st.multiselect(
                "Select video files",
                list_train,
                help="Videos in the 'videos' directory have been previously uploaded for "
                     "frame extraction.",
                format_func=lambda x: "/".join(x.split("/")[-1:]),
            )

        # insert an empty element to create empty space
        st.markdown("###")

        col0, col1 = st.columns(2, gap="large")
        with col0:
            # select number of frames to label per video
            n_frames_per_video = st.text_input(
                "Frames to label per video", 20,
                help="Specify the desired number of frames for labeling per video. "
                     "The app will select frames to maximize the diversity of animal poses "
                     "captured within each video."
            )
            st_n_frames_per_video = int(n_frames_per_video)
        with col1:
            # select range of video to pull frames from
            st_frame_range = st.slider(
                "Portion of video used for frame selection", 0.0, 1.0, (0.0, 1.0),
                help="Focus on selecting video sections where the animals are clearly visible and "
                     "performing the desired behaviors. "
                     "Skip any parts without the animals or with distracting elements like hands, "
                     "as these can confuse the model."
            )

        st_submit_button = st.button(
            "Extract frames",
            key="extract_frames_random",
            disabled=(
                (st_n_frames_per_video == 0)
                or len(st_videos) == 0
                or state.run_script_video_random
            )
        )
        if state.run_script_video_random:
            keys = [k for k, _ in state.works_dict.items()]  # cannot directly call keys()?
            for vid, status in state.st_extract_status.items():
                if status == "initialized":
                    p = 0.0
                elif status == "active":
                    vid_ = vid.replace(".", "_")
                    if vid_ in keys:
                        try:
                            p = state.works_dict[vid_].progress
                        except KeyError:
                            p = 100.0  # if work is deleted while accessing
                    else:
                        p = 100.0  # state.work.progress
                elif status == "complete":
                    p = 100.0
                else:
                    st.text(status)
                st.progress(p / 100.0, f"{vid} progress ({status}: {int(p)}\% complete)")
            st.warning("waiting for existing extraction to finish")
            state.last_execution_time = time.time()

        if state.st_submits > 0 and not st_submit_button and not state.run_script_video_random:
            if time.time() - state.last_execution_time < 10:
                st.markdown(PROCEED_FMT % PROCEED_STR, unsafe_allow_html=True)
        
        # Lightning way of returning the parameters
        if st_submit_button:

            state.st_submits += 1

            state.st_video_files_ = st_videos
            state.st_extract_status = {s: 'initialized' for s in st_videos}
            state.st_n_frames_per_video = st_n_frames_per_video
            state.st_frame_range = st_frame_range
            st.text("Request submitted!")
            state.run_script_video_random = True  # must the last to prevent race condition

            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

    elif st_mode == ZIPPED_FRAMES_STR:
        # upload zipped files to temporary directory
        frames_dir = os.path.join(state.proj_dir[1:], ZIPPED_TMP_DIR)
        os.makedirs(frames_dir, exist_ok=True)

        # initialize the file uploader
        uploaded_files = st.file_uploader(
            "Select zipped folders",
            type="zip",
            accept_multiple_files=True,
            help="Upload one zip file per video. The file name should be the"
                 " name of the video. The frames should be in the format 'img%08i.png',"
                 " i.e. a png file with a name that starts with 'img' and contains the"
                 " frame number with leading zeros such that there are 8 total digits"
                 " (e.g. 'img00003453.png')."
        )

        # for each of the uploaded files
        st_videos = []
        for uploaded_file in uploaded_files:
            # read it
            bytes_data = uploaded_file.read()
            # name it
            filename = uploaded_file.name.replace(" ", "_")
            filepath = os.path.join(frames_dir, filename)
            st_videos.append(filepath)
            if not state.run_script_zipped_frames:
                # write the content of the file to the path, but not while processing
                with open(filepath, "wb") as f:
                    f.write(bytes_data)
            # check files: TODO
            # state.st_error_flag, state.st_error_msg = check_files_in_zipfile(
            #     filepath, project_type=st_prev_format)

        st_submit_button_frames = st.button(
            "Extract frames",
            key="extract_frames_uploaded",
            disabled=len(st_videos) == 0 or state.run_script_zipped_frames,
        )

        if (
            state.st_submits > 0
            and not st_submit_button_frames
            and not state.run_script_zipped_frames
        ):
            if time.time() - state.last_execution_time < 10:
                st.markdown(PROCEED_FMT % PROCEED_STR, unsafe_allow_html=True)
                shutil.rmtree(frames_dir)

        # Lightning way of returning the parameters
        if st_submit_button_frames:

            state.st_submits += 1
            state.last_execution_time = time.time()
            state.st_frame_files_ = st_videos
            state.st_extract_status = {s: 'initialized' for s in st_videos}
            st.text("Request submitted!")
            state.run_script_zipped_frames = True  # must the last to prevent race condition

            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

    elif st_mode == VIDEO_MODEL_STR:

        selected_model_path = st.selectbox("Select a model", models_list)

        base_model_dir = abspath(os.path.join(
            state.proj_dir[1:], MODELS_DIR, selected_model_path, MODEL_VIDEO_PREDS_INFER_DIR
        ))

        if not os.path.exists(base_model_dir):
            st.text(
                "No video predictions avalibale for this model. "
                "Go to TRAIN/INFER tab to run infrence"
            )
            files = []
        else:
            files = os.listdir(base_model_dir)

        video_list = []
        if len(files) > 0:
            for file in files:
                if not file.endswith(".csv"):
                    continue
                if file.endswith("temporal_norm.csv") \
                        or file.endswith("error.csv") \
                        or file.endswith("short.csv"):
                    continue
                video_list.append(file.split(".")[0])

        # upload video files to temporary directory
        video_dir = os.path.join(state.proj_dir[1:], VIDEOS_TMP_DIR)
        os.makedirs(video_dir, exist_ok=True)

        if len(video_list) > 0:
            st_videos = st.multiselect("Select videos", video_list)

            # insert an empty element to create empty space
            st.markdown("##")

            col0, col1 = st.columns(2, gap="large")
            with col0:
                # select number of frames to label per video
                n_frames_per_video = st.text_input(
                    "Frames to label per video", 20,
                    help="Specify the desired number of frames for labeling per video. "
                         "The app will select frames to maximize the diversity of animal poses "
                         "captured within each video."
                )
                st_n_frames_per_video = int(n_frames_per_video)
            with col1:
                # select range of video to pull frames from
                st_frame_range = st.slider(
                    "Portion of video used for frame selection",
                    0.0, 1.0, (0.0, 1.0),
                    help="Focus on selecting video sections where the animals "
                         "are clearly visible and performing the desired behaviors. "
                         "Skip any parts without the animals or with distracting "
                         "elements like hands, as these can confuse the model."
                )

            st_submit_button_model_frames = st.button(
                "Extract frames",
                key="extract_frames_model",
                disabled=(
                    (st_n_frames_per_video == 0)
                    or len(st_videos) == 0
                    or state.run_script_video_model
                )
            )
            if state.run_script_video_model:
                keys = [k for k, _ in state.works_dict.items()]  # cannot directly call keys()?
                for vid, status in state.st_extract_status.items():
                    if status == "initialized":
                        p = 0.0
                    elif status == "active":
                        vid_ = vid.replace(".", "_")
                        if vid_ in keys:
                            try:
                                p = state.works_dict[vid_].progress
                            except KeyError:
                                p = 100.0  # if work is deleted while accessing
                        else:
                            p = 100.0  # state.work.progress
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)
                    st.progress(p / 100.0, f"{vid} progress ({status}: {int(p)}\% complete)")
                st.warning("waiting for existing extraction to finish")
                state.last_execution_time = time.time()

            if state.st_submits > 0 and not st_submit_button_model_frames \
                    and not state.run_script_video_model:
                if time.time() - state.last_execution_time < 10:
                    st.markdown(PROCEED_FMT % PROCEED_STR, unsafe_allow_html=True)

            # Lightning way of returning the parameters
            if st_submit_button_model_frames:
                state.st_submits += 1
                base_rel_path = os.path.join(state.proj_dir[1:], VIDEOS_INFER_DIR)
                state.st_video_files_ = [
                    os.path.join(base_rel_path, s + ".mp4") for s in st_videos
                ]
                state.model_dir = base_model_dir
                state.st_extract_status = {s: 'initialized' for s in state.st_video_files_}
                state.st_n_frames_per_video = st_n_frames_per_video
                state.st_frame_range = st_frame_range
                st.text("Request submitted!")
                state.run_script_video_model = True  # must the last to prevent race condition

            #     #force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f2f5;
        }
        .navbar {
            background-color: #3b5998;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
            padding: 20px;
            margin: 20px 0;
        }
        .card img {
            width: 100%;
            border-radius: 8px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 4px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .frame-counter {
            font-size: 20px;
            font-weight: normal;
            margin: 10px 0;
            text-align: center;
        }
        .image-container img {
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            margin-bottom: 10px;
            display: block;
        }
        .nav-buttons {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }
        .st-expander {
            background-color: white !important;
            border-radius: 8px;
            padding: 20px !important;
        }
        .centered-container {
            display: flex;
            justify-content: center;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .explanation {
            font-size: 18px;
            margin: 10px 20px;
            text-align: center;
        }
        .button-explanation {
            font-size: 16px;
            margin-left: 10px;
            margin-top: 10px;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()

    st.header("Check Labeled Frames")
    st_expander = st.expander("Expand for instractions")
    with st_expander:
        st.markdown(
        """
        Ensure accurate annotations with these steps:

        1. **Select Keypoints**: Use the multiselect box to choose keypoints to check.

        2. **Check Labels**: Click "Check labels" to upload and review frames, using arrows to navigate.

        3. **Correct Annotations**: Go back to Label Studio to correct any wrong annotations.

        4. **Download Frames**: Click "Download Frames" to save all annotated frames as a ZIP file.
        """
        )

    collected_data_csv = os.path.join(state.proj_dir[1:], COLLECTED_DATA_FILENAME)

    if os.path.exists(collected_data_csv):
        # Adding multiselect box for body parts
        df = pd.read_csv(collected_data_csv, header=[0, 1, 2])
        body_parts = list(set(df.columns.get_level_values(1).tolist()[1:])) 

        # Adding multiselect box for body parts
        selected_body_parts = st.multiselect(
            "Select body parts to check",
            options=["All"] + body_parts,
            default=["All"],
        )

        st.write("**Click Check labels to upload and review the labeled frames.**")

        st_start_check_labels = st.button(
            "Check labels",
            key="show_annotated_frames",
            disabled=(not os.path.exists(collected_data_csv) or state.run_script_check_labels)
        )
        st.caption("This button will be enabled once labeled frames are available for review.")
        if st_start_check_labels:
            state.st_check_status = "initialized"
            st.text("Request submitted!")
            state.selected_body_parts = selected_body_parts 
            state.run_script_check_labels = True

        st_autorefresh(interval=2000, key="refresh_check_labels")

        labeled_data_check_path = os.path.join(state.proj_dir[1:], 'labeled-data-check')

        if os.path.exists(labeled_data_check_path):
            if 'frame_index' not in st.session_state:
                st.session_state.frame_index = 0

            try:
                video_names = os.listdir(labeled_data_check_path)
            except Exception as e:
                st.error(f"Error reading directory: {e}")
                video_names = []

            if video_names:
                selected_video = st.selectbox("Select a video", video_names)
                frame_paths = get_frame_paths(os.path.join(labeled_data_check_path, selected_video))

                if frame_paths:
                    total_frames = len(frame_paths)
                    st.session_state.frame_index = max(0, min(st.session_state.frame_index, total_frames - 1))
                    current_frame_path = frame_paths[st.session_state.frame_index]

                    if os.path.exists(current_frame_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(current_frame_path, use_column_width=True)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Image not found: {current_frame_path}")

                    st.markdown('<div class="nav-buttons centered-container">', unsafe_allow_html=True)
                    if total_frames > 1:
                        _, col_prev, col_frame_count, col_next, _ = st.columns([2, 1, 2, 1, 2])
                        with col_prev:
                            if st.button("←", key="prev_button"):
                                if st.session_state.frame_index > 0:
                                    st.session_state.frame_index -= 1
                        with col_frame_count:
                            st.markdown(f'<span class="frame-counter">Frame {st.session_state.frame_index + 1} of {total_frames}</span>', unsafe_allow_html=True)
                        with col_next:
                            if st.button("→", key="next_button"):
                                if st.session_state.frame_index < total_frames - 1:
                                    st.session_state.frame_index += 1
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No frames found in the selected video folder.")
                    
                    col_left_outer, col_left, col_middle, col_right, col_right_outer = st.columns([2, 1, 2, 1, 2])
                    with col_right_outer:
                        # Provide a download button for annotated images
                        zip_buffer = zip_annotated_images(os.path.join(labeled_data_check_path, selected_video))
                        st.download_button(
                            label="Download All Frames",
                            data=zip_buffer.getvalue(),
                            file_name="labeled_data_check.zip",
                            mime="application/zip"
                        )
            else:
                st.warning("No annotated frames found.")
    else:
        st.warning("Annotated frames directory does not exist.")


