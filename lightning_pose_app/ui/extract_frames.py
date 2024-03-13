import cv2
from lightning.app import CloudCompute, LightningFlow, LightningWork
from lightning.app.structures import Dict
from lightning.app.utilities.state import AppState
import logging
import numpy as np
import os
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import zipfile

from lightning_pose_app import (
    LABELED_DATA_DIR,
    MODELS_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
    VIDEOS_INFER_DIR,
    VIDEOS_TMP_DIR,
    ZIPPED_TMP_DIR,
)
from lightning_pose_app.utilities import StreamlitFrontend, abspath
from lightning_pose_app.utilities import copy_and_reformat_video, get_frames_from_idxs

_logger = logging.getLogger('APP.EXTRACT_FRAMES')


class ExtractFramesWork(LightningWork):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.progress = 0.0
        self.progress_delta = 0.5
        self.work_is_done_extract_frames = False

        # updated externally by parent app
        self.trained_models = []
        self.proj_dir = None
        self.config_dict = None

    def _read_nth_frames(
        self,
        video_file: str,
        n: int = 1,
        resize_dims: int = 64,
    ) -> np.ndarray:

        from tqdm import tqdm

        # Open the video file
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            _logger.error(f"Error opening video file {video_file}")

        frames = []
        frame_counter = 0
        frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        with tqdm(total=int(frame_total)) as pbar:
            while cap.isOpened():
                # Read the next frame
                ret, frame = cap.read()
                if ret:
                    # If the frame was successfully read, then process it
                    if frame_counter % n == 0:
                        frame_resize = cv2.resize(frame, (resize_dims, resize_dims))
                        frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
                        frames.append(frame_gray.astype(np.float16))
                    frame_counter += 1
                    progress = frame_counter / frame_total * 100.0
                    # periodically update progress
                    if round(progress, 4) - self.progress >= self.progress_delta:
                        if progress > 100:
                            self.progress = 100.0
                        else:
                            self.progress = round(progress, 4)
                    pbar.update(1)
                else:
                    # If we couldn't read a frame, we've probably reached the end
                    break

        # When everything is done, release the video capture object
        cap.release()

        return np.array(frames)

    def _select_frame_idxs(
        self,
        video_file: str,
        resize_dims: int = 64,
        n_clusters: int = 20,
        frame_skip: int = 1,
        frame_range: list = [0, 1],
    ) -> np.ndarray:

        # check inputs
        if frame_skip != 1:
            raise NotImplementedError
        assert frame_range[0] >= 0
        assert frame_range[1] <= 1

        # read all frames, reshape, chop off unwanted portions of beginning/end
        frames = self._read_nth_frames(video_file, n=frame_skip, resize_dims=resize_dims)
        frame_count = frames.shape[0]
        beg_frame = int(float(frame_range[0]) * frame_count)
        end_frame = int(float(frame_range[1]) * frame_count) - 2  # leave room for context
        assert (end_frame - beg_frame) >= n_clusters, "valid video segment too short!"
        batches = np.reshape(frames, (frames.shape[0], -1))[beg_frame:end_frame]

        # take temporal diffs
        _logger.info('computing motion energy...')
        me = np.concatenate([
            np.zeros((1, batches.shape[1])),
            np.diff(batches, axis=0)
        ])
        # take absolute values and sum over all pixels to get motion energy
        me = np.sum(np.abs(me), axis=1)

        # find high me frames, defined as those with me larger than nth percentile me
        prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
        idxs_high_me = np.where(me > np.percentile(me, prctile))[0]

        # compute pca over high me frames
        _logger.info('performing pca over high motion energy frames...')
        pca_obj = PCA(n_components=np.min([batches[idxs_high_me].shape[0], 32]))
        embedding = pca_obj.fit_transform(X=batches[idxs_high_me])
        del batches  # free up memory

        # cluster low-d pca embeddings
        _logger.info('performing kmeans clustering...')
        kmeans_obj = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans_obj.fit(X=embedding)

        # find high me frame that is closest to each cluster center
        # kmeans_obj.cluster_centers_ is shape (n_clusters, n_pcs)
        centers = kmeans_obj.cluster_centers_.T[None, ...]
        # embedding is shape (n_frames, n_pcs)
        dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
        # dists is shape (n_frames, n_clusters)
        idxs_prototypes_ = np.argmin(dists, axis=0)
        # now index into high me frames to get overall indices, add offset
        idxs_prototypes = idxs_high_me[idxs_prototypes_] + beg_frame

        return idxs_prototypes

    @staticmethod
    def _select_frame_idxs_using_model(
        video_file: str,
        proj_dir: str,
        model_dir: str,
        n_frames_per_video: int,
        frame_range: list,
    ):
        # TODO: put real active learning code here
        # TODO: make sure to update test by making dummy prediction/metric files
        return np.arange(n_frames_per_video)

    @staticmethod
    def _export_frames(
        video_file: str,
        save_dir: str,
        frame_idxs: np.ndarray,
        format: str = "png",
        n_digits: int = 8,
        context_frames: int = 0,
    ) -> None:
        """

        Parameters
        ----------
        video_file: absolute path to video file from which to select frames
        save_dir: absolute path to directory in which selected frames are saved
        frame_idxs: indices of frames to grab
        format: only "png" currently supported
        n_digits: number of digits in image names
        context_frames: number of frames on either side of selected frame to also save

        """

        cap = cv2.VideoCapture(video_file)

        # expand frame_idxs to include context frames
        if context_frames > 0:
            context_vec = np.arange(-context_frames, context_frames + 1)
            frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
            frame_idxs.sort()
            frame_idxs = frame_idxs[frame_idxs >= 0]
            frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
            frame_idxs = np.unique(frame_idxs)

        # load frames from video
        frames = get_frames_from_idxs(cap, frame_idxs)

        # save out frames
        os.makedirs(save_dir, exist_ok=True)
        for frame, idx in zip(frames, frame_idxs):
            cv2.imwrite(
                filename=os.path.join(save_dir, "img%s.%s" % (str(idx).zfill(n_digits), format)),
                img=frame[0],
            )

    def _extract_frames(
        self,
        method: str,
        video_file: str,
        proj_dir: str,
        n_frames_per_video: int,
        frame_range: list = [0, 1],
        model_dir: str = "None",
    ) -> None:

        _logger.info(f"============== extracting frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done_extract_frames = False

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
        os.makedirs(save_dir, exist_ok=True)

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
            idxs_selected = self._select_frame_idxs(
                video_file=video_file_abs,
                resize_dims=resize_dims,
                n_clusters=n_frames_per_video,
                frame_range=frame_range,
            )
        elif method == "active":
            idxs_selected = self._select_frame_idxs_using_model(
                video_file=video_file_abs,
                proj_dir=proj_dir,
                model_dir=model_dir,
                n_frames_per_video=n_frames_per_video,
                frame_range=frame_range,
            )

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
        self._export_frames(
            video_file=video_file_abs, save_dir=save_dir, frame_idxs=idxs_selected,
            format=extension, n_digits=n_digits, context_frames=context_frames)

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _unzip_frames(
        self,
        video_file: str,
        proj_dir: str,
    ) -> None:

        _logger.info(f"============== unzipping frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done_extract_frames = False

        data_dir_rel = os.path.join(proj_dir, LABELED_DATA_DIR)
        if not os.path.exists(data_dir_rel):
            data_dir = abspath(data_dir_rel)
        else:
            data_dir = data_dir_rel
        # TODO
        # n_digits = 8
        # extension = "png"

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

        # save all contents to data directory
        # don't use copytree as the destination dir may already exist
        files = os.listdir(unzipped_dir)
        for file in files:
            src = os.path.join(unzipped_dir, file)
            dst = os.path.join(save_dir, file)
            shutil.copyfile(src, dst)

        # TODO:
        # - if SELECTED_FRAMES_FILENAME does not exist, assume all frames are for labeling and
        #   make this file

        # # save csv file inside same output directory
        # frames_to_label = np.array([
        #     "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected])
        # np.savetxt(
        #     os.path.join(save_dir, SELECTED_FRAMES_FILENAME),
        #     np.sort(frames_to_label),
        #     delimiter=",",
        #     fmt="%s"
        # )

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def run(self, action, **kwargs):
        if action == "extract_frames":
            new_vid_file = copy_and_reformat_video(
                video_file=abspath(kwargs["video_file"]),
                dst_dir=abspath(os.path.join(kwargs["proj_dir"], VIDEOS_DIR)),
            )
            # save relative rather than absolute path
            kwargs["video_file"] = '/'.join(new_vid_file.split('/')[-4:])
            self._extract_frames(method="random", **kwargs)
        elif action == "extract_frames_using_model":
            self._extract_frames(method="active", **kwargs)
        elif action == "unzip_frames":
            # TODO: maybe we need to reformat the file names?
            self._unzip_frames(**kwargs)
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

        # flag; used internally and externally
        self.run_script_video_random = False
        self.run_script_zipped_frames = False
        self.run_script_video_model = False

        # output from the UI
        self.st_extract_status = {}  # 'initialized' | 'active' | 'complete'
        self.st_video_files_ = []  # list of uploaded video files
        self.st_frame_files_ = []  # list of uploaded zipped frame files
        self.st_submits = 0
        self.st_frame_range = [0, 1]  # limits for frame selection
        self.st_n_frames_per_video = None
        self.model_dir = None  # this will be used for extracting frames given a model

    @property
    def st_video_files(self):
        return np.unique(self.st_video_files_).tolist()

    @property
    def st_frame_files(self):
        return np.unique(self.st_frame_files_).tolist()

    def _launch_works(self, action, video_files, work_kwargs, testing=False):

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
                        and self.works_dict[video_key].work_is_done_extract_frames:
                    # kill work
                    _logger.info(f"killing work from video {video_key}")
                    if not testing:  # cannot run stop() from tests for some reason
                        self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _extract_frames(self, video_files=None, n_frames_per_video=None, testing=False):

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
            action="extract_frames",
            video_files=video_files,
            work_kwargs=work_kwargs,
            testing=testing,
        )

    def _extract_frames_using_model(
            self, video_files=None, n_frames_per_video=None ,testing=False):

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

    def _unzip_frames(self, video_files=None):

        self.work_is_done_extract_frames = False

        if not video_files:
            video_files = self.st_frame_files

        work_kwargs = {
            'proj_dir': self.proj_dir,
        }
        self._launch_works(
            video_files=video_files,
            work_kwargs=work_kwargs,
            testing=testing,
        )

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def run(self, action, **kwargs):
        print(action)
        print(kwargs)
        if action == "extract_frames":
            self._extract_frames(**kwargs)
        elif action == "extract_frames_using_model":
            self._extract_frames_using_model(**kwargs)
        elif action == "unzip_frames":
            self._unzip_frames(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    st.markdown(
        """
        ## Extract frames for labeling
        """
    )

    if state.run_script_video_random or state.run_script_zipped_frames or state.run_script_video_model:
        # don't autorefresh during large file uploads, only during processing
        st_autorefresh(interval=5000, key="refresh_extract_frames_ui")

    VIDEO_RANDOM_STR = "Upload videos and automatically extract random frames"
    ZIPPED_FRAMES_STR = "Upload zipped files of frames"
    VIDEO_MODEL_STR = "Automatically extract frames using a given model"

    @st.cache_resource
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
        
    models_list = find_models(os.path.join(state.proj_dir[1:],MODELS_DIR))

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
                        except:
                            p = 100.0  # if work is deleted while accessing
                    else:
                        p = 100.0  # state.work.progress
                elif status == "complete":
                    p = 100.0
                else:
                    st.text(status)
                st.progress(p / 100.0, f"{vid} progress ({status}: {int(p)}\% complete)")
            st.warning("waiting for existing extraction to finish")

        if state.st_submits > 0 and not st_submit_button and not state.run_script_video_random:
            proceed_str = "Please proceed to the next tab to label frames."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

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
            help="Upload one zip file per video. The file name should be the name of the video. "
                 "The frames should be in the format 'img%08i.png', i.e. a png file with a name "
                 "that starts with 'img' and contains the frame number with leading zeros such "
                 "that there are 8 total digits (e.g. 'img00003453.png')."
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
            disabled=len(st_videos) == 0 or state.run_script_zipped_frames,
        )

        if (
            state.st_submits > 0
            and not st_submit_button_frames
            and not state.run_script_zipped_frames
        ):
            proceed_str = "Please proceed to the next tab to label frames."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

        # Lightning way of returning the parameters
        if st_submit_button_frames:

            state.st_submits += 1

            state.st_frame_files_ = st_videos
            state.st_extract_status = {s: 'initialized' for s in st_videos}
            st.text("Request submitted!")
            state.run_script_zipped_frames = True  # must the last to prevent race condition

            # force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

    elif st_mode == VIDEO_MODEL_STR:
        
        selected_model_path = st.selectbox("Select a model", models_list)

        


        # TODO (see above, L530)
        # first, we only want the user to see a dropdown menu with available models
        #   - copy over the function that shmuel wrote for video reader

        """
        once they select a model, show all available videos that have inference results
           x look in model_dir/video_preds, get all vids from here (skip this for now, too hard)
           - look in model_dir/video_preds_infer, get all vids from here (don't grab "short" vids)
               - find all .csv files 
               - skip videos with the string ".short.mp4"
        make sure to return model directory as the variable "model_dir", which gets stored below
        """
        base_model_dir = abspath(os.path.join(state.proj_dir[1:], MODELS_DIR, selected_model_path, MODEL_VIDEO_PREDS_INFER_DIR))
        if not os.path.exists(base_model_dir):
            st.text("No video predictions avalibale for this model. Go to TRAIN/INFER tab to run infrence")
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


##################################################################################
############    Use video uploader for upload videos for the active learning process
#       


        # upload video files to temporary directory
        video_dir = os.path.join(state.proj_dir[1:], VIDEOS_TMP_DIR)
        os.makedirs(video_dir, exist_ok=True)

        if len(video_list)>0:
            st_videos = st.multiselect("Select videos", video_list)
        
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

            st_submit_button_model_frames = st.button(
                "Extract frames",
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
                            except:
                                p = 100.0  # if work is deleted while accessing
                        else:
                            p = 100.0  # state.work.progress
                    elif status == "complete":
                        p = 100.0
                    else:
                        st.text(status)
                    st.progress(p / 100.0, f"{vid} progress ({status}: {int(p)}\% complete)")
                st.warning("waiting for existing extraction to finish")

            if state.st_submits > 0 and not st_submit_button_model_frames and not state.run_script_video_model:
                proceed_str = "Please proceed to the next tab to label frames."
                proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
                st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

            # Lightning way of returning the parameters
            if st_submit_button_model_frames:
                state.st_submits += 1
                base_rel_path = os.path.join(state.proj_dir[1:], VIDEOS_INFER_DIR)
                state.st_video_files_ = [os.path.join(base_rel_path, s + ".mp4") for s in st_videos]
                state.model_dir = base_model_dir
                state.st_extract_status = {s: 'initialized' for s in state.st_video_files_}
                state.st_n_frames_per_video = st_n_frames_per_video
                state.st_frame_range = st_frame_range
                st.text("Request submitted!")
                state.run_script_video_model = True  # must the last to prevent race condition

            #     #force rerun to show "waiting for existing..." message
            st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

#######################################################################################
########## Select the raw avalibale videos per model and extract frames 

        # if len(good_files) > 0:
                
        #         #mp4_files = [os.path.join(base_model_dir, f) for f in files if f.endswith(".mp4")]
                
        #         raw_videos_dir = [abspath(os.path.join(state.proj_dir[1:], "videos_infer",f)) for f in files if f.endswith('.mp4') and not f.endswith('short.mp4') and not f.endswith('labeled.mp4')]
        #         #filtered_videos = [f for f in filenames if f.endswith('.mp4') and not f.endswith('short.mp4') and not f.endswith('labeled.mp4')]
        #         selected_file = st.radio('Select an MP4 file:', raw_videos_dir)
        #         st.write(f'You selected: {selected_file}')
        # else:
        #     st.write("No Videos avalibale")



        # good_files is the list we want to show the user

        #mp4_files = [os.path.join(base_model_dir, f) if f.endswith(".mp4") for f in files]

        """
        # allow user to select multiple videos
        # the result is called "st_videos"
        
        # copy over some of the options from VIDEO_RANDOM_STR: frames/video and video slider
        # call these st_videos

        # extract frames button
        """
        # st_submit_button_model_frames = st.button(
        #     "Extract frames",
        #     disabled=(
        #         (st_n_frames_per_video == 0)
        #         or len(st_videos) == 0
        #         or state.run_script_video_random
        #     )

        # # # NOTE: everything below might be ok?
        # if st_submit_button_model_frames:
        #     state.st_submits += 1

        #     base_rel_path = os.path.join(state.proj_dir, VIDEOS_INFER_DIR)
        #     state.st_video_files_ = [os.path.join(base_rel_path, s + ".mp4") for s in st_videos]
        #     state.model_dir = model_dir
        #     state.st_extract_status = {s: 'initialized' for s in st_videos}
        #     st.text("Request submitted!")
        #     state.run_script_video_model = True  # must the last to prevent race condition

        # #     #force rerun to show "waiting for existing..." message
        # st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")

