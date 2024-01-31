import cv2
from lightning.app import CloudCompute, LightningFlow
from lightning.app.storage import FileSystem
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
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

from lightning_pose_app import LABELED_DATA_DIR, VIDEOS_DIR, VIDEOS_TMP_DIR, ZIPPED_TMP_DIR
from lightning_pose_app import SELECTED_FRAMES_FILENAME
from lightning_pose_app.utilities import StreamlitFrontend, WorkWithFileSystem
from lightning_pose_app.utilities import reencode_video, check_codec_format, get_frames_from_idxs


_logger = logging.getLogger('APP.EXTRACT_FRAMES')


class ExtractFramesWork(WorkWithFileSystem):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, name="extract", **kwargs)

        self.progress = 0.0
        self.progress_delta = 0.5
        self.work_is_done_extract_frames = False

    def _read_nth_frames(self, video_file, n=1, resize_dims=64):

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
        video_file,
        resize_dims=64,
        n_clusters=20,
        frame_skip=1,
        frame_range=[0, 1],
    ):

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
    def _export_frames(
        video_file: str,
        save_dir: str,
        frame_idxs: np.ndarray,
        format: str = "png",
        n_digits: int = 8,
        context_frames: int = 0,
    ):
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
            frame_idxs = (frame_idxs.squeeze()[None, :] + context_vec[:, None]).flatten()
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

    def _extract_frames(self, video_file, proj_dir, n_frames_per_video, frame_range=[0, 1]):

        _logger.info(f"============== extracting frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done_extract_frames = False

        # pull video from FileSystem
        self.get_from_drive([video_file])

        data_dir_rel = os.path.join(proj_dir, LABELED_DATA_DIR)
        data_dir = self.abspath(data_dir_rel)
        n_digits = 8
        extension = "png"
        context_frames = 2

        # check: does file exist?
        video_file_abs = self.abspath(video_file)
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

        # push extracted frames to drive
        self.put_to_drive([data_dir_rel])

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _reformat_video(self, video_file, **kwargs):

        # get new names (ensure mp4 file extension, no tmp directory)
        ext = os.path.splitext(os.path.basename(video_file))[1]
        video_file_mp4_ext = video_file.replace(f"{ext}", ".mp4")
        video_file_new = video_file_mp4_ext.replace(VIDEOS_TMP_DIR, VIDEOS_DIR)
        video_file_abs_new = self.abspath(video_file_new)

        # check 0: do we even need to reformat?
        if self._drive.isfile(video_file_new):
            return video_file_new

        # pull videos from FileSystem
        self.get_from_drive([video_file])
        video_file_abs = self.abspath(video_file)

        # check 1: does file exist?
        video_file_exists = os.path.exists(video_file_abs)
        if not video_file_exists:
            _logger.info(f"{video_file_abs} does not exist! skipping")
            return None

        # check 2: is file in the correct format for DALI?
        video_file_correct_codec = check_codec_format(video_file_abs)

        # reencode/rename
        if not video_file_correct_codec:
            _logger.info("re-encoding video to be compatable with Lightning Pose video reader")
            reencode_video(video_file_abs, video_file_abs_new)
            # remove old video from local files
            os.remove(video_file_abs)
        else:
            # make dir to write into
            os.makedirs(os.path.dirname(video_file_abs_new), exist_ok=True)
            # rename
            os.rename(video_file_abs, video_file_abs_new)

        # remove old video(s) from FileSystem
        if self._drive.isfile(video_file):
            self._drive.rm(video_file)
        if self._drive.isfile(video_file_mp4_ext):
            self._drive.rm(video_file_mp4_ext)

        # push possibly reformated, renamed videos to FileSystem
        self.put_to_drive([video_file_new])

        return video_file_new

    def _unzip_frames(self, video_file, proj_dir):
        
        _logger.info(f"============== unzipping frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done_extract_frames = False

        # pull video from FileSystem
        self.get_from_drive([video_file])

        data_dir_rel = os.path.join(proj_dir, LABELED_DATA_DIR)
        data_dir = self.abspath(data_dir_rel)
        # TODO
        # n_digits = 8
        # extension = "png"

        # check: does file exist?
        video_file_abs = self.abspath(video_file)
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

        # push extracted frames to drive
        self.put_to_drive([data_dir_rel])

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def run(self, action, **kwargs):
        if action == "reformat_video":
            self._reformat_video(**kwargs)
        elif action == "extract_frames":
            new_vid_file = self._reformat_video(**kwargs)
            kwargs["video_file"] = new_vid_file
            self._extract_frames(**kwargs)
        elif action == "unzip_frames":
            # TODO: maybe we need to reformat the file names?
            self._unzip_frames(**kwargs)
        else:
            pass


class ExtractFramesUI(LightningFlow):
    """UI to manage projects - create, load, modify."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # shared storage system
        self._drive = FileSystem()

        # updated externally by parent app
        self.proj_dir = None

        # works will be allocated once videos are uploaded
        self.works_dict = Dict()
        self.work_is_done_extract_frames = False

        # flag; used internally and externally
        self.run_script_video_random = False
        self.run_script_zipped_frames = False

        # output from the UI
        self.st_extract_status = {}  # 'initialized' | 'active' | 'complete'
        self.st_video_files_ = []  # list of uploaded video files
        self.st_frame_files_ = []  # list of uploaded zipped frame files
        self.st_submits = 0
        self.st_frame_range = [0, 1]  # limits for frame selection
        self.st_n_frames_per_video = None

    @property
    def st_video_files(self):
        return np.unique(self.st_video_files_).tolist()

    @property
    def st_frame_files(self):
        return np.unique(self.st_frame_files_).tolist()

    def _push_video(self, video_file):
        if video_file[0] == "/":
            src = os.path.join(os.getcwd(), video_file[1:])
            dst = video_file
        else:
            src = os.path.join(os.getcwd(), video_file)
            dst = "/" + video_file
        if not self._drive.isfile(dst) and os.path.exists(src):
            # only put to FileSystem under two conditions:
            # 1. file exists locally; if it doesn't, maybe it has already been deleted for a reason
            # 2. file does not already exist on FileSystem; avoids excessive file transfers
            _logger.debug(f"UI try put {dst}")
            self._drive.put(src, dst)
            _logger.debug(f"UI success put {dst}")

    def _extract_frames(self, video_files=None, n_frames_per_video=None):

        self.work_is_done_extract_frames = False

        if not video_files:
            video_files = self.st_video_files
        if not n_frames_per_video:
            n_frames_per_video = self.st_n_frames_per_video

        # launch works:
        # - sequential if local
        # - parallel if on cloud
        for video_file in video_files:
            video_key = video_file.replace(".", "_")  # keys cannot contain "."
            if video_key not in self.works_dict.keys():
                self.works_dict[video_key] = ExtractFramesWork(
                    cloud_compute=CloudCompute("default"),
                    parallel=is_running_in_cloud(),
                )
            status = self.st_extract_status[video_file]
            if status == "initialized" or status == "active":
                self.st_extract_status[video_file] = "active"
                # move video from ui machine to shared FileSystem
                self._push_video(video_file=video_file)
                # extract frames for labeling (automatically reformats video for DALI)
                self.works_dict[video_key].run(
                    action="extract_frames",
                    video_file="/" + video_file,
                    proj_dir=self.proj_dir,
                    n_frames_per_video=n_frames_per_video,
                    frame_range=self.st_frame_range,
                )
                self.st_extract_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done_extract_frames:
                    # kill work
                    _logger.info(f"killing work from video {video_key}")
                    self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _unzip_frames(self, video_files=None):

        self.work_is_done_extract_frames = False

        if not video_files:
            video_files = self.st_frame_files

        # launch works
        for video_file in video_files:
            video_key = video_file.replace(".", "_")  # keys cannot contain "."
            if video_key not in self.works_dict.keys():
                self.works_dict[video_key] = ExtractFramesWork(
                    cloud_compute=CloudCompute("default"),
                    parallel=is_running_in_cloud(),
                )
                status = self.st_extract_status[video_file]
                if status == "initialized" or status == "active":
                    self.st_extract_status[video_file] = "active"
                    # move file from ui machine to shared FileSystem
                    self._push_video(video_file=video_file)
                    # extract frames for labeling (automatically reformats video for DALI)
                    self.works_dict[video_key].run(
                        action="unzip_frames",
                        video_file="/" + video_file,
                        proj_dir=self.proj_dir,
                    )
                    self.st_extract_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done_extract_frames:
                    # kill work
                    _logger.info(f"killing work from video {video_key}")
                    self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def run(self, action, **kwargs):
        if action == "push_video":
            self._push_video(**kwargs)
        elif action == "extract_frames":
            self._extract_frames(**kwargs)
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

    if state.run_script_video_random or state.run_script_zipped_frames:
        # don't autorefresh during large file uploads, only during processing
        st_autorefresh(interval=5000, key="refresh_extract_frames_ui")

    VIDEO_RANDOM_STR = "Upload videos and automatically extract random frames"
    ZIPPED_FRAMES_STR = "Upload zipped files of frames"
    VIDEO_MODEL_STR = "Upload videos and automatically extract frames using a given model"

    st_mode = st.radio(
        "Select data upload option",
        options=[VIDEO_RANDOM_STR, ZIPPED_FRAMES_STR],
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
            type=['mp4', 'avi'],
            accept_multiple_files=True,
        )

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

        col0, col1 = st.columns(2, gap="large")
        with col0:
            # select number of frames to label per video
            n_frames_per_video = st.text_input("Frames to label per video", 20)
            st_n_frames_per_video = int(n_frames_per_video)
        with col1:
            # select range of video to pull frames from
            st_frame_range = st.slider(
                "Portion of video used for frame selection", 0.0, 1.0, (0.0, 1.0))

        st_submit_button = st.button(
            "Extract frames",
            disabled=(
                (st_n_frames_per_video == 0)
                or len(st_videos) == 0
                or state.run_script_video_random
            ),
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
            st.warning(f"waiting for existing extraction to finish")

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
            type='zip',
            accept_multiple_files=True,
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
        
        if state.st_submits > 0 and not st_submit_button_frames and not state.run_script_zipped_frames:
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
