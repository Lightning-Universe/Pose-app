import cv2
from lightning import CloudCompute, LightningFlow, LightningWork
from lightning.app.storage import FileSystem
from lightning.app.structures import Dict
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.state import AppState
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from lightning_pose_app.utilities import StreamlitFrontend
from lightning_pose_app.utilities import reencode_video, check_codec_format, get_frames_from_idxs


class ExtractFramesWork(LightningWork):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._drive = FileSystem()

        self.progress = 0.0
        self.progress_delta = 0.5
        self.work_is_done_extract_frames = False

    def get_from_drive(self, inputs):
        for i in inputs:
            print(f"EXTRACT drive get {i}")
            try:  # file may not be ready
                src = i  # shared
                dst = self.abspath(i)  # local
                self._drive.get(src, dst, overwrite=True)
                print(f"drive data saved at {dst}")
            except Exception as e:
                print(e)
                print(f"did not load {i} from drive")
                pass

    def put_to_drive(self, outputs):
        for o in outputs:
            print(f"EXTRACT drive try put {o}")
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
            print(f"EXTRACT drive success put {dst}")

    @staticmethod
    def abspath(path):
        if path[0] == "/":
            path_ = path[1:]
        else:
            path_ = path
        return os.path.abspath(path_)

    def read_nth_frames(self, video_file, n=1, resize_dims=64):

        from tqdm import tqdm

        # Open the video file
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Error opening video file {video_file}")

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

    def select_frame_idxs(self, video_file, resize_dims=64, n_clusters=20, frame_skip=1):

        frames = self.read_nth_frames(video_file, n=frame_skip, resize_dims=resize_dims)
        batches = np.reshape(frames, (frames.shape[0], -1))[:-2]  # leave room for context
        frame_count = batches.shape[0]

        # take temporal diffs
        print('computing motion energy...')
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
        print('performing pca over high motion energy frames...')
        pca_obj = PCA(n_components=np.min([batches[idxs_high_me].shape[0], 32]))
        embedding = pca_obj.fit_transform(X=batches[idxs_high_me])
        del batches  # free up memory

        # cluster low-d pca embeddings
        print('performing kmeans clustering...')
        kmeans_obj = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans_obj.fit(X=embedding)

        # find high me frame that is closest to each cluster center
        # kmeans_obj.cluster_centers_ is shape (n_clusters, n_pcs)
        centers = kmeans_obj.cluster_centers_.T[None, ...]
        # embedding is shape (n_frames, n_pcs)
        dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
        # dists is shape (n_frames, n_clusters)
        idxs_prototypes_ = np.argmin(dists, axis=0)
        # now index into high me frames to get overall indices
        idxs_prototypes = idxs_high_me[idxs_prototypes_]

        return idxs_prototypes

    @staticmethod
    def export_frames(
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

    def _extract_frames(self, video_file, proj_dir, n_frames_per_video):

        print(f"============== extracting frames from {video_file} ================")

        # set flag for parent app
        self.work_is_done_extract_frames = False

        # pull video from FileSystem
        self.get_from_drive([video_file])

        data_dir_rel = os.path.join(proj_dir, "labeled-data")
        data_dir = self.abspath(data_dir_rel)
        n_digits = 8
        extension = "png"
        context_frames = 2

        # check: does file exist?
        video_file_abs = self.abspath(video_file)
        video_file_exists = os.path.exists(video_file_abs)
        print(f"video file exists? {video_file_exists}")
        if not video_file_exists:
            print("skipping frame extraction")
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
        idxs_selected = self.select_frame_idxs(
            video_file=video_file_abs, resize_dims=resize_dims, n_clusters=n_frames_per_video)

        # save csv file inside same output directory
        frames_to_label = np.array([
            "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected])
        np.savetxt(
            os.path.join(save_dir, "selected_frames.csv"),
            np.sort(frames_to_label),
            delimiter=",",
            fmt="%s"
        )

        # save frames
        self.export_frames(
            video_file=video_file_abs, save_dir=save_dir, frame_idxs=idxs_selected,
            format=extension, n_digits=n_digits, context_frames=context_frames)

        # push extracted frames to drive
        self.put_to_drive([data_dir_rel])

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def _reformat_video(self, video_file, **kwargs):

        # pull videos from FileSystem
        self.get_from_drive([video_file])
        video_file_abs = self.abspath(video_file)

        # check 1: does file exist?
        video_file_exists = os.path.exists(video_file_abs)
        if not video_file_exists:
            print(f"{video_file_abs} does not exist! skipping")
            return

        # check 2: is file in the correct format for DALI?
        video_file_correct_codec = check_codec_format(video_file_abs)

        # get new name (ensure mp4 file extension, no tmp directory)
        ext = os.path.splitext(os.path.basename(video_file))[1]
        video_file_new = video_file.replace(f"{ext}", ".mp4").replace("videos_tmp", "videos")
        video_file_abs_new = self.abspath(video_file_new)

        # reencode/rename
        if not video_file_correct_codec:
            print("re-encoding video to be compatable with Lightning Pose video reader")
            reencode_video(video_file_abs, video_file_abs_new)
            # remove old video from local files
            # os.remove(video_file_abs)
        else:
            # make dir to write into
            os.makedirs(os.path.dirname(video_file_abs_new), exist_ok=True)
            # rename
            os.rename(video_file_abs, video_file_abs_new)

        # remove old video from FileSystem
        # self._drive.rm(video_file)

        # push possibly reformated, renamed videos to FileSystem
        self.put_to_drive([video_file_new])

        return video_file_new

    def run(self, action, **kwargs):
        if action == "reformat_video":
            self._reformat_video(**kwargs)
        elif action == "extract_frames":
            self._reformat_video(**kwargs)
            self._extract_frames(**kwargs)
        else:
            pass


class ExtractFramesUI(LightningFlow):
    """UI to set up project."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # self.work = ExtractFramesWork(
        #     cloud_compute=CloudCompute("default"),
        #     parallel=parallel,
        # )

        # works for frame extraction
        self.works_dict = Dict()
        self.work_is_done_extract_frames = False

        self._drive = FileSystem()

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script = False

        # send info to user
        self.st_video_files_ = []
        self.st_extract_status = {}  # 'initialized' | 'active' | 'complete'

        # save parameters for later run
        self.proj_dir = None

        # output from the UI
        self.st_submits = 0
        self.st_n_frames_per_video = None

    @property
    def st_video_files(self):
        return np.unique(self.st_video_files_).tolist()

    def _push_video(self, video_file):
        if video_file[0] == "/":
            src = os.path.join(os.getcwd(), video_file[1:])
            dst = video_file
        else:
            src = os.path.join(os.getcwd(), video_file)
            dst = "/" + video_file
        if not self._drive.isfile(dst):
            self._drive.put(src, dst)

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
                )
                self.st_extract_status[video_file] = "complete"

        # clean up works
        while len(self.works_dict) > 0:
            for video_key in list(self.works_dict):
                if (video_key in self.works_dict.keys()) \
                        and self.works_dict[video_key].work_is_done_extract_frames:
                    # kill work
                    print(f"killing work from video {video_key}")
                    self.works_dict[video_key].stop()
                    del self.works_dict[video_key]

        # set flag for parent app
        self.work_is_done_extract_frames = True

    def run(self, action, **kwargs):
        if action == "push_video":
            self._push_video(**kwargs)
        elif action == "extract_frames":
            self._extract_frames(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    st.markdown(
        """
        ## Extract frames for labeling
        """
    )

    if state.run_script:
        # don't autorefresh during large file uploads, only during processing
        st_autorefresh(interval=5000, key="refresh_extract_frames_ui")

    # upload video files to temporary directory
    video_dir = os.path.join(state.proj_dir[1:], "videos_tmp")
    os.makedirs(video_dir, exist_ok=True)

    # initialize the file uploader
    uploaded_files = st.file_uploader("Select video files", accept_multiple_files=True)

    # for each of the uploaded files
    st_videos = []
    for uploaded_file in uploaded_files:
        # read it
        bytes_data = uploaded_file.read()
        # name it
        filename = uploaded_file.name.replace(" ", "_")
        filepath = os.path.join(video_dir, filename)
        st_videos.append(filepath)
        if not state.run_script:
            # write the content of the file to the path, but not while processing
            with open(filepath, "wb") as f:
                f.write(bytes_data)

    # select number of frames to label per video
    n_frames_per_video = st.text_input("Frames to label per video", 20)
    st_n_frames_per_video = int(n_frames_per_video)

    st_submit_button = st.button(
        "Extract frames",
        disabled=(st_n_frames_per_video == 0) or len(st_videos) == 0 or state.run_script
    )
    if state.run_script:
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

    if state.st_submits > 0 and not st_submit_button and not state.run_script:
        proceed_str = "Please proceed to the next tab to label frames."
        proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
        st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

    # Lightning way of returning the parameters
    if st_submit_button:

        state.st_submits += 1

        state.st_video_files_ = st_videos
        state.st_extract_status = {s: 'initialized' for s in st_videos}
        state.st_n_frames_per_video = st_n_frames_per_video
        st.text("Request submitted!")
        state.run_script = True  # must the last to prevent race condition

        # force rerun to show "waiting for existing..." message
        st_autorefresh(interval=2000, key="refresh_extract_frames_after_submit")
