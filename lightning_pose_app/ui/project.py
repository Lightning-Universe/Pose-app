import copy
import glob
import logging
import os
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from lightning.app import LightningFlow
from lightning.app.utilities.state import AppState
from streamlit_autorefresh import st_autorefresh

from lightning_pose_app import (
    COLLECTED_DATA_FILENAME,
    LABELED_DATA_DIR,
    LABELSTUDIO_DB_DIR,
    LABELSTUDIO_METADATA_FILENAME,
    MODELS_DIR,
    SELECTED_FRAMES_FILENAME,
    VIDEOS_DIR,
)
from lightning_pose_app.backend.video import copy_and_reformat_video_directory
from lightning_pose_app.utilities import (
    StreamlitFrontend,
    abspath,
    compute_batch_sizes,
    compute_resize_dims,
    update_config,
)

from lightning_pose_app.backend.project import (
    extract_frames_from_pkg_slp,
    extract_labels_from_pkg_slp,
    get_keypoints_from_pkg_slp,
    get_keypoints_from_zipfile,
    check_files_in_zipfile,
    collect_dlc_labels,
)

_logger = logging.getLogger('APP.PROJECT')

# options for handling projects
CREATE_STR = "Create new project"
UPLOAD_STR = "Create new project from source (e.g. existing DLC project)"
LOAD_STR = "Load existing project"
DELETE_STR = "Delete existing project"


class ProjectUI(LightningFlow):
    """UI to set up project."""

    def __init__(self, *args, data_dir, default_config_dict, debug=False, **kwargs):

        super().__init__(*args, **kwargs)

        # initialize data_dir if it doesn't yet exist
        os.makedirs(abspath(data_dir), exist_ok=True)

        # save default config info for initializing new projects
        self.default_config_dict = default_config_dict

        # data for project named <PROJ_NAME> will be stored as
        # <data_dir>/<PROJ_NAME> (aka self.project_dir)
        #   ├── labeled-data/
        #   ├── videos/
        #   ├── models/
        #   ├── config.yaml
        #   └── CollectedData.csv
        self.data_dir = data_dir
        self.proj_dir = None
        self.config_name = None
        self.config_file = None
        self.config_dict = None
        self.model_dir = None
        self.trained_models = []
        self.n_labeled_frames = 0
        self.n_total_frames = 0

        # UI info
        self.run_script = False
        self.update_models = False
        self.count = 0  # counter for external app
        self.count_upload_existing = 0
        self.st_submits = 0  # counter for this streamlit app
        self.st_submits_delete = 0  # counter for this streamlit app
        self.initialized_projects = []

        self.st_project_name = None
        self.st_reset_project_name = False
        self.st_create_new_project = False
        self.st_delete_project = False
        self.st_upload_existing_project = False
        self.st_existing_project_format = None
        self.st_upload_existing_project_zippath = None
        self.st_upload_existing_project_slp = None
        self.st_error_flag = False
        self.st_error_msg = ""
        self.st_project_loaded = False
        self.st_new_vals = None

        # config data
        self.st_n_views = 0
        self.st_keypoints_ = []
        self.st_n_keypoints = 0
        self.st_pcasv_columns = []
        self.st_pcamv_columns = []

        # if True, do not expose project options to user, hard-code instead
        self.debug = debug

    @property
    def st_keypoints(self):
        # if len(np.unique(self.st_keypoints_)) == len(self.st_keypoints_):
        #     return self.st_keypoints_
        # else:
        #     return np.unique(self.st_keypoints_).tolist()  # hack to fix duplication bug
        kps = []
        for kp in self.st_keypoints_:
            if kp not in kps:
                kps.append(kp)
        return kps

    @property
    def proj_dir_abs(self):
        return abspath(self.proj_dir)

    def _find_initialized_projects(self):
        # find all directories inside the data_dir; these should be the projects
        # (except labelstudio database)
        projects = os.listdir(abspath(self.data_dir))
        # strip leading dirs to just get project names
        projects = [
            os.path.basename(p) for p in projects
            if not (p.endswith(LABELSTUDIO_DB_DIR) or p.endswith(".txt"))
        ]
        self.initialized_projects = list(np.unique(projects))

    def _update_paths(self, project_name=None, **kwargs):
        if not project_name:
            project_name = self.st_project_name
        # these will all be paths RELATIVE to the Pose-app directory
        if project_name:
            self.proj_dir = os.path.join(self.data_dir, project_name)
            self.config_name = f"model_config_{project_name}.yaml"
            self.config_file = os.path.join(self.proj_dir, self.config_name)
            self.model_dir = os.path.join(self.proj_dir, MODELS_DIR)

    def _update_project_config(self, new_vals_dict=None, **kwargs):
        """triggered by button click in UI"""

        if not new_vals_dict:
            new_vals_dict = self.st_new_vals

        # check to see if config exists; copy default config if not
        if (self.config_file is None) or (not os.path.exists(abspath(self.config_file))):
            _logger.debug(f"no config file at {self.config_file}")
            _logger.debug("loading default config")
            # copy default config
            config_dict = copy.deepcopy(self.default_config_dict)
            # empty out project-specific entries
            config_dict["data"]["image_orig_dims"]["width"] = None
            config_dict["data"]["image_orig_dims"]["height"] = None
            config_dict["data"]["image_resize_dims"]["width"] = None
            config_dict["data"]["image_resize_dims"]["height"] = None
            config_dict["data"]["data_dir"] = None
            config_dict["data"]["num_keypoints"] = None
            config_dict["data"]["keypoints"] = None
            config_dict["data"]["columns_for_singleview_pca"] = None
            config_dict["data"]["mirrored_column_matches"] = None
        else:
            _logger.debug("loading existing config")
            # load existing config
            config_dict = yaml.safe_load(open(abspath(self.config_file)))

        if new_vals_dict is not None:
            config_dict = update_config(config_dict, new_vals_dict)
            # save out updated config file locally
            if not os.path.exists(self.proj_dir_abs):
                os.makedirs(self.proj_dir_abs)
            yaml.dump(config_dict, open(abspath(self.config_file), "w"))

        # save current params
        self.config_dict = config_dict

    def _update_frame_shapes(self):

        from PIL import Image

        # load single frame from labeled data
        imgs = glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*", "*.png")) \
            + glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*", "*.jpg")) \
            + glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*", "*.jpeg")) \
            + glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*.png")) \
            + glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*.jpg")) \
            + glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*.jpeg"))
        if len(imgs) > 0:
            img = imgs[0]
            image = Image.open(img)
            # compute image resize height/width, between 128 and 384
            height_resize = compute_resize_dims(image.height)
            width_resize = compute_resize_dims(image.width)
            # compute batch sizes batch on image size
            train_batch_size, dali_base_seq_len, dali_ctx_seq_len = compute_batch_sizes(
                image.height, image.width,
            )
            self._update_project_config(new_vals_dict={
                "data": {
                    "image_orig_dims": {
                        "height": image.height,
                        "width": image.width,
                    },
                    "image_resize_dims": {
                        "height": height_resize,
                        "width": width_resize,
                    },
                },
                "training": {
                    "train_batch_size": train_batch_size,
                    "val_batch_size": 2 * train_batch_size,
                    "test_batch_size": 2 * train_batch_size,
                },
                "dali": {
                    "base": {
                        "train": {"sequence_length": dali_base_seq_len},
                        "predict": {"sequence_length": 4 * dali_base_seq_len},
                    },
                    "context": {
                        "train": {"batch_size": dali_ctx_seq_len},
                        "predict": {"sequence_length": 4 * dali_ctx_seq_len},
                    },
                },
            })
        else:
            _logger.debug(glob.glob(os.path.join(self.proj_dir_abs, LABELED_DATA_DIR, "*")))
            _logger.debug("did not find labeled data directory")

    def _compute_labeled_frame_fraction(self, timer=0.0):

        metadata_file = os.path.join(self.proj_dir, LABELSTUDIO_METADATA_FILENAME)

        try:
            proj_details = yaml.safe_load(open(abspath(metadata_file), "r"))
            n_labeled_frames = proj_details["n_labeled_tasks"]
            n_total_frames = proj_details["n_total_tasks"]
        except FileNotFoundError:
            _logger.debug(f"could not find {metadata_file}")
            n_labeled_frames = None
            n_total_frames = None
        except Exception as e:
            _logger.warning(e)
            n_labeled_frames = None
            n_total_frames = None

        self.n_labeled_frames = n_labeled_frames
        self.n_total_frames = n_total_frames

    def _load_project_defaults(self, **kwargs):

        # check to see if config exists
        if self.config_file and os.path.exists(abspath(self.config_file)):
            # set values from config
            config_dict = yaml.safe_load(open(abspath(self.config_file)))
            self.st_keypoints_ = config_dict["data"]["keypoints"]
            self.st_n_keypoints = config_dict["data"]["num_keypoints"]
            self.st_pcasv_columns = config_dict["data"]["columns_for_singleview_pca"]
            self.st_pcamv_columns = config_dict["data"]["mirrored_column_matches"]
            self.st_n_views = 1 if len(self.st_pcamv_columns) == 0 else len(self.st_pcamv_columns)
            # save current params
            self.config_dict = config_dict

    def _update_trained_models_list(self, **kwargs):

        if os.path.isdir(abspath(self.model_dir)):
            trained_models = []
            # this returns a list of model training days
            dirs_day = os.listdir(abspath(self.model_dir))
            # loop over days and find HH-MM-SS
            for dir_day in dirs_day:
                fullpath1 = os.path.join(abspath(self.model_dir), dir_day)
                dirs_time = os.listdir(fullpath1)
                for dir_time in dirs_time:
                    fullpath2 = os.path.join(fullpath1, dir_time)
                    trained_models.append('/'.join(fullpath2.split('/')[-2:]))
            self.trained_models = trained_models

    def _upload_existing_project(self, **kwargs):

        # only run once
        if self.count_upload_existing > 0:
            return

        def contains_videos(file_or_dir):
            if os.path.isfile(file_or_dir):
                return False
            else:
                files_or_dirs = os.listdir(file_or_dir)
                if any([f.endswith(".avi") or f.endswith(".mp4") for f in files_or_dirs]):
                    return True
                else:
                    return False

        def find_top_level_dir(initial_path, target_file_name=None, target_dir=None):
            for root, dirs, files in os.walk(initial_path, topdown=True):
                if target_dir:
                    if target_dir in dirs:
                        return root
                else:
                    if target_file_name in files:
                        return root

        finished_copy_files = False
        try:
            if self.st_existing_project_format == "Lightning Pose":

                if not os.path.exists(self.st_upload_existing_project_zippath):
                    _logger.error(
                        f"Could not find zipped project file at \
                        {self.st_upload_existing_project_zippath};"
                        f" aborting"
                    )
                    return

                with zipfile.ZipFile(self.st_upload_existing_project_zippath) as z:
                    unzipped_dir = self.st_upload_existing_project_zippath.replace(".zip", "")
                    z.extractall(path=unzipped_dir)

                top_level_dir = find_top_level_dir(
                    unzipped_dir,
                    target_file_name=COLLECTED_DATA_FILENAME
                )
                files_and_dirs = os.listdir(top_level_dir)
                for file_or_dir in files_and_dirs:
                    src = os.path.join(top_level_dir, file_or_dir)
                    if file_or_dir.endswith(".csv"):
                        # copy labels csv file
                        dst = os.path.join(self.proj_dir_abs, COLLECTED_DATA_FILENAME)
                        shutil.copyfile(src, dst)
                    elif contains_videos(src):
                        # copy videos over, make sure they are in proper format
                        dst_dir = os.path.join(self.proj_dir_abs, file_or_dir)
                        copy_and_reformat_video_directory(src_dir=src, dst_dir=dst_dir)
                    else:
                        # copy other files
                        dst = os.path.join(self.proj_dir_abs, file_or_dir)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copyfile(src, dst)

                # flag finish coping all files
                finished_copy_files = True

            elif self.st_existing_project_format == "DLC":

                if not os.path.exists(self.st_upload_existing_project_zippath):
                    _logger.error(
                        f"Could not find zipped project file at \
                        {self.st_upload_existing_project_zippath};"
                        f" aborting"
                    )
                    return

                with zipfile.ZipFile(self.st_upload_existing_project_zippath) as z:
                    unzipped_dir = self.st_upload_existing_project_zippath.replace(".zip", "")
                    z.extractall(path=unzipped_dir)

                # copy files
                top_level_dir = find_top_level_dir(unzipped_dir, target_dir=LABELED_DATA_DIR)
                files_and_dirs = os.listdir(top_level_dir)
                req_dlc_dirs = ["labeled-data", "videos"]
                for d in req_dlc_dirs:
                    assert d in files_and_dirs, \
                        f"zipped DLC directory must include folder named {d}"
                    src = os.path.join(top_level_dir, d)
                    dst = os.path.join(self.proj_dir_abs, d)
                    if d == "labeled-data":
                        shutil.copytree(src, dst)
                    else:
                        copy_and_reformat_video_directory(src_dir=src, dst_dir=dst)

                # create single csv file of labels out of video-specific label files
                df_all = collect_dlc_labels(self.proj_dir_abs)
                csv_file = os.path.join(self.proj_dir_abs, COLLECTED_DATA_FILENAME)
                _logger.debug(f"Attempting to save collected data to {csv_file}")
                df_all.to_csv(csv_file)

                # flag finish coping all files
                finished_copy_files = True

            # remove zipped file from project folder
            if finished_copy_files:
                if os.path.exists(self.st_upload_existing_project_zippath):
                    os.remove(self.st_upload_existing_project_zippath)
                if os.path.isdir(unzipped_dir):
                    shutil.rmtree(unzipped_dir)

            elif self.st_existing_project_format == "SLEAP":
                if not os.path.exists(self.st_upload_existing_project_slp):
                    _logger.error(
                        f"Could not find SLEAP project file at \
                        {self.st_upload_existing_project_slp};"
                        f" aborting"
                    )
                    return

                file_path = self.st_upload_existing_project_slp
                os.makedirs(self.proj_dir_abs, exist_ok=True)

                # Extract and save CollectedData.csv
                csv_file = os.path.join(self.proj_dir_abs, COLLECTED_DATA_FILENAME)
                df = extract_labels_from_pkg_slp(file_path)
                df.to_csv(csv_file, index=False, header=False)

                # Extract frames from the slp file - labele data folder been created in the process
                extract_frames_from_pkg_slp(file_path, self.proj_dir_abs)

                # Create a videos folder for future use
                videos_dir = os.path.join(self.proj_dir_abs, 'videos')
                os.makedirs(videos_dir, exist_ok=True)

            else:
                raise NotImplementedError("Can only import 'Lightning Pose' or 'DLC' projects")

        except Exception as e:
            print(f"An error occurred: {e}")

        # create 'selected_frames.csv' file for each video subdirectory
        # this is required to import frames into label studio, so that we don't confuse context
        # frames with labeled frames
        csv_file = os.path.join(self.proj_dir_abs, COLLECTED_DATA_FILENAME)
        df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        frames = np.array(df.index)
        vids = np.unique([f.split('/')[1] for f in frames])
        for vid in vids:
            frames_to_label = np.array([f.split('/')[2] for f in frames if f.split('/')[1] in vid])
            save_dir = os.path.join(
                self.proj_dir_abs, LABELED_DATA_DIR, vid, SELECTED_FRAMES_FILENAME)
            _logger.debug(f"Saving selected frames to {save_dir}")
            np.savetxt(
                save_dir,
                np.sort(frames_to_label),
                delimiter=",",
                fmt="%s"
            )

        # update config file with frame shapes
        self._update_frame_shapes()

        # update counter
        self.count_upload_existing += 1

    def _delete_project(self, **kwargs):
        # delete project locally
        if os.path.exists(self.proj_dir_abs):
            shutil.rmtree(self.proj_dir_abs)

        # TODO: how to delete from label studio db?

        # update project info
        self.st_project_name = ""
        self.st_delete_project = False
        self.run(action="find_initialized_projects")

    def run(self, action, **kwargs):

        if action == "find_initialized_projects":
            self._find_initialized_projects()
        elif action == "update_paths":
            self._update_paths(**kwargs)
        elif action == "update_project_config":
            self._update_project_config(**kwargs)
        elif action == "update_frame_shapes":
            self._update_frame_shapes()
        elif action == "compute_labeled_frame_fraction":
            self._compute_labeled_frame_fraction(**kwargs)
        elif action == "load_project_defaults":
            self._load_project_defaults(**kwargs)
        elif action == "update_trained_models_list":
            self._update_trained_models_list(**kwargs)
        elif action == "upload_existing_project":
            self._upload_existing_project(**kwargs)
        elif action == "delete_project":
            self._delete_project(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    # ----------------------------------------------------
    # landing
    # ----------------------------------------------------
    with st.sidebar:
        st.title(""" Welcome to Lightning Pose App! """)
        st.write(
            "The first tab of the app is the project manager. Here you will be able to"
            " create new projects and load or delete existing projects under your account."
        )
        st.write("## To move forward, you will need to complete all the steps in this tab.")
        st.write("##")
        st.markdown("**Need further help? Check the:**")
        st.markdown(
            "App [documentation]"
            "(https://pose-app.readthedocs.io/en/latest/source/tabs/manage_project.html#)",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Github [repository](https://github.com/Lightning-Universe/Pose-app.html#)",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Lightning Pose [documentation]"
            "(https://lightning-pose.readthedocs.io/en/latest/.html#)",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Ensemble Kalman Smoother (EKS) [documentation]"
            "(https://pose-app.readthedocs.io/en/latest/source/tabs/train_infer.html#tab-train-infer-ensemble#)",  # noqa
            unsafe_allow_html=True,
        )

    st.header("Manage Lightning Pose projects")

    st_mode = st.radio(
        label="Check the box that applies:",
        options=[CREATE_STR, UPLOAD_STR, LOAD_STR, DELETE_STR],
        disabled=state.st_project_loaded,
        index=2 if (state.st_project_loaded and not state.st_create_new_project) else 0,
        help="Create a new project from scratch, upload an existing DLC project as a foundation "
             "for your new project, continue work on an ongoing lightning pose project, or remove "
             "a project from your projects."
    )
    st.text(f"Available projects: {state.initialized_projects}")

    if st_mode == LOAD_STR:
        st_project_name = st.selectbox(
            "Select existing project",
            sorted(state.initialized_projects),
        )
    elif st_mode == DELETE_STR:
        st_project_name = st.selectbox(
            "Select existing project",
            sorted(state.initialized_projects),
        )
    else:
        st_project_name = st.text_input(
            "Enter project name (must be at least 3 characters)",
            value="" if (not state.st_project_loaded or state.st_reset_project_name)
            else state.st_project_name
        )

    # ----------------------------------------------------
    # determine project status - load existing, create new
    # ----------------------------------------------------
    # we'll only allow config updates once the user has defined an allowable project name
    if st_project_name:
        # Check no other keys but letters, numbers
        st_project_name = st_project_name.replace(' ', '_')
        if st_mode == LOAD_STR:
            if st_project_name not in state.initialized_projects:
                # catch user error
                st.error(f"No project named {st_project_name} found; "
                         f"available projects are {state.initialized_projects}")
                enter_data = False
            elif state.st_project_loaded:
                # keep entering data after project has been loaded
                enter_data = True
            else:
                # load project for first time
                project_loaded = st.button(
                    "Load project",
                    disabled=True if not st_project_name != "" else False
                )
                enter_data = False
                if project_loaded:
                    # specify config file
                    config_file = os.path.join(
                        state.data_dir, st_project_name, f"model_config_{st_project_name}.yaml")
                    # update project manager
                    state.config_file = config_file
                    state.st_submits += 1
                    state.st_project_name = st_project_name
                    state.st_project_loaded = True
                    # let user know their data has been loaded
                    proceed_str = "Project loaded successfully! Proceed to following tabs."
                    proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
                    st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)
                    enter_data = True
                    if state.st_submits == 1:
                        # signal to lightning app that project has been loaded; this will
                        # - trigger loading of project config
                        # - populate other UIs with relevant project info
                        state.run_script = True
                        state.st_submits += 1  # prevent this block from running again
                        time.sleep(2)  # allow main event loop to catch up
                        st.experimental_rerun()  # update everything

        elif st_mode == CREATE_STR or st_mode == UPLOAD_STR:
            if state.st_project_loaded:
                # when coming back to tab from another
                enter_data = True
            elif st_project_name in state.initialized_projects:
                # catch user error
                st.error(f"A project named {st_project_name} already exists; "
                         f"choose a unique project name or select `Load existing project` above")
                enter_data = False
            elif st_project_name != "":
                # allow creation of new project
                enter_data = True
                state.st_create_new_project = True
            else:
                # catch remaining errors
                enter_data = False

            if st_mode == UPLOAD_STR:
                state.st_upload_existing_project = True
                enter_data = False

        elif st_mode == DELETE_STR:
            if st_project_name not in state.initialized_projects:
                # catch user error
                st.error(f"A project named {st_project_name} does not exist; "
                         f"choose an existing project name from the list above")
                st_project_name = ""  # stop downstream actions on this run
                state.st_reset_project_name = True  # stop downstream actions on future runs
            else:
                state.st_reset_project_name = False
                state.st_delete_project = True
            enter_data = False

        else:
            # catch remaining errors
            enter_data = False
    else:
        # cannot enter data until project name has been entered
        enter_data = False

    # ----------------------------------------------------
    # upload existing project
    # ----------------------------------------------------
    # initialize the file uploader

    if st_project_name and st_mode == UPLOAD_STR:

        st_prev_format = st.radio(
            "Select uploaded project format",
            options=["DLC", "Lightning Pose", "SLEAP"],  # TODO: SLEAP, MARS?
            help="Select the file format that the project is stored at."
            " If DLC selected make sure the zipped folder has meet all reqierments"
        )
        state.st_existing_project_format = st_prev_format

        if state.st_existing_project_format in ["DLC", "Lightning Pose"]:
            uploaded_file = st.file_uploader(
                "Upload project in .zip file", type="zip", accept_multiple_files=False)
            if uploaded_file is not None:
                # read it
                bytes_data = uploaded_file.read()
                # name it
                filename = uploaded_file.name
                filename_temp = filename.replace(".zip", '_temp.zip')
                filepath = os.path.join(os.getcwd(), "data", filename_temp)
                # write the content of the file to the path if it doesn't already exist
                if not os.path.exists(filepath):
                    with open(filepath, "wb") as f:
                        f.write(bytes_data)
                # check files
                state.st_error_flag, state.st_error_msg = check_files_in_zipfile(
                    filepath, project_type=st_prev_format)
                # grab keypoint names
                st_keypoints = get_keypoints_from_zipfile(filepath, project_type=st_prev_format)
                # update relevant vars

                state.st_upload_existing_project_zippath = filepath
                enter_data = True
                st_mode = CREATE_STR

        elif state.st_existing_project_format == "SLEAP":
            uploaded_file = st.file_uploader(
                "Upload project in .pkg.slp file",
                type="pkg.slp",
                accept_multiple_files=False,
                key="uploader2"
            )

            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                filename = uploaded_file.name
                filename_temp = filename.replace(".pkg.slp", '_temp.pkg.slp')
                filepath = os.path.join(os.getcwd(), "data", filename_temp)
                # write the content of the file to the path if it doesn't already exist
                if not os.path.exists(filepath):
                    with open(filepath, "wb") as f:
                        f.write(bytes_data)

                st_keypoints = get_keypoints_from_pkg_slp(filepath)

                state.st_upload_existing_project_slp = filepath
                enter_data = True
                st_mode = CREATE_STR

        st.caption(
            "If your zip file is larger than the 200MB limit, see the [FAQ]"
            "(https://pose-app.readthedocs.io/en/latest/source/faqs.html#faq-upload-limit)",
            unsafe_allow_html=True,
        )

    if state.st_error_flag:
        st.markdown(state.st_error_msg, unsafe_allow_html=True)
        enter_data = False

    # ----------------------------------------------------
    # user input for data config
    # ----------------------------------------------------

    # set defaults; these values are:
    # - used as field defaults below
    # - automatically updated in the main Flow from ProjectDataIO once the config file is specified
    st_n_views = state.st_n_views
    if not state.st_upload_existing_project:
        st_keypoints_ = state.st_keypoints_
        st_keypoints = []
        for kp in st_keypoints_:
            if kp not in st_keypoints:
                st_keypoints.append(kp)
        # if we are uploading existing project, we don't want to sort via np.unique, need to keep
        # keypoints in the correct order
    st_n_keypoints = state.st_n_keypoints
    st_pcasv_columns = np.array(state.st_pcasv_columns, dtype=np.int32)
    st_pcamv_columns = np.array(state.st_pcamv_columns, dtype=np.int32)

    if state.debug and enter_data:
        # hard-code params for debugging, skip manual entry
        st_n_views = 2
        st_keypoints = ["nose_top", "nose_bottom"]
        st_n_keypoints = len(st_keypoints)
        st_pcasv_columns = [0, 1]
        st_pcamv_columns = np.array([[0], [1]], dtype=np.int32)
        pcamv_ready = True

    elif not state.debug:

        # camera views
        if enter_data:
            st.markdown("")
            st.divider()
            st.markdown("")
            st.markdown(
                "##### Camera views",
                help="Support for multiple views is currently limited to either fusing the views "
                     "into single frames or utilizing a mirror to generate multiple views from a "
                     "single camera",
            )
            n_views = st.text_input(
                "Enter number of camera views:",
                disabled=not enter_data,
                value="" if not state.st_project_loaded else str(st_n_views),
            )
            st.caption(
                "For a multiview option check the [documentation]"
                "(https://lightning-pose.readthedocs.io/en/latest/source/"
                "user_guide_advanced/multiview_fused.html#)",
                unsafe_allow_html=True
            )
            if n_views:
                st_n_views = int(n_views)
            else:
                st_n_views = 0
            st.markdown("")

        # keypoints
        if st_n_views > 0:
            st.divider()
            st.markdown("##### Define keypoints")
            e1 = st.expander("Expand to see an example")
            keypoint_instructions = """
                **Instructions**:
                If your data has multiple views, make sure to create an entry for each bodypart
                in each view below like in the following example with 2 views (top and bottom):
                ```
                nose_top
                l_ear_top
                r_ear_top
                nose_bottom
                l_ear_bottom
                r_ear_bottom
                corner1_top
                ```
                It is also possible to track keypoints that are only present in a subset of the
                views, such as the keypoint `corner1_top` above.
            """
            e1.markdown(keypoint_instructions)
            if state.st_upload_existing_project:
                value = "\n".join(st_keypoints)
            elif not state.st_project_loaded:
                value = ""
            else:
                value = "\n".join(st_keypoints)
            keypoints = st.text_area(
                "Enter keypoint names (one per line, no spaces or dashes)\n\n"
                "The order here determines the labeling order",
                disabled=not enter_data,
                value=value,
            )
            # ensure no spaces or dashes
            st_keypoints = keypoints.replace(" ", "_").replace("-", "_").strip().split("\n")
            if len(st_keypoints) == 1 and st_keypoints[0] == "":
                st_keypoints = []
            st_n_keypoints = len(st_keypoints)
            st.markdown(f"You have defined {st_n_keypoints} keypoints across {st_n_views} views")
            st.markdown("")
            if state.st_project_loaded:
                st.warning(
                    "Currently, there is no option to update keypoint names in an existing "
                    "project. "
                    "Please start a new project with your full list of keypoints in advance."
                )

        # pca singleview
        if st_n_keypoints > 1:
            st.divider()
            st.markdown("##### Select subset of keypoints for Pose PCA")
            # st.markdown("""
            #     **Instructions**:
            #     The selected subset will be used for a Pose PCA loss on unlabeled videos.
            #     The subset should be keypoints that are not usually occluded (such as a tongue)
            #     and are not static (such as the corner of a box).
            # """)
            e2 = st.expander("Expand for further instructions")
            e2.markdown("""
                **When selecting keypoints for Pose PCA on unlabeled videos, focus on**:
                * **Selecting points with consistent visibility**, avoiding those prone to
                 occlusion (e.g., tongue) during movement.
                * **Selecting points that exhibit dynamic changes**,
                excluding static elements (e.g., corner of a box)
                offering minimal pose information.
            """)
            e2.write("*The selected subset will be used for a Pose PCA loss on unlabeled videos")
            pcasv_selected = [False for _ in st_keypoints]
            for k, kp in enumerate(st_keypoints):
                pcasv_selected[k] = st.checkbox(
                    kp,
                    disabled=not enter_data,
                    value=False if not state.st_project_loaded else (k in st_pcasv_columns),
                    key=f"pca_singleview_{kp}",
                )
            st_pcasv_columns = list(np.where(pcasv_selected)[0])
            st.markdown("")

        # pca multiview
        if st_n_keypoints > 1 and st_n_views > 1:

            st.markdown("##### Select subset of body parts for Multiview PCA")
            e3 = st.expander("Expand for further instructions")
            e3.markdown("""
                Select the same body part from different POV's.
                The selected subset will be used for a Multiview PCA loss on unlabeled videos.
                The subset should be keypoints that are usually visible in all camera views.
            """)
            n_bodyparts = st.text_input(
                "Enter number of body parts visible in all views:",
                value="" if not state.st_project_loaded else str(len(st_pcamv_columns[0])),
            )
            if n_bodyparts:
                st_n_bodyparts = int(n_bodyparts)
            else:
                st_n_bodyparts = 0

            if st_n_bodyparts > 0:

                st_pcamv_columns = np.zeros((st_n_views, st_n_bodyparts), dtype=np.int32)

                # set column titles
                cols_title = st.columns(st_n_views + 1)
                for c, col in enumerate(cols_title[1:]):
                    col.text(f"View {c}")
                # build table
                for r in range(st_n_bodyparts):
                    cols = st.columns(st_n_views + 1)
                    # set row titles
                    cols[0].text("")
                    cols[0].text("")
                    cols[0].text(f"Bodypart {r}")
                    # set bodypart dropdowns
                    for c, col in enumerate(cols[1:]):
                        kp = col.selectbox(
                            "", st_keypoints,
                            key=f"Bodypart {r} view {c}",
                            index=c * st_n_bodyparts + r
                        )
                        st_pcamv_columns[c, r] = np.where(np.array(st_keypoints) == kp)[0]

            st.markdown("")
        else:
            st_n_bodyparts = st_n_keypoints

        # construct config file
        if (st_n_keypoints > 1 and st_n_views > 1 and st_n_bodyparts > 0) \
                or (st_n_keypoints > 0 and st_n_views == 1):
            pcamv_ready = True
        else:
            pcamv_ready = False

    # ----------------------------------------------------
    # export data
    # ----------------------------------------------------

    if st_n_keypoints > 0 and st_n_views > 0 and pcamv_ready:

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("##### Export project configuration")

        need_update_pcamv = False
        if st_pcamv_columns is not None and len(st_pcamv_columns) > 0:
            if len(st_pcamv_columns.flatten()) != len(np.unique(st_pcamv_columns)):
                need_update_pcamv = True
                st.warning(
                    "Duplicate entries in PCA Multiview selections; each entry should be unique")

        # store dataset-specific values in order to update config.yaml file later
        st_new_vals = {"data": {}, "hydra": {"run": {}, "sweep": {}}}
        st_new_vals["data"]["data_dir"] = os.path.join(state.data_dir[1:], st_project_name)
        st_new_vals["data"]["video_dir"] = VIDEOS_DIR
        st_new_vals["data"]["csv_file"] = COLLECTED_DATA_FILENAME
        st_new_vals["data"]["num_keypoints"] = st_n_keypoints
        st_new_vals["data"]["keypoints"] = st_keypoints
        data_dir = st_new_vals["data"]["data_dir"]
        st_new_vals["hydra"]["run"]["dir"] = os.path.join(
            data_dir, MODELS_DIR, "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
        st_new_vals["hydra"]["sweep"]["dir"] = os.path.join(
            data_dir, MODELS_DIR, "multirun", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")

        if len(st_pcasv_columns) > 0:
            # need to convert all elements to int instead of np.int, streamlit can't cache ow
            st_new_vals["data"]["columns_for_singleview_pca"] = [int(t) for t in st_pcasv_columns]
        else:
            st_new_vals["data"]["columns_for_singleview_pca"] = []

        if st_pcamv_columns is not None and len(st_pcamv_columns) > 0:
            # need to convert all elements to int instead of np.int, streamlit can't cache ow
            st_new_vals["data"]["mirrored_column_matches"] = st_pcamv_columns.tolist()
        else:
            st_new_vals["data"]["mirrored_column_matches"] = []

        if state.st_project_loaded:
            st_submit_button = st.button(
                "Update project",
                disabled=need_update_pcamv or state.st_project_loaded,
            )
        else:
            st_submit_button = st.button(
                "Create project",
                disabled=need_update_pcamv or state.st_project_loaded,
            )
        if state.st_submits > 0:
            proceed_str = """
                Proceed to the next tab to extract frames for labeling.<br /><br />
                Use this LabelStudio login information:<br />
                <strong>username</strong>: user@localhost<br />
                <strong>password</strong>: pw
            """
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

        # Lightning way of returning the parameters
        if st_submit_button:

            state.st_submits += 1

            state.st_project_name = st_project_name
            state.st_project_loaded = True
            state.st_new_vals = st_new_vals

            state.st_n_views = st_n_views
            state.st_keypoints_ = st_new_vals["data"]["keypoints"]
            state.st_n_keypoints = st_n_keypoints
            state.st_pcasv_columns = st_new_vals["data"]["columns_for_singleview_pca"]
            state.st_pcamv_columns = st_new_vals["data"]["mirrored_column_matches"]

            st.text("Request submitted!")
            state.run_script = True  # must the last to prevent race condition
            st_autorefresh(interval=2000, key="refresh_project_ui")

    elif st_project_name and st_mode == DELETE_STR:

        st_submit_button = st.button("Delete project")

        if st_submit_button:

            state.st_submits_delete += 1
            state.st_project_name = st_project_name
            state.st_reset_project_name = True

            st.text("Request submitted!")
            state.run_script = True  # must the last to prevent race condition
            st_autorefresh(interval=1000, key="refresh_project_ui")
