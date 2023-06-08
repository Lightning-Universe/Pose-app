import copy
import glob
from lightning import LightningFlow, LightningWork
from lightning.app.storage import FileSystem
from lightning.app.utilities.state import AppState
import math
import numpy as np
import os
import shutil
from streamlit_autorefresh import st_autorefresh
import time
import streamlit as st
import yaml

from lightning_pose_app.utilities import StreamlitFrontend


class ProjectUI(LightningFlow):
    """UI to set up project."""

    def __init__(self, *args, data_dir, default_config_dict, debug=False, **kwargs):

        super().__init__(*args, **kwargs)

        self._drive = FileSystem()

        # initialize data_dir if it doesn't yet exist
        if not self._drive.isdir(data_dir):
            d = self.abspath(data_dir)
            os.makedirs(d, exist_ok=True)
            f = os.path.join(d, "tmp.txt")
            with open(f, "w") as fs:
                fs.write("tmp")
            self._drive.put(f, os.path.join(data_dir, "tmp.txt"))

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
        self.model_dir = None
        self.trained_models = []
        self.n_labeled_frames = 0
        self.n_total_frames = 0

        # UI info
        self.run_script = False
        self.update_models = False
        self.count = 0  # counter for external app
        self.st_submits = 0  # counter for this streamlit app
        self.initialized_projects = []
        self.st_project_name = None
        self.st_create_new_project = False
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
        return np.unique(self.st_keypoints_).tolist()

    def _get_from_drive_if_not_local(self, file_or_dir):

        if not os.path.exists(self.abspath(file_or_dir)):
            try:
                print(f"PROJECT drive try get {file_or_dir}")
                src = file_or_dir  # shared
                dst = self.abspath(file_or_dir)  # local
                self._drive.get(src, dst, overwrite=True)
                print(f"PROJECT drive success get {file_or_dir}")
            except Exception as e:
                print(e)
                print(f"could not find {file_or_dir} in {self.data_dir}")
        else:
            print(f"loading local version of {file_or_dir}")

    def _put_to_drive_remove_local(self, file_or_dir, remove_local=True):
        print(f"PROJECT put to drive {file_or_dir}")
        src = self.abspath(file_or_dir)  # local
        if os.path.isfile(src):
            dst = file_or_dir  # shared
            self._drive.put(src, dst)
        elif os.path.isdir(src):
            files_local = os.listdir(src)
            existing_files = self._drive.listdir(file_or_dir)
            for file_or_dir_local in files_local:
                rel_path = os.path.join(file_or_dir, file_or_dir_local)
                if rel_path not in existing_files:
                    src = self.abspath(rel_path)
                    dst = rel_path
                    self._drive.put(src, dst)
                else:
                    print(f"{rel_path} already exists on FileSystem! not updating")
        # clean up the local object
        if remove_local:
            if os.path.isfile(self.abspath(file_or_dir)):
                os.remove(self.abspath(file_or_dir))
            else:
                shutil.rmtree(self.abspath(file_or_dir))

    @staticmethod
    def abspath(path):
        if path[0] == "/":
            path_ = path[1:]
        else:
            path_ = path
        return os.path.abspath(path_)

    def _find_initialized_projects(self):
        # find all directories inside the data_dir; these should be the projects
        # (except labelstudio database)
        projects = self._drive.listdir(self.data_dir)
        # strip leading dirs to just get project names
        projects = [
            os.path.basename(p) for p in projects
            if not (p.endswith("labelstudio_db") or p.endswith(".txt"))
        ]
        self.initialized_projects = list(np.unique(projects))

    def _update_paths(self, project_name=None, **kwargs):
        if not project_name:
            project_name = self.st_project_name
        # these will all be paths RELATIVE to the FileSystem root
        if project_name:
            self.proj_dir = os.path.join(self.data_dir, project_name)
            self.config_name = f"model_config_{project_name}.yaml"
            self.config_file = os.path.join(self.proj_dir, self.config_name)
            self.model_dir = os.path.join(self.proj_dir, "models")  # hardcoded in train_infer.py

    def _update_project_config(self, new_vals_dict=None, **kwargs):
        """triggered by button click in UI"""

        if not new_vals_dict:
            new_vals_dict = self.st_new_vals

        # check to see if config exists locally; if not, try pulling from drive
        if self.config_file:
            self._get_from_drive_if_not_local(self.config_file)

        # check to see if config exists; copy default config if not
        if (self.config_file is None) or (not os.path.exists(self.abspath(self.config_file))):
            print(f"no config file at {self.config_file}")
            print("loading default config")
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
            print("loading existing config")
            # load existing config
            config_dict = yaml.safe_load(open(self.abspath(self.config_file)))

        # update config using new_vals_dict; assume this is a dict of dicts
        # new_vals_dict = {
        #     "data": new_data_dict,
        #     "eval": new_eval_dict,
        #     ...}
        if new_vals_dict is not None:
            for sconfig_name, sconfig_dict in new_vals_dict.items():
                for key, val in sconfig_dict.items():
                    if isinstance(val, dict):
                        # update config file up to depth 2
                        for key1, val1 in val.items():
                            config_dict[sconfig_name][key][key1] = val1
                    else:
                        config_dict[sconfig_name][key] = val
            # save out updated config file locally
            if not os.path.exists(self.abspath(self.proj_dir)):
                os.makedirs(self.abspath(self.proj_dir))
            yaml.dump(config_dict, open(self.abspath(self.config_file), "w"))

        # push data to drive and clean up local file
        self._put_to_drive_remove_local(self.config_file)

    def _update_frame_shapes(self):

        from PIL import Image

        # get labeled data from drive
        labeled_data_dir = os.path.join(self.proj_dir, "labeled-data")
        # check to see if config exists locally; if not, try pulling from drive
        self._get_from_drive_if_not_local(labeled_data_dir)

        # load single frame from labeled data
        imgs = glob.glob(os.path.join(self.abspath(self.proj_dir), "labeled-data", "*", "*.png"))
        if len(imgs) > 0:
            img = imgs[0]
            image = Image.open(img)
            self._update_project_config(new_vals_dict={
                "data": {
                    "image_orig_dims": {
                        "height": image.height,
                        "width": image.width
                    },
                    "image_resize_dims": {
                        "height": 2 ** (math.floor(math.log(image.height, 2))),
                        "width": 2 ** (math.floor(math.log(image.width, 2))),
                    }
                }
            })
        else:
            print(glob.glob(os.path.join(self.abspath(self.proj_dir), "labeled-data", "*")))
            print("did not find labeled data directory in FileSystem")

    def _compute_labeled_frame_fraction(self, timer=0.0):
        # TODO: don't want to have metadata filename hard-coded here

        metadata_file = os.path.join(self.proj_dir, "label_studio_metadata.yaml")
        self._get_from_drive_if_not_local(metadata_file)

        try:
            proj_details = yaml.safe_load(open(self.abspath(metadata_file), "r"))
            n_labeled_frames = proj_details["n_labeled_tasks"]
            n_total_frames = proj_details["n_total_tasks"]
        except FileNotFoundError:
            print(f"could not find {metadata_file}")
            n_labeled_frames = None
            n_total_frames = None
        except Exception as e:
            print(e)
            n_labeled_frames = None
            n_total_frames = None

        # remove local file so that Work is forced to load updated versions from Drive
        if os.path.exists(self.abspath(metadata_file)):
            os.remove(self.abspath(metadata_file))

        self.n_labeled_frames = n_labeled_frames
        self.n_total_frames = n_total_frames

    def _load_project_defaults(self, **kwargs):

        # check to see if config exists locally; if not, try pulling from drive
        if self.config_file:
            self._get_from_drive_if_not_local(self.config_file)

        # check to see if config exists
        if self.config_file and os.path.exists(self.abspath(self.config_file)):
            # set values from config
            config_dict = yaml.safe_load(open(self.abspath(self.config_file)))
            self.st_keypoints_ = config_dict["data"]["keypoints"]
            self.st_n_keypoints = config_dict["data"]["num_keypoints"]
            self.st_pcasv_columns = config_dict["data"]["columns_for_singleview_pca"]
            self.st_pcamv_columns = config_dict["data"]["mirrored_column_matches"]
            self.st_n_views = 1 if len(self.st_pcamv_columns) == 0 else len(self.st_pcamv_columns)

    def _update_trained_models_list(self, **kwargs):

        if self._drive.isdir(self.model_dir):
            trained_models = []
            # this returns a list of model training days
            dirs_day = self._drive.listdir(self.model_dir)
            # loop over days and find HH-MM-SS
            for dir_day in dirs_day:
                dirs_time = self._drive.listdir("/" + dir_day)
                for dir_time in dirs_time:
                    trained_models.append('/'.join(dir_time.split('/')[-2:]))
            self.trained_models = trained_models

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
        elif action == "put_file_to_drive":
            self._put_to_drive_remove_local(**kwargs)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):

    # ----------------------------------------------------
    # landing
    # ----------------------------------------------------

    st.markdown(""" ## Manage Lightning Pose project """)

    st_mode = st.radio(
        "",
        options=["Create new project", "Load existing project"],
        disabled=state.st_project_loaded,
        index=1 if (state.st_project_loaded and not state.st_create_new_project) else 0,
    )

    st.text(f"Available projects: {state.initialized_projects}")

    st_project_name = st.text_input(
        "Enter project name",
        value="" if not state.st_project_loaded else state.st_project_name)

    # ----------------------------------------------------
    # determine project status - load existing, create new
    # ----------------------------------------------------
    # we'll only allow config updates once the user has defined an allowable project name
    if st_project_name:
        if st_mode == "Load existing project":
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
                    proceed_str = "Project loaded successfully! You may proceed to following tabs."
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
        elif st_mode == "Create new project":
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
        else:
            # catch remaining errors
            enter_data = False
    else:
        # cannot enter data until project name has been entered
        enter_data = False

    # ----------------------------------------------------
    # user input for data config
    # ----------------------------------------------------

    # set defaults; these values are:
    # - used as field defaults below
    # - automatically updated in the main Flow from ProjectDataIO once the config file is specified
    st_n_views = state.st_n_views
    st_keypoints = np.unique(state.st_keypoints_).tolist()  # duplication bug fix, not solved
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
            st.markdown("")
            st.markdown("##### Camera views")
            n_views = st.text_input(
                "Enter number of camera views:",
                disabled=not enter_data,
                value="" if not state.st_project_loaded else str(st_n_views),
            )
            if n_views:
                st_n_views = int(n_views)
            else:
                st_n_views = 0
            st.markdown("")

        # keypoints
        if st_n_views > 0:
            st.markdown("##### Define keypoints")
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
                It is also possible to track keypoints that are only present in a subset of the views,
                such as the keypoint `corner1_top` above.
            """
            st.markdown(keypoint_instructions)
            keypoints = st.text_area(
                "Enter keypoint names (one per line, determines labeling order):",
                disabled=not enter_data,
                value="" if not state.st_project_loaded else "\n".join(st_keypoints),
            )
            st_keypoints = keypoints.strip().split("\n")
            if len(st_keypoints) == 1 and st_keypoints[0] == "":
                st_keypoints = []
            st_n_keypoints = len(st_keypoints)
            st.markdown(f"You have defined {st_n_keypoints} keypoints across {st_n_views} views")
            st.markdown("")

        # pca singleview
        if st_n_keypoints > 1:
            st.markdown("##### Select subset of keypoints for Pose PCA")
            st.markdown("""
                **Instructions**:
                The selected subset will be used for a Pose PCA loss on unlabeled videos.
                The subset should be keypoints that are not usually occluded (such as a tongue)
                and are not static (such as the corner of a box).
            """)
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
            st.markdown("""
                **Instructions**:
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
                            f"", st_keypoints, key=f"Bodypart {r} view {c}",
                            index=c * st_n_bodyparts + r
                        )
                        st_pcamv_columns[c, r] = np.where(np.array(st_keypoints) == kp)[0]

            st.markdown("")
        else:
            st_n_bodyparts = st_n_keypoints

        # construct config file
        if (st_n_keypoints > 1 and st_n_views > 1 and st_n_bodyparts > 0) \
                or (st_n_keypoints > 1 and st_n_views == 1):
            pcamv_ready = True
        else:
            pcamv_ready = False

    # ----------------------------------------------------
    # export data
    # ----------------------------------------------------

    if st_n_keypoints > 1 and st_n_views > 0 and pcamv_ready:

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
        # TODO: some of this is updated in ProjectDataIO.update_config also, should unify
        st_new_vals = {"data": {}, "hydra": {"run": {}, "sweep": {}}}
        st_new_vals["data"]["data_dir"] = os.path.join(state.data_dir, st_project_name)
        st_new_vals["data"]["video_dir"] = "videos"
        st_new_vals["data"]["csv_file"] = "CollectedData.csv"
        st_new_vals["data"]["num_keypoints"] = st_n_keypoints
        st_new_vals["data"]["keypoints"] = st_keypoints
        data_dir = st_new_vals["data"]["data_dir"]
        st_new_vals["hydra"]["run"]["dir"] = os.path.join(
            data_dir, "models", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
        st_new_vals["hydra"]["sweep"]["dir"] = os.path.join(
            data_dir, "models", "multirun", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")

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
            st_submit_button = st.button("Update project", disabled=need_update_pcamv)
        else:
            st_submit_button = st.button("Create project", disabled=need_update_pcamv)
        if state.st_submits > 0:
            proceed_str = """
                Please proceed to the next tab to extract frames for labeling.<br />
                Use the following login information:<br />
                username: user@localhost<br />
                password: pw
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
