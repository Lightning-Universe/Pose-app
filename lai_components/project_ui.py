import copy
import glob
from lightning import LightningFlow, LightningWork
from lightning.app.storage.drive import Drive
from lightning.app.utilities.state import AppState
import math
import numpy as np
import os
import shutil
from PIL import Image
import streamlit as st
import yaml

from lai_components.vsc_streamlit import StreamlitFrontend


class ProjectDataIO(LightningWork):

    def __init__(self, *args, drive_name, config_dir, data_dir, default_config_dict, **kwargs):

        super().__init__(*args, **kwargs)

        self.drive = Drive(drive_name)

        # config for project named <PROJ_NAME> will be stored as
        # <config_dir>/config_<PROJ_NAME>.yaml
        self.config_dir = config_dir
        # if not os.path.exists(self.config_dir):
        #     os.makedirs(self.config_dir)
        self.config_name = None
        self.config_file = None

        # save default config info for initializing new projects
        self.default_config_dict = default_config_dict

        # data for project named <PROJ_NAME> will be stored as
        # <data_dir>/<PROJ_NAME>
        #   ├── labeled-data/
        #   ├── videos/
        #   └── CollectedData.csv
        self.data_dir = data_dir
        # if not os.path.exists(self.data_dir):
        #     os.makedirs(self.data_dir)

        self.proj_dir = None
        self.model_dir = None
        self.test_videos_dir = None
        self.n_labeled_frames = 0
        self.n_total_frames = 0
        self.time = 0.0

    def _get_from_drive_if_not_local(self, file):
        if not os.path.exists(file):
            try:
                print(f"{file} does not exist; getting from drive")
                self.drive.get(file)
            except Exception:
                print(f"could not find {file} in drive")

    def _put_to_drive_remove_local(self, file_or_dir):
        print(f"putting {file_or_dir} to drive")
        self.drive.put(file_or_dir)
        # # clean up the local object
        # if os.path.isfile(file_or_dir):
        #     os.remove(file_or_dir)
        # else:
        #     shutil.rmtree(file_or_dir)

    def update_paths(self, project_name, **kwargs):
        self.proj_dir = os.path.join(self.data_dir, project_name)
        self.config_dir = self.proj_dir
        self.config_name = f"model_config_{project_name}.yaml"
        self.config_file = os.path.join(self.config_dir, self.config_name)
        self.model_dir = os.path.join(self.proj_dir, "models")
        self.test_videos_dir = os.path.join(self.proj_dir, "videos")

    def update_project_config(self, new_vals_dict=None, **kwargs):
        """triggered by button click in UI"""

        # check to see if config exists locally; if not, try pulling from drive
        self._get_from_drive_if_not_local(self.config_file)

        # check to see if config exists; copy default config
        if (self.config_file is None) or (not os.path.exists(self.config_file)):
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
            # load existing config
            config_dict = yaml.safe_load(open(self.config_file))

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
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            yaml.dump(config_dict, open(self.config_file, "w"))

        # push data to drive and clean up local file
        self._put_to_drive_remove_local(self.config_file)

    def update_frame_shapes(self):

        # get labeled data from drive
        labeled_data_dir = os.path.join(self.proj_dir, "labeled-data")
        # check to see if config exists locally; if not, try pulling from drive
        self._get_from_drive_if_not_local(labeled_data_dir)

        # load single frame from labeled data
        imgs = glob.glob(os.path.join(self.proj_dir, "labeled-data", "*", "*.png"))
        if len(imgs) > 0:
            img = imgs[0]
            image = Image.open(img)
            self.update_project_config({
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

        # remove local files
        shutil.rmtree(labeled_data_dir)

    def compute_labeled_frame_fraction(self, timer=0.):
        # TODO: don't want to have metadata filename hard-coded here

        metadata_file = os.path.join(self.proj_dir, "label_studio_metadata.yaml")
        self._get_from_drive_if_not_local(metadata_file)

        try:
            proj_details = yaml.safe_load(open(metadata_file, "r"))
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
        return n_labeled_frames, n_total_frames

    def run(self, action, **kwargs):

        if action == "update_paths":
            self.update_paths(**kwargs)
        elif action == "update_project_config":
            self.update_project_config(**kwargs)
        elif action == "update_frame_shapes":
            self.update_frame_shapes()
        elif action == "compute_labeled_frame_fraction":
            n = 15
            new_time = kwargs["timer"]
            if (new_time - self.time) > n:
                self.n_labeled_frames, self.n_total_frames = self.compute_labeled_frame_fraction(
                    timer=new_time)


class ProjectUI(LightningFlow):
    """UI to set up project."""

    def __init__(self, *args, config_dir, data_dir, **kwargs):
        super().__init__(*args, **kwargs)

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script = False
        self.count = 0

        self.data_dir = data_dir
        self.config_dir = config_dir
        self.config_file = None
        self.initialized_projects = os.listdir(self.data_dir) if os.path.exists(self.data_dir) \
            else []
        self.create_new_project = False
        self.keypoints = None

        # output from the UI
        self.st_submits = 0
        self.st_project_name = None
        self.st_project_loaded = False
        self.st_new_vals = None

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


@st.cache
def get_project_defaults(config_file):

    if config_file is not None:
        # set values from config
        config_dict = yaml.safe_load(open(config_file))
        st_keypoints = config_dict["data"]["keypoints"]
        st_n_keypoints = config_dict["data"]["num_keypoints"]
        st_pcasv_columns = config_dict["data"]["columns_for_singleview_pca"]
        st_pcamv_columns = config_dict["data"]["mirrored_column_matches"]
        st_n_views = 1 if st_pcamv_columns is None else len(st_pcamv_columns)
    else:
        # reset values
        st_n_views = 0
        st_keypoints = []
        st_n_keypoints = 0
        st_pcasv_columns = []
        st_pcamv_columns = np.array([])

    return st_n_views, st_keypoints, st_n_keypoints, st_pcasv_columns, st_pcamv_columns


def _render_streamlit_fn(state: AppState):

    # ----------------------------------------------------
    # landing
    # ----------------------------------------------------

    st.markdown(
        """
        ## Manage Lightning Pose project
        """
    )

    st_mode = st.radio(
        "Create new project or load existing?",
        options=["Create new project", "Load existing project"]
    )

    st_project_name = st.text_input("Enter project name", value="")

    if st_project_name and st_mode == "Load existing project":
        if st_project_name not in state.initialized_projects:
            st.error(f"No project named {st_project_name} found; "
                     f"available projects are {state.initialized_projects}")
        else:
            project_loaded = st.button(
                "Load project",
                disabled=True if not st_project_name != "" else False
            )
            if project_loaded:
                # specify config file
                state.config_dir = os.path.join(state.data_dir, st_project_name)
                config_file = os.path.join(
                    state.config_dir, f"model_config_{st_project_name}.yaml")
                # update project manager
                state.config_file = config_file
                state.st_submits += 1
                state.st_project_name = st_project_name
                state.st_project_loaded = True

    st_n_views, st_keypoints, st_n_keypoints, st_pcasv_columns, st_pcamv_columns = \
        get_project_defaults(state.config_file)

    if st_project_name:
        if st_mode == "Load existing project" and state.st_project_loaded:
            enter_data = True
            # let user know they're data has been loaded
            proceed_str = "Project loaded successfully! You may proceed to subsequent tabs."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)
            if state.st_submits == 1:
                # signal to lightning app that project has been loaded; this will populate other
                # UIs with relevant project info
                state.keypoints = st_keypoints
                state.run_script = True
                state.st_submits += 1
        elif (st_mode == "Create new project") and (st_project_name in state.initialized_projects):
            st.error(f"A project named {st_project_name} already exists; "
                     f"choose a unique project name or select `Load existing project` above")
            enter_data = False
        elif st_mode == "Create new project" and st_project_name != "":
            enter_data = True
            state.create_new_project = True
        else:
            enter_data = False
    else:
        enter_data = False

    # add some whitespace
    st.markdown("")
    st.markdown("")

    # ----------------------------------------------------
    # user input for data config
    # ----------------------------------------------------

    DEBUG = True

    if DEBUG and enter_data:
        st_n_views = 2
        st_keypoints = ["nose_top", "nose_bottom"]
        st_n_keypoints = len(st_keypoints)
        st_pcasv_columns = [0, 1]
        st_n_bodyparts = 1
        st_pcamv_columns = np.array([[0], [1]], dtype=np.int)
        pcamv_ready = True

    else:
        # camera views
        if enter_data:
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
    
                The order in which you list the keypoints here determines the labeling order.
            """
            st.markdown(keypoint_instructions)
            keypoints = st.text_area(
                "Enter keypoint names (one per line):",
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
        if len(st_keypoints) > 1:
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

                )
            st_pcasv_columns = list(np.where(pcasv_selected)[0])
            st.markdown("")

        # pca multiview
        if len(st_keypoints) > 1 and st_n_views > 1:

            st.markdown("##### Select subset of body parts for Multiview PCA")
            st.markdown("""
                **Instructions**:
                The selected subset will be used for a Multiview PCA loss on unlabeled videos.
                The subset should be keypoints that are usually visible in all camera views.
            """)
            # pcasv_selected = [False for _ in st_keypoints]
            # for k, kp in enumerate(st_keypoints):
            #     pcasv_selected[k] = st.checkbox(kp, disabled=not enter_data)
            # st.markdown("")
            n_bodyparts = st.text_input(
                "Enter number of body parts visible in all views:",
                value="" if not state.st_project_loaded else str(len(st_pcamv_columns[0])),
            )
            if n_bodyparts:
                st_n_bodyparts = int(n_bodyparts)
            else:
                st_n_bodyparts = 0

            if st_n_bodyparts > 0:

                st_pcamv_columns = np.zeros((st_n_views, st_n_bodyparts), dtype=np.int)

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

                print(st_pcamv_columns)
            st.markdown("")
        else:
            st_n_bodyparts = 0

        # construct config file
        if (len(st_keypoints) > 1 and st_n_views > 1 and st_n_bodyparts > 0) \
                or (len(st_keypoints) > 1 and st_n_views == 1):
            pcamv_ready = True
        else:
            pcamv_ready = False

    if len(st_keypoints) > 1 and st_n_views > 0 and pcamv_ready:

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("##### Export project configuration")
        # give user slightly different info depending on where they are in their workflow
        if state.st_project_loaded:
            st.markdown("""Click on the button below to update project configuration.""")
        else:
            st.markdown("""
                Click on the button below to create a new project; you will then be able to start
                labeling data and train models!
            """)

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
        st_new_vals["data"]["video_dir"] = os.path.join(st_new_vals["data"]["data_dir"], "videos")
        st_new_vals["data"]["csv_file"] = "CollectedData.csv"
        st_new_vals["data"]["num_keypoints"] = st_n_keypoints
        st_new_vals["data"]["keypoints"] = st_keypoints
        st_new_vals["hydra"]["run"]["dir"] = os.path.join(
            st_new_vals["data"]["data_dir"], "models",
            "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
        st_new_vals["hydra"]["sweep"]["dir"] = os.path.join(
            st_new_vals["data"]["data_dir"], "models", "multirun",
            "${now:%Y-%m-%d}", "${now:%H-%M-%S}")

        if len(st_pcasv_columns) > 0:
            # need to convert all elements to int instead of np.int, streamlit can't cache ow
            st_new_vals["data"]["columns_for_singleview_pca"] = [int(t) for t in st_pcasv_columns]
        else:
            st_new_vals["data"]["columns_for_singleview_pca"] = None

        if st_pcamv_columns is not None and len(st_pcamv_columns) > 0:
            # need to convert all elements to int instead of np.int, streamlit can't cache ow
            st_new_vals["data"]["mirrored_column_matches"] = [
                [int(t_) for t_ in t] for t in st_pcamv_columns]
        else:
            st_new_vals["data"]["mirrored_column_matches"] = None

        if state.st_project_loaded:
            st_submit_button = st.button("Update project", disabled=need_update_pcamv)
        else:
            st_submit_button = st.button("Create project", disabled=need_update_pcamv)
        if state.st_submits > 0:
            proceed_str = "Please proceed to the next tab to extract frames for labeling."
            proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
            st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

        # Lightning way of returning the parameters
        if st_submit_button:

            state.st_submits += 1

            state.st_project_name = st_project_name
            state.st_project_loaded = True
            state.st_new_vals = st_new_vals
            state.keypoints = st_new_vals["data"]["keypoints"]

            state.run_script = True  # must the last to prevent race condition
