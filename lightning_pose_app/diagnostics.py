from lightning import CloudCompute, LightningFlow
from lightning.app.utilities.state import AppState
import os
import streamlit as st

from lightning_pose_app.bashwork import LitBashWork
from lightning_pose_app.build_configs import LitPoseNoGpuBuildConfig, lightning_pose_dir
from lightning_pose_app.utilities import args_to_dict, dict_to_args, StreamlitFrontend


class DiagnosticsUI(LightningFlow):
    """UI to run Fiftyone and Streamlit apps."""

    def __init__(
        self,
        *args,
        drive_name,
        fiftyone_kwargs={},
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # fiftyone worker
        self.fiftyone_kwargs = fiftyone_kwargs
        self.fiftyone = LitBashWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=LitPoseNoGpuBuildConfig(),  # get fiftyone
            drive_name=drive_name,
        )

        # streamlit labeled worker
        # self.my_streamlit_frame = LitBashWork(
        #     cloud_compute=CloudCompute("default"),
        #     cloud_build_config=StreamlitBuildConfig(),  # this may not be necessary
        #     drive_name=drive_name,
        # )

        # streamlit video worker
        # self.my_streamlit_video = LitBashWork(
        #     cloud_compute=CloudCompute("default"),
        #     cloud_build_config=StreamlitBuildConfig(),  # this may not be necessary
        #     drive_name=drive_name,
        # )

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script = False

        # params updated externally by top-level flow
        self.fiftyone_datasets = []
        self.hydra_outputs = {}
        self.proj_dir = None
        self.config_name = None

        # submit count
        self.submit_count = 0

        # output from the UI
        self.st_script_args = self.fiftyone_kwargs["script_args"]
        self.st_script_args_append = None
        self.st_model_display_names = None
        self.st_submit = False
        self.st_dataset_name = None
        self.st_model_dirs = None

    def set_hydra_outputs(self, names: dict):
        self.hydra_outputs.update(names)

    def add_hydra_output(self, name: str):
        self.hydra_outputs.update(name)

    def start_fiftyone(self):
        """run fiftyone"""
        cmd = "fiftyone app launch --address $host --port $port"
        self.fiftyone.run(cmd, wait_for_exit=False, cwd=lightning_pose_dir)

    def find_fiftyone_datasets(self):
        # get existing fiftyone datasets
        cmd = "fiftyone datasets list"
        # TODO: need to pull in the fiftyone db
        self.fiftyone.run(cmd)
        if self.fiftyone.last_args() == cmd:
            names = []
            for x in self.fiftyone.stdout:
                if x.endswith("No datasets found"):
                    continue
                if x.startswith("Migrating database"):
                    continue
                if x.endswith("python"):
                    continue
                names.append(x)
            self.fiftyone_datasets = names
        else:
            pass

    def build_fiftyone_dataset(self):
        cmd = "python" \
              + " " + self.fiftyone_kwargs["script_name"] \
              + " " + self.st_script_args_append \
              + " " + self.st_script_args \
              + " " + "eval.fiftyone.dataset_to_create=images"
        self.fiftyone.run(cmd, cwd=self.fiftyone_kwargs["script_dir"], timer=self.st_dataset_name)

        # add dataset name to list for user to see
        self.fiftyone_datasets.append(self.st_dataset_name)

    def start_st_frame(self):
        """run streamlit for labeled frames"""

        # TODO: UPDATE FROM ANKIT

        # set labeled csv
        csv_file = os.path.join(
            lightning_pose_dir, "toy_datasets/toymouseRunningData/CollectedData_.csv")
        labeled_csv_args = f"--labels_csv={csv_file}"

        # set prediction files (hard code some paths for now)
        prediction_file_args = ""
        for model_dir in self.diagnostics_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.diagnostics_ui.outputs_dir, model_dir, "predictions.csv")
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.diagnostics_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.diagnostics_ui.outputs_dir, self.diagnostics_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/labeled_frame_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + labeled_csv_args \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_frame.run(cmd, wait_for_exit=False, cwd=".")

    def start_st_video(self):
        """run streamlit for videos"""

        # TODO: UPDATE FROM ANKIT

        # set prediction files (hard code some paths for now)
        # select random video
        prediction_file_args = ""
        video_file = "test_vid.csv"  # TODO: find random vid in predictions directory
        for model_dir in self.diagnostics_ui.st_model_dirs:
            abs_file = os.path.join(
                lightning_pose_dir, self.diagnostics_ui.outputs_dir, model_dir, "video_preds", video_file)
            prediction_file_args += f" --prediction_files={abs_file}"

        # set model names
        model_name_args = ""
        for name in self.diagnostics_ui.st_model_display_names:
            model_name_args += f" --model_names={name}"

        # set data config (take config from first selected model)
        cfg_file = os.path.join(
            lightning_pose_dir, self.diagnostics_ui.outputs_dir, self.diagnostics_ui.st_model_dirs[0], ".hydra",
            "config.yaml")
        # replace relative paths of example dataset
        cfg = yaml.safe_load(open(cfg_file))
        if not os.path.isabs(cfg["data"]["data_dir"]):
            # data_dir = cfg["data"]["data_dir"]
            cfg["data"]["data_dir"] = os.path.abspath(os.path.join(
                lightning_pose_dir, cfg["data"]["data_dir"]))
        # resave file
        yaml.safe_dump(cfg, open(cfg_file, "w"))

        data_cfg_args = f" --data_cfg={cfg_file}"

        cmd = "streamlit run ./tracking-diagnostics/apps/video_diagnostics.py" \
              + " --server.address {host} --server.port {port}" \
              + " -- " \
              + " " + prediction_file_args \
              + " " + model_name_args \
              + " " + data_cfg_args
        self.my_streamlit_video.run(cmd, wait_for_exit=False, cwd=".")

    def run(self, action, **kwargs):

        if action == "find_fiftyone_datsets":
            self.find_fiftyone_datasets()
        elif action == "start_fiftyone":
            self.start_fiftyone()
        elif action == "build_fiftyone_dataset":
            self.build_fiftyone_dataset()
        elif action == "start_st_frame":
            self.start_st_frame()
        elif action == "start_st_video":
            self.start_st_video()

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def set_script_args(model_dirs: [str], script_args: str):

    script_args_dict = args_to_dict(script_args)

    # enrich the args
    # eval.video_file_to_plot="</ABSOLUTE/PATH/TO/VIDEO.mp4>" \

    # eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>","</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
    # eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"]
    # eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]

    if model_dirs:
        path_list = ','.join([f"'{x}'" for x in model_dirs])
        script_args_dict["eval.hydra_paths"] = f"[{path_list}]"

    # these will be controlled by the runners. remove if set manually
    script_args_dict.pop('eval.fiftyone.address', None)
    script_args_dict.pop('eval.fiftyone.port', None)
    script_args_dict.pop('eval.fiftyone.launch_app_from_script', None)
    script_args_dict.pop('eval.fiftyone.dataset_to_create', None)
    script_args_dict.pop('eval.fiftyone.dataset_name', None)
    script_args_dict.pop('eval.fiftyone.model_display_names', None)

    return dict_to_args(script_args_dict), script_args_dict


def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset"""

    st.markdown(
        """
        ## Prepare diagnostics

        Choose two models for evaluation.

        If you just trained models these will be provided as defaults, but can be updated by using 
        the drop-down menus.

        Click 'Prepare' to begin preparation of the diagnostics. These diagnostics will be 
        displayed in the following three tabs:
        * Labeled Preds: view model predictions and ground truth on all images using FiftyOne
        * Labeled Diagnostics: view metrics on labeled frames in streamlit
        * Video Diagnostics: view metrics on unlabeled videos in streamlit

        """
    )

    st.markdown(
        """
        #### Select models
        """
    )

    # hard-code two models for now
    st_model_dirs = [None for _ in range(2)]
    st_model_display_names = [None for _ in range(2)]

    # select first model (supervised)
    tmp = st.selectbox(
        "Select Model 1",
        [k for k, v in sorted(state.hydra_outputs.items(), reverse=True)]
    )
    st_model_dirs[0] = tmp
    tmp = st.text_input("Display name for Model 1")
    st_model_display_names[0] = tmp

    # select second model (semi-supervised)
    options = [k for k, v in sorted(state.hydra_outputs.items(), reverse=True)]
    if st_model_dirs[0]:
        options.remove(st_model_dirs[0])

    tmp = st.selectbox("Select Model 2", options)
    st_model_dirs[1] = tmp
    tmp = st.text_input("Display name for Model 2")
    st_model_display_names[1] = tmp

    # make model dirs absolute paths
    for i in range(2):
        if not os.path.isabs(st_model_dirs[i]):
            st_model_dirs[i] = os.path.join(os.getcwd(), state.proj_dir, "models", st_model_dirs[i])

    # dataset names
    existing_datasets = state.fiftyone_datasets
    st.write(f"Existing Fifityone datasets:\n{', '.join(existing_datasets)}")
    st_dataset_name = st.text_input("Choose dataset name other than the above existing names")
    if st_dataset_name in existing_datasets:
        st.error(f"{st_dataset_name} exists. Please choose a new name.")
        st_dataset_name = None

    # parse
    st_script_args, script_args_dict = set_script_args(
        model_dirs=st_model_dirs, script_args=state.st_script_args)

    # build dataset
    st_submit_button = st.button(
        "Initialize diagnostics",
        disabled=True if ((st_dataset_name is None) or (st_dataset_name == "") or state.run_script)
        else False)
    if state.run_script:
        st.warning(f"waiting for existing dataset creation to finish")
    if st_model_dirs[0] is None or st_model_dirs[1] is None:
        st.warning(f"select at least one model to continue")
    if (st_dataset_name is None) or (st_dataset_name == ""):
        st.warning(f"enter a unique dataset name to continue")

    # Lightning way of returning the parameters
    if st_submit_button:

        state.submit_count += 1

        # save streamlit options to flow object
        state.st_dataset_name = st_dataset_name
        state.st_model_display_names = st_model_display_names
        state.st_model_dirs = st_model_dirs
        state.st_script_args = st_script_args

        # set key-value pairs that will be used as script args
        model_names = ','.join([f"'{x}'" for x in st_model_display_names])
        script_args_append = f" --config-path={os.path.join(os.getcwd(), state.proj_dir)}"
        script_args_append += f" --config-name={state.config_name}"
        script_args_append += f" eval.fiftyone.dataset_name={st_dataset_name}"
        script_args_append += f" eval.fiftyone.model_display_names=[{model_names}]"
        script_args_append += f" eval.fiftyone.launch_app_from_script=false"
        state.st_script_args_append = script_args_append

        state.run_script = True  # must the last to prevent race condition
