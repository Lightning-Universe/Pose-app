from lightning.app import CloudCompute, LightningFlow
from lightning.app.storage import FileSystem
from lightning.app.utilities.state import AppState
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml

from lightning_pose_app import MODELS_DIR
from lightning_pose_app.build_configs import LitPoseBuildConfig, lightning_pose_dir
from lightning_pose_app.utilities import StreamlitFrontend, WorkWithFileSystem


class FiftyoneWork(WorkWithFileSystem):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, name="fiftyone", **kwargs)

        self.fiftyone_launched = False
        self.fiftyone_datasets = []

    def start_fiftyone(self):
        """run fiftyone"""
        if not self.fiftyone_launched:
            import fiftyone as fo
            fo.launch_app(
                dataset=None,
                remote=True,
                address=self.host,
                port=self.port,
            )
            self.fiftyone_launched = True

    def find_fiftyone_datasets(self):
        """get existing fiftyone datasets"""
        # NOTE: we could migrate the fiftyone database back and forth between the Drive but this
        # seems lke overkill? the datasets are quick to make and users probably don't care so much
        # about these datasets; can return to this later
        import fiftyone as fo
        out = fo.list_datasets()
        datasets = []
        for x in out:
            if x.endswith("No datasets found"):
                continue
            if x.startswith("Migrating database"):
                continue
            if x.endswith("python"):
                continue
            if x in datasets:
                continue
            datasets.append(x)
        self.fiftyone_datasets = datasets

    def build_fiftyone_dataset(
            self, config_file: str, dataset_name: str, model_dirs: list, model_names: list,
        ):

        if dataset_name in self.fiftyone_datasets:
            return

        from lightning_pose.utils.fiftyone import FiftyOneFactory, check_dataset
        from omegaconf import DictConfig
        
        # pull models (relative path)
        self.get_from_drive(model_dirs)

        # pull config (relative path)
        self.get_from_drive([config_file])

        # load config (absolute path)
        cfg = DictConfig(yaml.safe_load(open(self.abspath(config_file), "r")))

        # edit config (add fiftyone key before making DictConfig, otherwise error)
        model_dirs_abs = [os.path.join(os.getcwd(), x[1:]) for x in model_dirs]
        cfg.data.data_dir = os.path.join(os.getcwd(), cfg.data.data_dir)
        cfg.eval.fiftyone.build_speed = "fast"
        cfg.eval.fiftyone.dataset_name = dataset_name
        cfg.eval.fiftyone.model_display_names = model_names
        cfg.eval.hydra_paths = model_dirs_abs

        # build dataset
        FiftyOneClass = FiftyOneFactory(dataset_to_create="images")()
        fo_plotting_instance = FiftyOneClass(cfg=cfg)
        dataset = fo_plotting_instance.create_dataset()
        # create metadata and print if there are problems
        check_dataset(dataset)
        # print the name of the dataset
        fo_plotting_instance.dataset_info_print()

        # add dataset name to list for user to see
        self.fiftyone_datasets.append(dataset_name)

    def run(self, action, **kwargs):

        # these functions require fiftyone and/or lighting-pose to be installed
        # each function imports the necessary functions directly
        # if imports are at the top of this module errors will arise in the orchestrator when 
        # importing from this module since the proper packages are not yet installed (cloud only)

        if action == "start_fiftyone":
            self.start_fiftyone(**kwargs)
        elif action == "find_fiftyone_datasets":
            self.find_fiftyone_datasets(**kwargs)
        elif action == "build_fiftyone_dataset":
            self.build_fiftyone_dataset(**kwargs)


class FiftyoneConfigUI(LightningFlow):
    """UI to run Fiftyone and Streamlit apps."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.work = FiftyoneWork(
            cloud_compute=CloudCompute("default"),
            cloud_build_config=LitPoseBuildConfig(),  # get fiftyone
        )

        # control runners
        # True = Run Jobs.  False = Do not Run jobs
        # UI sets to True to kickoff jobs
        # Job Runner sets to False when done
        self.run_script = False

        # params updated externally by top-level flow
        self.fiftyone_datasets = []
        self.trained_models = []
        self.proj_dir = None
        self.config_name = None

        # submit count
        self.submit_count = 0
        self.submit_success = False

        # output from the UI
        self.st_submit = False
        self.st_dataset_name = None
        self.st_model_dirs = [None for _ in range(2)]
        self.st_model_display_names = [None for _ in range(2)]

    def run(self, action, **kwargs):

        if action == "start_fiftyone":
            self.work.run(action=action, **kwargs)
            
        elif action == "find_fiftyone_datasets":
            self.work.run(action=action, **kwargs)
            self.fiftyone_datasets = self.work.fiftyone_datasets

        elif action == "build_fiftyone_dataset":
            self.work.run(
                action=action,
                config_file=os.path.join(self.proj_dir, self.config_name),  # relative paths
                dataset_name=self.st_dataset_name,
                model_dirs=self.st_model_dirs,  # relative paths
                model_names=self.st_model_display_names,
                **kwargs,
            )
            self.fiftyone_datasets = self.work.fiftyone_datasets

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_streamlit_fn)


def _render_streamlit_fn(state: AppState):
    """Create Fiftyone Dataset"""

    # force rerun to update page
    st_autorefresh(interval=2000, key="refresh_page")

    st.markdown(
        """
        ## Prepare Fiftyone diagnostics

        Choose two models for evaluation.

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

    # ---------------------------------------------------------
    # collect input from users
    # ---------------------------------------------------------
    with st.form(key="fiftyone_form", clear_on_submit=True):

        col0, col1 = st.columns(2)

        with col0:

            # select first model (supervised)
            options1 = sorted(state.trained_models, reverse=True)
            tmp = st.selectbox("Select Model 1", options=options1, disabled=state.run_script)
            st_model_dirs[0] = tmp
            tmp = st.text_input(
                "Display name for Model 1", value="model_1", disabled=state.run_script)
            st_model_display_names[0] = tmp

        with col1:

            # select second model (semi-supervised)
            options2 = sorted(state.trained_models, reverse=True)
            if st_model_dirs[0]:
                options2.remove(st_model_dirs[0])

            tmp = st.selectbox("Select Model 2", options=options2, disabled=state.run_script)
            st_model_dirs[1] = tmp
            tmp = st.text_input(
                "Display name for Model 2", value="model_2", disabled=state.run_script)
            st_model_display_names[1] = tmp

        # make model dirs paths relative to FileSystem
        for i in range(2):
            if st_model_dirs[i] and not os.path.isabs(st_model_dirs[i]):
                st_model_dirs[i] = os.path.join(state.proj_dir, MODELS_DIR, st_model_dirs[i])

        # dataset names
        existing_datasets = state.fiftyone_datasets
        st.write(f"Existing Fifityone datasets:\n{', '.join(existing_datasets)}")
        st_dataset_name = st.text_input(
            "Choose dataset name other than the above existing names", disabled=state.run_script)

        # build dataset
        st.markdown("""
            Diagnostics will be displayed in the following 'Fiftyone' tab.
            """)
        st_submit_button = st.form_submit_button("Prepare Fiftyone dataset", disabled=state.run_script)

    # ---------------------------------------------------------
    # check user input
    # ---------------------------------------------------------
    if st_model_display_names[0] is None \
            or st_model_display_names[1] is None \
            or st_model_display_names[0] == st_model_display_names[1]:
        st_submit_button = False
        state.submit_success = False
        st.warning(f"Must choose two unique model display names")
    if st_model_dirs[0] is None or st_model_dirs[1] is None:
        st_submit_button = False
        state.submit_success = False
        st.warning(f"Must choose two models to continue")
    if st_model_dirs[0] == st_model_dirs[1]:
        st_submit_button = False
        state.submit_success = False
        st.warning(f"Must choose two unique models to continue")
    if st_submit_button and \
            (st_dataset_name in existing_datasets
             or st_dataset_name is None
             or st_dataset_name == ""):
        st_submit_button = False
        state.submit_success = False
        st.warning(f"Enter a unique dataset name to continue")
    if state.run_script:
        st.warning(f"Waiting for existing dataset creation to finish "
                   f"(may take 30 seconds to update)")
    if state.submit_count > 0 \
            and not state.run_script \
            and not st_submit_button \
            and state.submit_success:
        proceed_str = "Diagnostics are ready to view in the following tab."
        proceed_fmt = "<p style='font-family:sans-serif; color:Green;'>%s</p>"
        st.markdown(proceed_fmt % proceed_str, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # build fiftyone dataset
    # ---------------------------------------------------------
    # this will only be run once when the user clicks the button; 
    # on the following pass the button click will be set to False again
    if st_submit_button:

        state.submit_count += 1

        # save streamlit options to flow object only on button click
        state.st_dataset_name = st_dataset_name
        state.st_model_dirs = st_model_dirs
        state.st_model_display_names = st_model_display_names

        # reset form
        st_dataset_name = None
        st_model_dirs = [None for _ in range(2)]
        st_model_display_names = [None for _ in range(2)]

        st.text("Request submitted!")
        state.submit_success = True
        state.run_script = True  # must the last to prevent race condition

        # force rerun to update warnings
        st_autorefresh(interval=2000, key="refresh_diagnostics_submitted")
