"""Package constants"""

__version__ = "1.5.0"


# dir where lightning pose package lives, relative to Pose-app root
LIGHTNING_POSE_DIR = "lightning-pose"

# directory name constants; relative to Pose-app/data
LABELSTUDIO_DB_DIR = "labelstudio_db"

# directory name constants; relative to project_dir
LABELED_DATA_DIR = "labeled-data"
LABELED_DATA_CHECK_DIR = "labeled-data-check"
VIDEOS_DIR = "videos"
VIDEOS_TMP_DIR = "videos_tmp"
VIDEOS_INFER_DIR = "videos_infer"
ZIPPED_TMP_DIR = "frames_tmp"
MODELS_DIR = "models"
MODEL_VIDEO_PREDS_TRAIN_DIR = "video_preds"
MODEL_VIDEO_PREDS_INFER_DIR = "video_preds_infer"

# file name constants; relative to project_dir
COLLECTED_DATA_FILENAME = "CollectedData.csv"
LABELSTUDIO_METADATA_FILENAME = "label_studio_metadata.yaml"
LABELSTUDIO_TASKS_FILENAME = "label_studio_tasks.pkl"
LABELSTUDIO_CONFIG_FILENAME = "label_studio_config.xml"

# file name constants; relative to project_dir/<LABELED_DATA_DIR>/<video_name>
SELECTED_FRAMES_FILENAME = "selected_frames.csv"

# file name constatns; relative to project_dir/MODELS_DIR/<date>/<time>/
ENSEMBLE_MEMBER_FILENAME = "models_for_ensemble.txt"
