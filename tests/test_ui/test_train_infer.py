from lightning.app import CloudCompute
import numpy as np
import os
import pandas as pd
import shutil
import yaml

from lightning_pose_app import (
    LIGHTNING_POSE_DIR,
    MODELS_DIR, 
    VIDEOS_DIR,
)
from lightning_pose_app.ui.train_infer import (
    VIDEO_LABEL_NONE, 
    VIDEO_LABEL_INFER, 
    VIDEO_LABEL_INFER_LABEL,
) 


def test_train_infer_work(root_dir, tmp_proj_dir, video_file):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app.ui.project import ProjectUI
    from lightning_pose_app.ui.train_infer import LitPose

    work = LitPose()

    # ----------------
    # helper flow
    # ----------------
    # load default config and pass to project manager
    config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
    default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))
    flow = ProjectUI(
        data_dir="/data",
        default_config_dict=default_config_dict,
    )
    proj_name = os.path.split(tmp_proj_dir)[-1]
    flow.run(action="update_paths", project_name=proj_name)
    flow.run(action="update_frame_shapes")

    # ----------------
    # train
    # ----------------
    base_dir = os.path.join(root_dir, tmp_proj_dir)
    config_overrides = {
        "data": {
            "data_dir": base_dir,
            "video_dir": os.path.join(base_dir, VIDEOS_DIR),
            "num_keypoints": 17,
        },
        "eval": {
            "test_videos_directory": os.path.join(base_dir, VIDEOS_DIR),
            "predict_vids_after_training": False,
            "save_vids_after_training": False,
        },
        "model": {
            "model_type": "heatmap",
            "losses_to_use": [],
        },
        "training": {
            "imgaug": "dlc",
            "max_epochs": 5,
        },
    }
    results_dir = os.path.join(base_dir, MODELS_DIR, "date0/time0")
    work._train(
        config_file=os.path.join(tmp_proj_dir, flow.config_name),
        config_overrides=config_overrides,
        results_dir=results_dir,
    )
    results_artifacts = os.listdir(results_dir)
    assert work.work_is_done_training
    assert os.path.exists(results_dir)
    assert "predictions.csv" in results_artifacts
    assert "lightning_logs" not in results_artifacts
    assert "video_preds" not in results_artifacts

    # output videos
    # config_overrides[]
    
    # ----------------
    # infer
    # ----------------
    # TODO

    # ----------------
    # clean up
    # ----------------
    del flow
    del work
