from datetime import datetime
import os
import yaml

from omegaconf import DictConfig

from lightning_pose_app import (
    LIGHTNING_POSE_DIR,
    MODELS_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODEL_VIDEO_PREDS_TRAIN_DIR,
    VIDEOS_DIR,
)
from lightning_pose_app.utilities import update_config


def test_train(root_dir, tmp_proj_dir, video_file):

    from lightning_pose_app.backend.train_infer import train

    # load example config
    config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
    cfg = DictConfig(yaml.safe_load(open(
        os.path.join(config_dir, "config_mirror-mouse-example.yaml"),
        "r",
    )))

    # ----------------------------
    # train, do not output videos
    # ----------------------------
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
            "max_epochs": 2,
            "check_val_every_n_epoch": 2,
        },
    }
    # update config with user-provided overrides
    cfg = update_config(cfg, config_overrides)
    # create directory to save results
    results_dir_0 = os.path.join(
        base_dir,
        MODELS_DIR,
        datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST"),
    )
    train(
        cfg=cfg,
        results_dir=results_dir_0,
        work=None,
    )
    results_artifacts_0 = os.listdir(results_dir_0)
    assert os.path.exists(results_dir_0)
    assert "predictions.csv" in results_artifacts_0
    assert "lightning_logs" not in results_artifacts_0
    assert MODEL_VIDEO_PREDS_TRAIN_DIR not in results_artifacts_0

    # ----------------------------
    # train, output videos
    # ----------------------------
    cfg.eval.predict_vids_after_training = True
    cfg.eval.save_vids_after_training = True
    results_dir_1 = os.path.join(
        base_dir,
        MODELS_DIR,
        datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST"),
    )
    train(
        cfg=cfg,
        results_dir=results_dir_1,
        work=None,
    )
    results_artifacts_1 = os.listdir(results_dir_1)
    assert os.path.exists(results_dir_1)
    assert "predictions.csv" in results_artifacts_1
    assert "lightning_logs" not in results_artifacts_1
    assert MODEL_VIDEO_PREDS_TRAIN_DIR in results_artifacts_1
    labeled_vid_dir = os.path.join(results_dir_1, MODEL_VIDEO_PREDS_TRAIN_DIR, "labeled_videos")
    assert os.path.exists(labeled_vid_dir)
    assert len(os.listdir(labeled_vid_dir)) > 0


def test_inference_with_metrics():
    pass


def test_make_labeled_video(video_file, video_file_pred_df, tmpdir):

    from lightning_pose_app.backend.train_infer import make_labeled_video

    save_file = os.path.join(tmpdir, 'test_output.mp4')
    make_labeled_video(
        video_file=video_file,
        preds_df=video_file_pred_df,
        save_file=save_file,
        video_start_time=0.0,
        confidence_thresh=0.0,
    )
    assert os.path.isfile(save_file)
