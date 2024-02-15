from datetime import datetime
import os
import yaml

from lightning_pose_app import (
    LIGHTNING_POSE_DIR,
    MODELS_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODEL_VIDEO_PREDS_TRAIN_DIR,
    VIDEOS_DIR,
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
            "max_epochs": 2,
            "check_val_every_n_epoch": 2,
        },
    }
    model_name_0 = datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST")
    results_dir_0 = os.path.join(base_dir, MODELS_DIR, model_name_0)
    work._train(
        config_file=os.path.join(tmp_proj_dir, flow.config_name),
        config_overrides=config_overrides,
        results_dir=results_dir_0,
    )
    results_artifacts_0 = os.listdir(results_dir_0)
    assert work.work_is_done_training
    assert os.path.exists(results_dir_0)
    assert "predictions.csv" in results_artifacts_0
    assert "lightning_logs" not in results_artifacts_0
    assert "video_preds" not in results_artifacts_0

    # ----------------
    # output videos
    # ----------------
    config_overrides["eval"]["predict_vids_after_training"] = True
    config_overrides["eval"]["save_vids_after_training"] = True
    model_name_1 = datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST")
    results_dir_1 = os.path.join(base_dir, MODELS_DIR, model_name_1)
    work._train(
        config_file=os.path.join(tmp_proj_dir, flow.config_name),
        config_overrides=config_overrides,
        results_dir=results_dir_1,
    )
    results_artifacts_1 = os.listdir(results_dir_1)
    assert work.work_is_done_training
    assert os.path.exists(results_dir_1)
    assert "predictions.csv" in results_artifacts_1
    assert "lightning_logs" not in results_artifacts_1
    assert MODEL_VIDEO_PREDS_TRAIN_DIR in results_artifacts_1
    labeled_vid_dir = os.path.join(results_dir_1, MODEL_VIDEO_PREDS_TRAIN_DIR, "labeled_videos")
    assert os.path.exists(labeled_vid_dir)
    assert len(os.listdir(labeled_vid_dir)) > 0

    # ----------------
    # infer
    # ----------------
    work._run_inference(
        model_dir=os.path.join(tmp_proj_dir, MODELS_DIR, model_name_0),
        video_file=video_file,
    )
    results_dir_2 = os.path.join(base_dir, MODELS_DIR, model_name_0, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_2 = os.listdir(results_dir_2)
    assert work.work_is_done_inference
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_2
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_2
    # assert preds.replace(".csv", "_pca_singleview_error.csv") in results_artifacts_2
    # assert preds.replace(".csv", "_pca_multiview_error.csv") in results_artifacts_2
    assert preds.replace(".csv", ".short.mp4") in results_artifacts_2
    assert preds.replace(".csv", ".short.csv") in results_artifacts_2
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_2

    # ----------------
    # fiftyone
    # ----------------
    # just run and make sure it doesn't fail
    work._make_fiftyone_dataset(
        config_file=os.path.join(tmp_proj_dir, flow.config_name),
        results_dir=results_dir_1,
        config_overrides=config_overrides,
    )

    # ----------------
    # clean up
    # ----------------
    del flow
    del work


def test_train_infer_ui(root_dir, tmp_proj_dir, video_file):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app.ui.project import ProjectUI
    from lightning_pose_app.ui.train_infer import TrainUI, VIDEO_LABEL_NONE

    base_dir = os.path.join(root_dir, tmp_proj_dir)

    flow = TrainUI()

    # set attributes
    flow.proj_dir = "/" + str(tmp_proj_dir)
    flow.st_train_status = {
        "super": "initialized",
        "semisuper": None,
        "super ctx": None,
        "semisuper ctx": None,
    }
    flow.st_losses = {"super": []}
    flow.st_train_label_opt = VIDEO_LABEL_NONE  # don't run inference on vids
    flow.st_max_epochs = 2

    # ----------------
    # helper flow
    # ----------------
    # load default config and pass to project manager
    config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
    default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))
    flowp = ProjectUI(
        data_dir="/data",
        default_config_dict=default_config_dict,
    )
    proj_name = os.path.split(tmp_proj_dir)[-1]
    flowp.run(action="update_paths", project_name=proj_name)
    flowp.run(action="update_frame_shapes")
    flowp.run(
        action="update_project_config",
        new_vals_dict={
            "data": {"num_keypoints": 17},
            "training": {"check_val_every_n_epoch": 2},  # match flow.st_max_epochs
        },
    )

    # ----------------
    # train
    # ----------------
    model_name_0 = "date_flow/time_flow"
    flow.st_datetimes = {"super": model_name_0}
    flow.run(action="train", config_filename=f"model_config_{proj_name}.yaml")

    # check flow state
    assert flow.st_train_status["super"] == "complete"
    assert flow.work.progress == 0.0
    assert flow.work.work_is_done_training

    # check output files
    results_dir_0 = os.path.join(base_dir, MODELS_DIR, model_name_0)
    results_artifacts_0 = os.listdir(results_dir_0)
    assert os.path.exists(results_dir_0)
    assert "predictions.csv" in results_artifacts_0
    assert "lightning_logs" not in results_artifacts_0
    assert "video_preds" not in results_artifacts_0

    # ----------------
    # infer
    # ----------------
    flow.st_infer_status[video_file] = "initialized"
    flow.st_inference_model = model_name_0
    flow.run(action="run_inference", video_files=[video_file], testing=True)

    # check flow state
    assert flow.st_infer_status[video_file] == "complete"
    assert flow.work_is_done_inference
    assert len(flow.works_dict) == 0

    # check output files
    results_dir_1 = os.path.join(base_dir, MODELS_DIR, model_name_0, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_1 = os.listdir(results_dir_1)
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_1
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_1
    # assert preds.replace(".csv", "_pca_singleview_error.csv") in results_artifacts_2
    # assert preds.replace(".csv", "_pca_multiview_error.csv") in results_artifacts_2
    assert preds.replace(".csv", ".short.mp4") in results_artifacts_1
    assert preds.replace(".csv", ".short.csv") in results_artifacts_1
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_1

    # ----------------
    # determine type
    # ----------------
    flow.run(action="determine_dataset_type")
    assert not flow.allow_context

    # ----------------
    # clean up
    # ----------------
    del flowp
    del flow
