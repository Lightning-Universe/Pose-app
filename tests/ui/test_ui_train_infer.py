from datetime import datetime
import os
import yaml

from lightning_pose_app import (
    ENSEMBLE_MEMBER_FILENAME,
    LIGHTNING_POSE_DIR,
    MODELS_DIR,
    MODEL_VIDEO_PREDS_INFER_DIR,
    MODEL_VIDEO_PREDS_TRAIN_DIR,
    VIDEOS_DIR,
)
from lightning_pose_app.ui.train_infer import create_ensemble_directory


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
            "max_epochs": 1,
            "check_val_every_n_epoch": 1,
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
    assert work.work_is_done
    assert os.path.exists(results_dir_0)
    assert "predictions.csv" in results_artifacts_0
    assert "lightning_logs" not in results_artifacts_0
    assert MODEL_VIDEO_PREDS_TRAIN_DIR not in results_artifacts_0

    # ----------------------------
    # train, output videos
    # ----------------------------
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
    assert work.work_is_done
    assert os.path.exists(results_dir_1)
    assert "predictions.csv" in results_artifacts_1
    assert "lightning_logs" not in results_artifacts_1
    assert MODEL_VIDEO_PREDS_TRAIN_DIR in results_artifacts_1
    labeled_vid_dir = os.path.join(results_dir_1, MODEL_VIDEO_PREDS_TRAIN_DIR, "labeled_videos")
    assert os.path.exists(labeled_vid_dir)
    assert len(os.listdir(labeled_vid_dir)) > 0

    # ----------------------------
    # infer, output labeled clip
    # ----------------------------
    work._run_inference(
        model_dir=os.path.join(tmp_proj_dir, MODELS_DIR, model_name_0),
        video_file=video_file,
        make_labeled_video_full=False,
        make_labeled_video_clip=True,
    )
    results_dir_2 = os.path.join(base_dir, MODELS_DIR, model_name_0, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_2 = os.listdir(results_dir_2)
    assert work.work_is_done
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_2
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_2
    assert preds.replace(".csv", ".labeled.mp4") not in results_artifacts_2
    assert preds.replace(".csv", ".short.mp4") in results_artifacts_2
    assert preds.replace(".csv", ".short.csv") in results_artifacts_2
    assert preds.replace(".csv", ".short_temporal_norm.csv") in results_artifacts_2
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_2

    # ----------------------------
    # infer, output full labeled
    # ----------------------------
    # also tests loading of predictions from previous inference
    work._run_inference(
        model_dir=os.path.join(tmp_proj_dir, MODELS_DIR, model_name_0),
        video_file=video_file,
        make_labeled_video_full=True,
        make_labeled_video_clip=False,
    )
    results_artifacts_2 = os.listdir(results_dir_2)
    assert work.work_is_done
    assert preds.replace(".csv", ".labeled.mp4") in results_artifacts_2

    # ----------------------------
    # run eks
    # ----------------------------
    model_name_eks = datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST")
    ensemble_dir = os.path.join(tmp_proj_dir, MODELS_DIR, model_name_eks)
    work._run_eks(
        ensemble_dir=ensemble_dir,
        model_dirs=[
            "/" + os.path.join(tmp_proj_dir, MODELS_DIR, model_name_0),
            "/" + os.path.join(tmp_proj_dir, MODELS_DIR, model_name_1),
        ],
        video_file=video_file,
        make_labeled_video_full=True,
        make_labeled_video_clip=False,
    )
    results_dir_eks = os.path.join(
        base_dir, MODELS_DIR, model_name_eks, MODEL_VIDEO_PREDS_INFER_DIR,
    )
    results_artifacts_eks = os.listdir(results_dir_eks)
    assert work.work_is_done
    assert preds in results_artifacts_eks
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_eks
    assert preds.replace(".csv", ".labeled.mp4") in results_artifacts_eks

    # ----------------------------
    # fiftyone
    # ----------------------------
    # just run and make sure it doesn't fail
    work._make_fiftyone_dataset(
        config_file=os.path.join(tmp_proj_dir, flow.config_name),
        results_dir=results_dir_1,
        config_overrides=config_overrides,
    )

    # ----------------------------
    # clean up
    # ----------------------------
    del flow
    del work


def test_train_infer_ui(root_dir, tmp_proj_dir, video_file):
    """Test run methods of TrainUI object."""

    from lightning_pose_app.ui.project import ProjectUI
    from lightning_pose_app.ui.train_infer import TrainUI, VIDEO_LABEL_NONE

    base_dir = os.path.join(root_dir, tmp_proj_dir)

    flow = TrainUI()

    # set attributes
    rng_seed_0 = 0
    flow.proj_dir = "/" + str(tmp_proj_dir)
    flow.n_labeled_frames = 90
    flow.n_total_frames = 90
    flow.st_losses = []
    flow.st_train_label_opt = VIDEO_LABEL_NONE  # don't run inference on vids
    flow.st_max_epochs = 1

    # ----------------------------
    # helper flow
    # ----------------------------
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
            "training": {"check_val_every_n_epoch": 1},  # match flow.st_max_epochs
        },
    )

    # ----------------------------
    # train 0
    # ----------------------------
    flow.st_datetime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S")
    flow.st_train_flag["super"] = True
    flow.st_rng_seed_data_pt = [rng_seed_0]
    flow.st_train_status = {}
    flow.run(action="train", config_filename=f"model_config_{proj_name}.yaml")

    # check flow state
    assert len(flow.st_train_status.keys()) == 4  # 4 potential models to fit
    assert sum([val == "complete" for _, val in flow.st_train_status.items()]) == 1
    for key, val in flow.st_train_status.items():
        if val == "complete":
            model_name_0 = key
            break
    assert flow.work.progress == 0.0
    assert flow.work.work_is_done

    # check output files
    results_dir_0 = os.path.join(base_dir, MODELS_DIR, model_name_0)
    results_artifacts_0 = os.listdir(results_dir_0)
    assert os.path.exists(results_dir_0)
    assert "predictions.csv" in results_artifacts_0
    assert "lightning_logs" not in results_artifacts_0
    assert "video_preds" not in results_artifacts_0

    # check config
    config_file = os.path.join(results_dir_0, "config.yaml")
    config_dict = yaml.safe_load(open(config_file))
    assert len(config_dict["model"]["losses_to_use"]) == 0
    assert config_dict["training"]["rng_seed_data_pt"] == rng_seed_0

    # ----------------------------
    # train 1 (for eks)
    # ----------------------------
    rng_seed_1 = 1
    flow.st_rng_seed_data_pt = [rng_seed_1]
    flow.st_train_status = {}
    flow.run(action="train", config_filename=f"model_config_{proj_name}.yaml")

    # check
    for key, val in flow.st_train_status.items():
        if val == "complete":
            model_name_1 = key
            break
    results_dir_1 = os.path.join(base_dir, MODELS_DIR, model_name_1)
    config_file = os.path.join(results_dir_1, "config.yaml")
    config_dict = yaml.safe_load(open(config_file))
    assert len(config_dict["model"]["losses_to_use"]) == 0
    assert config_dict["training"]["rng_seed_data_pt"] == rng_seed_1

    # ----------------------------
    # inference (single model)
    # ----------------------------
    flow.st_inference_model = model_name_0
    flow.st_label_full = False
    flow.st_label_short = True
    flow.st_infer_status = {}
    flow.run(action="run_inference", video_files=[video_file], testing=True)

    # check flow state
    keys = list(flow.st_infer_status.keys())
    assert len(keys) == 1
    assert flow.st_infer_status[keys[0]] == "complete"
    assert flow.work_is_done_inference
    assert len(flow.works_dict) == 0

    # check output files
    results_dir_0a = os.path.join(base_dir, MODELS_DIR, model_name_0, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_0a = os.listdir(results_dir_0a)
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_0a
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_0a
    assert preds.replace(".csv", ".labeled.mp4") not in results_artifacts_0a
    assert preds.replace(".csv", ".short.mp4") in results_artifacts_0a
    assert preds.replace(".csv", ".short.csv") in results_artifacts_0a
    assert preds.replace(".csv", ".short_temporal_norm.csv") in results_artifacts_0a
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_0a

    # ----------------------------
    # inference (ensemble/eks)
    # ----------------------------
    st_datetime = datetime.today().strftime("%Y-%m-%d/%H-%M-%S_eks")
    ensemble_dir = os.path.join(tmp_proj_dir, MODELS_DIR, st_datetime)
    create_ensemble_directory(
        ensemble_dir=ensemble_dir,
        model_dirs=[
            "/" + os.path.join(tmp_proj_dir, MODELS_DIR, model_name_0),
            "/" + os.path.join(tmp_proj_dir, MODELS_DIR, model_name_1),
        ],
    )

    flow.st_inference_model = st_datetime  # takes relative path
    flow.st_label_full = True
    flow.st_label_short = True
    flow.st_infer_status = {}
    flow.run(action="run_inference", video_files=[video_file], testing=True)

    # check flow state
    assert len(flow.st_infer_status) == 1  # one for eks
    for key, val in flow.st_infer_status.items():
        assert val == "complete"
    assert flow.work_is_done_inference
    assert flow.work_is_done_eks
    assert len(flow.works_dict) == 0

    # check that inference was run with model 1
    results_dir_1a = os.path.join(base_dir, MODELS_DIR, model_name_1, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_1a = os.listdir(results_dir_1a)
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_1a
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_1a
    assert preds.replace(".csv", ".labeled.mp4") in results_artifacts_1a
    assert preds.replace(".csv", ".short.mp4") in results_artifacts_1a
    assert preds.replace(".csv", ".short.csv") in results_artifacts_1a
    assert preds.replace(".csv", ".short_temporal_norm.csv") in results_artifacts_1a
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_1a

    # check that eks was run
    results_dir_eks = os.path.join(base_dir, MODELS_DIR, st_datetime, MODEL_VIDEO_PREDS_INFER_DIR)
    results_artifacts_eks = os.listdir(results_dir_eks)
    preds = os.path.basename(video_file).replace(".mp4", ".csv")
    assert preds in results_artifacts_eks
    assert preds.replace(".csv", "_temporal_norm.csv") in results_artifacts_eks
    assert preds.replace(".csv", ".labeled.mp4") in results_artifacts_eks
    assert preds.replace(".csv", ".short.csv") in results_artifacts_eks
    assert preds.replace(".csv", ".short_temporal_norm.csv") in results_artifacts_eks
    assert preds.replace(".csv", ".short.labeled.mp4") in results_artifacts_eks

    # ----------------------------
    # determine dataset type
    # ----------------------------
    flow.run(action="determine_dataset_type")
    assert not flow.allow_context

    # ----------------------------
    # clean up
    # ----------------------------
    del flowp
    del flow


def test_create_ensemble_directory(tmpdir):

    ensemble_dir = os.path.join(tmpdir, "ensemble_dir")
    model_dirs = [
        os.path.join(tmpdir, "model_dir_0"),
        os.path.join(tmpdir, "model_dir_1"),
    ]
    create_ensemble_directory(ensemble_dir=ensemble_dir, model_dirs=model_dirs)

    # check file exists
    ensemble_list_file = os.path.join(ensemble_dir, ENSEMBLE_MEMBER_FILENAME)
    assert os.path.isfile(ensemble_list_file)

    # check file contains the correct paths
    with open(ensemble_list_file, "r") as file:
        model_dirs_saved = [line.strip() for line in file.readlines()]
    assert len(model_dirs_saved) == len(model_dirs)
    for model_dir in model_dirs:
        assert model_dir in model_dirs_saved
