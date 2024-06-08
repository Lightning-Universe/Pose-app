import os
import shutil
import yaml

from lightning_pose_app import (
    LIGHTNING_POSE_DIR,
    LABELSTUDIO_DB_DIR,
    LABELSTUDIO_METADATA_FILENAME,
    MODELS_DIR,
)


def test_project_ui(root_dir, tmp_proj_dir):

    from lightning_pose_app.ui.project import ProjectUI

    proj_name = os.path.split(tmp_proj_dir)[-1]
    proj_dir_abs = os.path.join(root_dir, tmp_proj_dir)

    # load default config and pass to project manager
    config_dir = os.path.join(LIGHTNING_POSE_DIR, "scripts", "configs")
    default_config_dict = yaml.safe_load(open(os.path.join(config_dir, "config_default.yaml")))

    flow = ProjectUI(
        data_dir="/data",
        default_config_dict=default_config_dict,
    )

    # -------------------
    # test find projects
    # -------------------
    flow.run(action="find_initialized_projects")
    assert proj_name in flow.initialized_projects
    assert LABELSTUDIO_DB_DIR not in flow.initialized_projects

    # -------------------
    # test update paths
    # -------------------
    flow.run(action="update_paths")
    assert flow.proj_dir is None

    flow.run(action="update_paths", project_name=proj_name)
    assert flow.proj_dir == "/" + str(tmp_proj_dir)
    assert flow.config_name == f"model_config_{proj_name}.yaml"
    assert flow.config_file == "/" + os.path.join(tmp_proj_dir, flow.config_name)
    assert flow.model_dir == "/" + os.path.join(tmp_proj_dir, MODELS_DIR)
    assert flow.proj_dir_abs == proj_dir_abs

    # -------------------
    # test update config
    # -------------------
    # config dict is initially none
    assert flow.config_dict is None

    # update project config with no vals should set config_dict to defaults (+ "keypoints"} and
    # NOT save out config
    config_init = flow.default_config_dict.copy()
    config_init["data"]["keypoints"] = None
    flow.run(action="update_project_config", new_vals=None)
    assert flow.config_dict == config_init
    assert not os.path.exists(os.path.join(root_dir, flow.config_file[1:]))

    # update with new vals; should update object attribute and yaml file
    new_vals_dict_0 = {
        "data": {"kepoints": ["nose", "tail"], "num_keypoints": 2},
        "training": {"train_batch_size": 2},
    }
    flow.run(action="update_project_config", new_vals_dict=new_vals_dict_0)
    config_file = os.path.join(root_dir, flow.config_file[1:])
    assert os.path.exists(config_file)
    config_dict_saved = yaml.safe_load(open(config_file))
    for key1, val1 in new_vals_dict_0.items():
        for key2, val2 in val1.items():
            assert flow.config_dict[key1][key2] == val2
            assert config_dict_saved[key1][key2] == val2

    # update with new vals again; make sure previous new vals remain
    new_vals_dict_1 = {
        "data": {"columns_for_singleview_pca": [0, 1], "mirrored_column_matches": []},
        "training": {"val_batch_size": 2},
    }
    flow.run(action="update_project_config", new_vals_dict=new_vals_dict_1)
    config_dict_saved = yaml.safe_load(open(config_file))
    for key1, val1 in new_vals_dict_0.items():
        for key2, val2 in val1.items():
            assert flow.config_dict[key1][key2] == val2
            assert config_dict_saved[key1][key2] == val2
    for key1, val1 in new_vals_dict_1.items():
        for key2, val2 in val1.items():
            assert flow.config_dict[key1][key2] == val2
            assert config_dict_saved[key1][key2] == val2

    # -------------------
    # test update shapes
    # -------------------
    new_vals_dict_2 = {
        "data": {
            "image_orig_dims": {
                "height": 406,
                "width": 396,
            },
            "image_resize_dims": {
                "height": 256,
                "width": 256,
            },
        },
    }
    flow.run(action="update_frame_shapes")
    config_dict_saved = yaml.safe_load(open(config_file))
    for key1, val1 in new_vals_dict_2.items():
        for key2, val2 in val1.items():
            assert flow.config_dict[key1][key2] == val2
            assert config_dict_saved[key1][key2] == val2

    # -------------------
    # test update shapes
    # -------------------
    # should return None if no label studio metadata file
    metadata_file = os.path.join(root_dir, tmp_proj_dir, LABELSTUDIO_METADATA_FILENAME)
    flow.run(action="compute_labeled_frame_fraction")
    if not os.path.exists(metadata_file):
        assert flow.n_labeled_frames is None
        assert flow.n_total_frames is None
    else:
        assert flow.n_labeled_frames is not None
        assert flow.n_total_frames is not None

    # -------------------
    # test load defaults
    # -------------------
    assert flow.st_keypoints_ == []
    assert flow.st_n_keypoints == 0
    assert flow.st_pcasv_columns == []
    assert flow.st_pcamv_columns == []
    assert flow.st_n_views == 0
    flow.run(action="load_project_defaults")
    assert flow.st_keypoints_ == config_dict_saved["data"]["keypoints"]
    assert flow.st_n_keypoints == config_dict_saved["data"]["num_keypoints"]
    assert flow.st_n_keypoints == 2
    assert flow.st_pcasv_columns == config_dict_saved["data"]["columns_for_singleview_pca"]
    assert len(flow.st_pcasv_columns) == 2
    assert flow.st_pcamv_columns == config_dict_saved["data"]["mirrored_column_matches"]
    assert len(flow.st_pcamv_columns) == 0
    assert flow.st_n_views == 1

    # -------------------
    # test find models
    # -------------------
    m1 = "00-11-22/33-44-55"
    m2 = "aa-bb-cc/dd-ee-ff"
    m1_path = os.path.join(root_dir, tmp_proj_dir, MODELS_DIR, m1)
    m2_path = os.path.join(root_dir, tmp_proj_dir, MODELS_DIR, m2)
    os.makedirs(m1_path, exist_ok=True)
    os.makedirs(m2_path, exist_ok=True)
    flow.run(action="update_trained_models_list")
    assert len(flow.trained_models) == 2
    assert m1 in flow.trained_models
    assert m2 in flow.trained_models

    # -------------------
    # test upload existing
    # -------------------
    # TODO: fill out tests for uploading an existing project

    # -------------------
    # test delete project
    # -------------------
    # copy project
    copy_proj_name = "TEMP_TEST_PROJECT"
    src = proj_dir_abs
    dst = proj_dir_abs.replace(proj_name, copy_proj_name)
    shutil.copytree(src, dst)
    flow.run(action="find_initialized_projects")
    assert proj_name in flow.initialized_projects
    assert copy_proj_name in flow.initialized_projects

    flow.proj_dir = flow.proj_dir.replace(proj_name, copy_proj_name)
    flow.run(action="delete_project")
    assert flow.st_project_name == ""
    assert not os.path.exists(dst)
    assert copy_proj_name not in flow.initialized_projects

    # -------------------
    # cleanup
    # -------------------
    del flow
