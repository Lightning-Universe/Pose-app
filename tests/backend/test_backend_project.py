import os
import h5py

import pandas as pd
import zipfile


def test_extract_video_names_from_pkg_slp(root_dir, tmpdir):

    from lightning_pose_app.backend.project import extract_video_names_from_pkg_slp

    hdf_file_path = os.path.join(root_dir, "tests/backend/test_sleap.pkg.slp")
    with h5py.File(hdf_file_path, 'r') as hdf_file:
        video_names = extract_video_names_from_pkg_slp(hdf_file)
        assert isinstance(video_names, dict)
        assert len(video_names) == 2
        for key, value in video_names.items():
            assert key.startswith('video')
            assert value.endswith('.mp4')


def test_extract_frames_from_pkg_slp(root_dir, tmpdir):

    from lightning_pose_app.backend.project import extract_frames_from_pkg_slp

    hdf_file_path = os.path.join(root_dir, "tests/backend/test_sleap.pkg.slp")
    base_output_dir = str(tmpdir)

    print(f"Using HDF5 file: {hdf_file_path}")
    print(f"Base output directory: {base_output_dir}")

    # Run the function
    extract_frames_from_pkg_slp(hdf_file_path, base_output_dir)

    # Check if frames are extracted correctly
    assert os.path.exists(base_output_dir)
    extracted_dirs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d))
    ]

    print(f"Extracted directories: {extracted_dirs}")

    assert len(extracted_dirs) == 1
    for extracted_dir in extracted_dirs:
        assert os.path.basename(extracted_dir) == 'labeled-data', \
            f"Expected directory name 'labeled-data', but got {os.path.basename(extracted_dir)}"

    # TODO: add more tests


def test_extract_labels_from_pkg_slp(root_dir):

    from lightning_pose_app.backend.project import extract_labels_from_pkg_slp

    hdf_file_path = os.path.join(root_dir, "tests/backend/test_sleap.pkg.slp")

    labels_df = extract_labels_from_pkg_slp(hdf_file_path)

    # remove bodyparts and get a unique list - should be {'nose', 'tail', 'paw'}
    filtered_list = [item for item in labels_df.iloc[1].tolist() if item != 'bodyparts']
    assert set(filtered_list) == {'nose', 'tail', 'paw'}

    # Verify the frame names in the first column is in the correct format
    for frame in labels_df.iloc[3:, 0]:
        video_name = 'test_vid_resized' if 'test_vid_resized' in frame else 'test_vid2_resized'
        frame_number = frame.split('/')[-1].replace('img', '').replace('.png', '')
        expected_frame_name = f"labeled-data/{video_name}/img{frame_number}.png"
        assert frame == expected_frame_name, f"Expected {expected_frame_name}, but got {frame}"


def test_get_keypoints_from_pkg_slp(root_dir):

    from lightning_pose_app.backend.project import get_keypoints_from_pkg_slp

    hdf_file_path = os.path.join(root_dir, "tests/backend/test_sleap.pkg.slp")
    keypoints = get_keypoints_from_pkg_slp(hdf_file_path)
    assert isinstance(keypoints, list)
    assert keypoints == ["nose", "tail", "paw"]


def test_get_keypoints_from_zipfile(tmpdir, root_dir, tmp_proj_dir):

    from lightning_pose_app.backend.project import get_keypoints_from_zipfile

    proj_dir_abs = os.path.join(root_dir, tmp_proj_dir)
    csv_file_dir = os.path.join(proj_dir_abs, 'CollectedData.csv')  # Use the correct CSV filename

    # Ensure the project directory exists
    os.makedirs(proj_dir_abs, exist_ok=True)

    # Load the CSV file
    df = pd.read_csv(csv_file_dir, header=[0, 1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(1).unique().tolist()

    # Create a zip file from the project directory
    proj_zipped = os.path.join(tmpdir, (proj_dir_abs + '.zip'))
    with zipfile.ZipFile(proj_zipped, 'w') as zipf:
        for root, dirs, files in os.walk(proj_dir_abs):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(proj_dir_abs, '..')),
                )

    # Run the function and check the keypoints
    keypoints_lp = get_keypoints_from_zipfile(proj_zipped, "Lightning Pose")
    assert keypoints_lp == bodyparts, f'Expected keypoints {bodyparts} but got {keypoints_lp}'

    if os.path.exists(proj_zipped):
        os.remove(proj_zipped)


def create_test_zip_file(zip_path, file_structure=None, csv_content=None):

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if file_structure:
            for file_path, file_content in file_structure.items():
                if file_content is None:
                    zipf.writestr(file_path + '/', '')
                else:
                    zipf.writestr(file_path, file_content)

        if csv_content:
            for csv_path, csv_lines in csv_content.items():
                zipf.writestr(csv_path, csv_lines)


def test_check_files_in_zipfile(tmpdir):

    from lightning_pose_app.backend.project import check_files_in_zipfile

    test_cases = [
        {
            "description": "All expected directories present for DLC",
            "file_structure": {
                "videos/": None,
                "labeled-data/": None,
                "labeled-data/test_vid/": None
            },
            "project_type": "DLC",
            "expected_error_flag": False,
            "expected_error_msg": ""
        },
        {
            "description": "Missing labeled-data directory for DLC",
            "file_structure": {
                "videos/": None
            },
            "project_type": "DLC",
            "expected_error_flag": True,
            "expected_error_msg": "ERROR: Your directory of labeled-data must be named "
                                  "\"labeled-data\" (can be empty)."
        },
        {
            "description": "All expected directories present for Lightning Pose",
            "file_structure": {
                "videos/": None,
                "labeled-data/": None
            },
            "csv_content": {
                "CollectedData.csv": "some,data\n"
            },
            "project_type": "Lightning Pose",
            "expected_error_flag": False,
            "expected_error_msg": ""
        },
        {
            "description": "Missing CollectedData.csv for Lightning Pose",
            "file_structure": {
                "videos/": None,
                "labeled-data/": None
            },
            "project_type": "Lightning Pose",
            "expected_error_flag": True,
            "expected_error_msg": "ERROR: Your directory of CollectedData.csv must be named "
                                  "\"CollectedData.csv\" (can be empty)."
        }
    ]

    for case in test_cases:
        test_zip_path = os.path.join(tmpdir, "test.zip")

        # Create test zip file
        create_test_zip_file(
            test_zip_path,
            file_structure=case.get("file_structure"),
            csv_content=case.get("csv_content")
        )

        # Run function
        error_flag, error_msg = check_files_in_zipfile(
            test_zip_path, project_type=case["project_type"]
        )

        # Assertions
        assert error_flag == case["expected_error_flag"], (
            f"Failed: {case['description']} - error_flag mismatch"
        )

        # Adjust the error message assertion to handle the HTML formatting
        if case["expected_error_msg"]:
            assert case["expected_error_msg"] in error_msg, (
                f"Failed: {case['description']} - error_msg mismatch"
            )
        else:
            assert error_msg.strip() == "<p style='font-family:sans-serif; color:Red;'></p>", (
                f"Failed: {case['description']} - error_msg mismatch"
            )


def test_collect_dlc_labels(tmpdir):

    from lightning_pose_app.backend.project import collect_dlc_labels

    labeled_data_dir = tmpdir.mkdir("labeled-data")

    subdir_video1 = labeled_data_dir.mkdir("video1")
    data_video1 = {
        "scorer": [
            "bodyparts", "coords",
            "labeled-data/video1/img00001.png",
            "labeled-data/video1/img00002.png",
            "labeled-data/video1/img00003.png"
        ],
        "lightning_tracker": ["nose", "x", 1, 2, 3],
        "lightning_tracker1": ["nose", "y", 1, 2, 3],
        "lightning_tracker2": ["tail", "x", 1, 2, 3],
        "lightning_tracker3": ["tail", "y", 1, 2, 3]
    }
    df_csv_video1 = pd.DataFrame(data_video1)
    df_csv_video1.to_csv(subdir_video1.join("CollectedData_video1.csv"))
    # Add secound dataset
    subdir_video2 = labeled_data_dir.mkdir("video2")
    data_video2 = {
        "scorer": [
            "bodyparts", "coords",
            "labeled-data/video2/img00001.png",
            "labeled-data/video2/img00002.png",
            "labeled-data/video2/img00003.png"
        ],
        "lightning_tracker": ["nose", "x", 1, 2, 3],
        "lightning_tracker1": ["nose", "y", 1, 2, 3],
        "lightning_tracker2": ["tail", "x", 1, 2, 3],
        "lightning_tracker3": ["tail", "y", 1, 2, 3]
    }
    df_csv_video2 = pd.DataFrame(data_video2)
    df_csv_video2.to_csv(subdir_video2.join("CollectedData_video2.csv"))

    # Call the function with the temporary directory
    df_all = collect_dlc_labels(tmpdir)

    df1 = pd.read_csv(
        subdir_video1.join("CollectedData_video1.csv"),
        header=[0, 1, 2],
        index_col=0
    )
    df2 = pd.read_csv(
        subdir_video2.join("CollectedData_video2.csv"),
        header=[0, 1, 2],
        index_col=0
    )
    expected_df = pd.concat([df1, df2])

    pd.testing.assert_frame_equal(df_all, expected_df)


def test_check_project_has_labels(tmp_proj_dir):

    from lightning_pose_app.backend.project import check_project_has_labels

    project_name = os.path.basename(tmp_proj_dir)
    missing_items = check_project_has_labels(
        proj_dir=tmp_proj_dir,
        project_name=project_name,
    )
    assert len(missing_items) == 1
    assert missing_items[0] == f"model_config_{project_name}.yaml"


def test_find_models(tmpdir):
    from lightning_pose_app.backend.project import find_models

    # create model directories
    models = {
        "00-11-22/33-44-55": {"predictions": True, "config": True},
        "11-22-33/44-55-66": {"predictions": True, "config": False},
        "22-33-44/55-66-77": {"predictions": False, "config": True},
        "33-44-55/66-77-88": {"predictions": False, "config": False},
    }
    model_parent = os.path.join(str(tmpdir), "models")
    for model, files in models.items():
        model_dir = os.path.join(model_parent, model)
        os.makedirs(model_dir)
        if files["predictions"]:
            os.mknod(os.path.join(model_dir, "predictions.csv"))
        if files["config"]:
            os.mknod(os.path.join(model_dir, "config.yaml"))

    # test 1: find all model directories
    trained_models = find_models(
        model_parent,
        must_contain_predictions=False,
        must_contain_config=False
    )
    for model in models.keys():
        assert model in trained_models

    # test 2: find trained model directories with predictions.csv
    trained_models = find_models(
        model_parent,
        must_contain_predictions=True,
        must_contain_config=False
    )
    for model, files in models.items():
        if files["predictions"]:
            assert model in trained_models
        else:
            assert model not in trained_models

    # test 3: find trained model directories with config.yaml
    trained_models = find_models(
        model_parent,
        must_contain_predictions=False,
        must_contain_config=True
    )
    for model, files in models.items():
        if files["config"]:
            assert model in trained_models
        else:
            assert model not in trained_models

    # test 4: find trained model directories with both predictions.csv and config.yaml
    trained_models = find_models(
        model_parent,
        must_contain_predictions=True,
        must_contain_config=True
    )
    for model, files in models.items():
        if files["predictions"] and files["config"]:
            assert model in trained_models
        else:
            assert model not in trained_models
