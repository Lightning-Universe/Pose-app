import os
import json
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import io
import zipfile
import glob
import logging
import matplotlib.pyplot as plt

from lightning_pose_app import (
    COLLECTED_DATA_FILENAME,
    LABELED_DATA_DIR,
    VIDEOS_DIR,
)

_logger = logging.getLogger('APP.BACKEND.PROJECT')


def extract_video_names_from_pkg_slp(hdf_file: str) -> dict:
    video_names = {}
    for video_group_name in hdf_file.keys():
        if video_group_name.startswith('video'):
            source_video_path = f'{video_group_name}/source_video'
            if source_video_path in hdf_file:
                source_video_json = hdf_file[source_video_path].attrs['json']
                source_video_dict = json.loads(source_video_json)
                video_filename = source_video_dict['backend']['filename']
                video_names[video_group_name] = video_filename
    return video_names


def extract_frames_from_pkg_slp(file_path: str, base_output_dir: str):
    # Extract frames from pkg.slp file to an output folder
    with h5py.File(file_path, 'r') as hdf_file:

        video_names = extract_video_names_from_pkg_slp(hdf_file)
        # Extract and save images for each video
        for video_group, video_filename in video_names.items():
            output_dir = os.path.join(
                base_output_dir, LABELED_DATA_DIR, os.path.basename(video_filename).split('.')[0]
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if video_group in hdf_file and 'video' in hdf_file[video_group]:
                video_data = hdf_file[f'{video_group}/video'][:]
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_names = []
                for i, (img_bytes, frame_number) in enumerate(zip(video_data, frame_numbers)):
                    img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                    frame_name = f"img{str(frame_number).zfill(8)}.png"
                    img.save(f"{output_dir}/{frame_name}")
                    frame_names.append(frame_name)
                    print(f"Saved frame {frame_number} as {frame_name}")


def extract_labels_from_pkg_slp(file_path: str) -> pd.DataFrame:
    # Function to extract data and create the required DataFrame for multiple videos
    data_frames = []
    scorer_row, bodyparts_row, coords_row = None, None, None

    with h5py.File(file_path, 'r') as hdf_file:

        video_names = extract_video_names_from_pkg_slp(hdf_file)

        # Extract data for each video
        for video_group, video_filename in video_names.items():
            if video_group in hdf_file and 'frames' in hdf_file:
                frames_dataset = hdf_file['frames']
                frame_references = {
                    frame['frame_id']: frame['frame_idx']
                    for frame in frames_dataset
                    if frame['video'] == int(video_group.replace('video', ''))
                }

                # Correct frame references for the current video group
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_id_to_number = {
                    frame_id: frame_numbers[idx]
                    for idx, frame_id in enumerate(frame_references.keys())
                }

                # Extract instances and points
                points_dataset = hdf_file['points']
                instances_dataset = hdf_file['instances']

                data = []
                for idx, instance in enumerate(instances_dataset):
                    try:
                        frame_id = instance['frame_id']
                        if frame_id not in frame_id_to_number:
                            continue
                        frame_idx = frame_id_to_number[frame_id]
                        point_id_start = instance['point_id_start']
                        point_id_end = instance['point_id_end']

                        points = points_dataset[point_id_start:point_id_end]

                        keypoints_flat = []
                        for kp in points:
                            x, y = kp['x'], kp['y']
                            if np.isnan(x) or np.isnan(y):
                                x, y = None, None
                            keypoints_flat.extend([x, y])

                        data.append([frame_idx] + keypoints_flat)
                    except Exception as e:
                        print(f"Skipping invalid instance {idx}: {e}")

                if data:
                    metadata_json = hdf_file['metadata'].attrs['json']
                    metadata_dict = json.loads(metadata_json)
                    nodes = metadata_dict['nodes']
                    instance_names = [node['name'] for node in nodes]

                    keypoints = [f'{name}' for name in instance_names]
                    columns = [
                        'frame'
                    ] + [
                        f'{kp}_x' for kp in keypoints
                    ] + [
                        f'{kp}_y' for kp in keypoints
                    ]
                    scorer_row = ['scorer'] + ['lightning_tracker'] * (len(columns) - 1)
                    bodyparts_row = ['bodyparts'] + [f'{kp}' for kp in keypoints for _ in (0, 1)]
                    coords_row = ['coords'] + ['x', 'y'] * len(keypoints)

                    labels_df = pd.DataFrame(data, columns=columns)
                    video_base_name = os.path.basename(video_filename).split('.')[0]
                    labels_df['frame'] = labels_df['frame'].apply(
                        lambda x: (
                            f"labeled-data/{video_base_name}/"
                            f"img{str(int(x)).zfill(8)}.png"
                        )
                    )
                    labels_df = labels_df.groupby('frame', as_index=False).first()
                    data_frames.append(labels_df)

    if data_frames:
        # Combine all data frames into a single DataFrame
        combined_df = pd.concat(data_frames, ignore_index=True)

        # Add the scorer, bodyparts, and coords rows at the top
        header_df = pd.DataFrame(
            [scorer_row, bodyparts_row, coords_row],
            columns=combined_df.columns
        )
        final_df = pd.concat([header_df, combined_df], ignore_index=True)
        final_df.columns = [None] * len(final_df.columns)  # Set header to None

        return final_df


def get_keypoints_from_pkg_slp(file_path: str) -> list:
    keypoints = []

    with h5py.File(file_path, 'r') as hdf_file:
        # Extract instance names from metadata JSON
        metadata_json = hdf_file['metadata'].attrs['json']
        metadata_dict = json.loads(metadata_json)
        nodes = metadata_dict['nodes']
        keypoints = [node['name'] for node in nodes]

    return keypoints


def get_keypoints_from_zipfile(filepath: str, project_type: str = "Lightning Pose") -> list:
    if project_type not in ["DLC", "Lightning Pose"]:
        raise NotImplementedError
    keypoints = []
    with zipfile.ZipFile(filepath) as z:
        for filename in z.namelist():
            if project_type in ["DLC", "Lightning Pose"]:
                if filename.endswith('.csv'):
                    with z.open(filename) as f:
                        for idx, line in enumerate(f):
                            if idx == 1:
                                header = line.decode('utf-8').split(',')
                                if len(header) % 2 == 0:
                                    # new DLC format
                                    keypoints = header[2::2]
                                else:
                                    # LP/old DLC format
                                    keypoints = header[1::2]
                                break
            if len(keypoints) > 0:
                break
    return keypoints


def check_files_in_zipfile(filepath: str, project_type: str = "Lightning Pose") -> tuple:
    if project_type not in ["DLC", "Lightning Pose"]:
        raise NotImplementedError

    if project_type == "DLC":
        expected_dirs = [VIDEOS_DIR, LABELED_DATA_DIR]
    else:
        expected_dirs = [VIDEOS_DIR, LABELED_DATA_DIR, COLLECTED_DATA_FILENAME]

    error_flag = False
    error_msgs = []  # Collect error messages in a list

    with zipfile.ZipFile(filepath) as z:
        files = z.namelist()

        # Iterate over each expected directory and check if it's present
        for expected_dir in expected_dirs:
            # Adjusting the logic to check the presence of directories correctly
            if not any(f"{expected_dir}" in file for file in files):
                error_flag = True
                # Append specific error message for the missing directory
                error_msgs.append(
                    f"ERROR: Your directory of {expected_dir} must be named "
                    f"\"{expected_dir}\" (can be empty)."
                )

    # Joining all error messages with breaks for HTML formatting,
    # if you're displaying this in a web context
    error_msg = "<br /><br />".join(error_msgs)

    proceed_fmt = "<p style='font-family:sans-serif; color:Red;'>%s</p>"

    return error_flag, proceed_fmt % error_msg


def collect_dlc_labels(dlc_dir: str) -> pd.DataFrame:
    """Collect video-specific labels from DLC project and save in a single pandas dataframe."""

    dirs = os.listdir(os.path.join(dlc_dir, "labeled-data"))
    dirs.sort()
    dfs = []
    for d in dirs:
        try:
            csv_file = glob.glob(os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.csv"))[0]
            df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            if len(df_tmp.index.unique()) != df_tmp.shape[0]:
                # new DLC labeling scheme that splits video/image in different cells
                levels1 = ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2")
                vids = df_tmp.loc[:, levels1]
                levels2 = ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2")
                imgs = df_tmp.loc[:, levels2]
                new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                df_tmp1 = df_tmp.drop(levels1, axis=1)
                df_tmp2 = df_tmp1.drop(levels2, axis=1)
                df_tmp2.index = new_col
                df_tmp = df_tmp2
        except IndexError:
            try:
                h5_file = glob.glob(
                    os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.h5")
                )[0]
                df_tmp = pd.read_hdf(h5_file)
                if isinstance(df_tmp.index, pd.core.indexes.multi.MultiIndex):
                    # new DLC labeling scheme that splits video/image in different cells
                    imgs = [i[2] for i in df_tmp.index]
                    vids = [df_tmp.index[0][1] for _ in imgs]
                    new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                    df_tmp1 = df_tmp.reset_index().drop(
                        columns="level_0").drop(columns="level_1").drop(columns="level_2")
                    df_tmp1.index = new_col
                    df_tmp = df_tmp1
            except IndexError:
                _logger.error(f"Could not find labels for {d}; skipping")
                continue

        dfs.append(df_tmp)
    df_all = pd.concat(dfs)

    return df_all


def zip_project_for_export(proj_dir: str) -> str:

    project_name = os.path.basename(proj_dir)
    zip_filename = f"{project_name}.zip"
    zip_filepath = os.path.join(os.path.dirname(proj_dir), zip_filename)

    if not os.path.exists(proj_dir):
        raise FileNotFoundError(f"The project directory {proj_dir} does not exist.")

    items_to_zip = [
        os.path.join(proj_dir, LABELED_DATA_DIR),
        os.path.join(proj_dir, COLLECTED_DATA_FILENAME),
        os.path.join(proj_dir, f'model_config_{project_name}.yaml')
    ]

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in items_to_zip:
            if os.path.exists(item):
                if os.path.isdir(item):
                    for root, _, files in os.walk(item):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, proj_dir))
                else:
                    zipf.write(item, os.path.relpath(item, proj_dir))
            else:
                raise FileNotFoundError(
                    f"The item {item} does not exist in the project directory."
                )

    return zip_filepath


def check_project_has_labels(proj_dir: str, project_name: str) -> list:

    labeled_data_dir = os.path.join(proj_dir, LABELED_DATA_DIR)
    collected_data_file = os.path.join(proj_dir, COLLECTED_DATA_FILENAME)
    config_file = os.path.join(proj_dir, f'model_config_{project_name}.yaml')

    missing_items = []
    if not os.path.exists(labeled_data_dir):
        missing_items.append('labeled-data directory')
    if not os.path.exists(collected_data_file):
        missing_items.append(COLLECTED_DATA_FILENAME)
    if not os.path.exists(config_file):
        missing_items.append(f'model_config_{project_name}.yaml')

    return missing_items


#### Check labels functions-----------------------------------------------------------------

def get_frame_number(filename: str) -> int:
    return int(filename.replace('img', '').replace('.png', ''))


def get_frame_paths(video_folder_path: str):
    return [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith('.png')]

# Adjust the convert_csv_to_dict function
def convert_csv_to_dict(csv_path: str) -> dict:
    proj_dir = os.path.dirname(csv_path)
    try:
        annotations = pd.read_csv(csv_path, header=[1, 2], index_col=0)
        data_dict = {}
        for index, row in annotations.iterrows():
            frame_rel_path = index
            video = os.path.basename(os.path.dirname(frame_rel_path))
            frame_number = get_frame_number(os.path.basename(frame_rel_path))

            bodyparts = {}
            for bodypart in annotations.columns.levels[0]:
                try:
                    x = row[(bodypart, 'x')]
                    y = row[(bodypart, 'y')]
                    bodyparts[bodypart] = {'x': x, 'y': y}
                except KeyError as e:
                    print(f"Error extracting {bodypart} coordinates: {e}")

            data_dict[frame_rel_path] = {
                'frame_full_path': os.path.join(proj_dir, frame_rel_path),
                'video': video,
                'frame_number': frame_number,
                'bodyparts': bodyparts
            }
        return data_dict
    except Exception as e:
        print(f"Error converting CSV to dictionary: {e}")
    return {}


def validate_annotations_dict(annotations_dict: dict):
        for frame_path, data in annotations_dict.items():
            for bodypart, coords in data['bodyparts'].items():
                if 'x' not in coords or 'y' not in coords:
                    st.write(f"Missing coordinates for {bodypart} in frame {frame_path}")
                elif np.isnan(coords['x']) or np.isnan(coords['y']):
                    print(f"Invalid coordinates for {bodypart} in frame {frame_path}: ({coords['x']}, {coords['y']})")

# change name to annotate_frame
def annotate_frames(image_path: str, annotations: dict, output_path: str):
    try:
        image = Image.open(image_path)
        fig, ax = plt.subplots()

        if image.mode == 'L':  # Grayscale
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)

        for label, coords in annotations.items():
            try:
                x = coords['x']
                y = coords['y']
                ax.plot(x, y, 'ro')  # Red dot marker
                ax.text(x, y, label, color='white', fontsize=12, ha='right', va='bottom')
            except ValueError as e:
                print(f"Error plotting {label}: {e}")

        ax.axis('off')

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        output_file = os.path.join(output_path, os.path.basename(image_path))
        plt.savefig(output_file)
        plt.close()
        _logger.info(f"Annotated frame saved at: {output_file}")
    except Exception as e:
        _logger.error(f"Failed to plot annotations for {image_path}: {e}")


# def save_annotated_frames(proj_dir: str) -> None:
#     labeled_data_path = os.path.join(proj_dir, 'labeled-data')
#     labeled_data_check_path = os.path.join(proj_dir, 'labeled-data-check')

#     # Ensure the directory structure is created
#     os.makedirs(labeled_data_check_path, exist_ok=True)

#     # Convert CSV to dictionary
#     collected_data_file_path = os.path.join(proj_dir, COLLECTED_DATA_FILENAME)
#     annotations_dict = convert_csv_to_dict(proj_dir)

#     # Copy and annotate frames
#     for frame_full_path, data in annotations_dict.items():
#         video = data['video']
#         frame_annotations = data['bodyparts']

#         video_folder_path = os.path.join(labeled_data_check_path, video)
#         frame_full_path_abs = data['frame_full_path']

#         if os.path.exists(frame_full_path_abs):  # Ensure the frame file exists before attempting to plot
#             plot_frame_with_annotations(frame_full_path_abs, frame_annotations, video_folder_path)
#         else:
#             print(f"File not found: {frame_full_path_abs}")