import glob
import io
import json
import logging
import os
import zipfile

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import csv

from lightning_pose_app import COLLECTED_DATA_FILENAME, LABELED_DATA_DIR, VIDEOS_DIR

_logger = logging.getLogger('APP.BACKEND.PROJECT')


def extract_frames_from_pkg_slp(file_path, base_output_dir):
    """
    Extracts frames and keypoint data from a .slp (SLEAP) file, saving them to specified directories.
    Also generates CSV files for frame paths and keypoint coordinates.

    Args:
        file_path (str): Path to the .slp file.
        base_output_dir (str): Directory where extracted data will be saved.

    Creates:
        - labeled-data/: Directory with extracted frames.
        - videos/: Directory reserved for video-related data.
        - selected_frames.csv: CSV file with a list of frame file names per video.
        - CollectedData.csv: CSV file consolidating keypoint data and frame paths.
    """

    # ===========================
    # Step 1: Create Output Directories
    # ===========================
    os.makedirs(base_output_dir, exist_ok=True)
    labeled_data_dir = os.path.join(base_output_dir, LABELED_DATA_DIR)
    os.makedirs(labeled_data_dir, exist_ok=True)
    videos_dir = os.path.join(base_output_dir, VIDEOS_DIR)
    os.makedirs(videos_dir, exist_ok=True)
    print(f"Created output directories at {base_output_dir}")

    # ===========================
    # Step 2: Open and Parse the .slp File
    # ===========================
    with h5py.File(file_path, 'r') as hdf_file:
        video_names = {}         # Maps video group names to video filenames
        frame_dimensions = {}    # Stores (width, height) for each video group

        # ---------------------------
        # Extract Video Metadata
        # ---------------------------
        for video_group_name in hdf_file.keys():
            source_video_path = f'{video_group_name}/source_video'
            # Check if the group name starts with 'video' and contains 'source_video'
            if video_group_name.startswith('video') and source_video_path in hdf_file:
                try:
                    # Retrieve the JSON metadata for the source video
                    source_video_json = hdf_file[source_video_path].attrs.get('json', '{}')
                    if source_video_json:
                        source_video_dict = json.loads(source_video_json)
                        video_filename = source_video_dict.get('backend', {}).get('filename')
                        if video_filename:
                            video_names[video_group_name] = video_filename
                            print(f"Found video: {video_filename} in group {video_group_name}")

                            # Extract frame dimensions from the first frame
                            video_data = hdf_file[f'{video_group_name}/video'][:]
                            if len(video_data) > 0:
                                # Convert the first frame's byte data to an image
                                img = Image.open(io.BytesIO(np.array(video_data[0], dtype=np.uint8)))
                                frame_dimensions[video_group_name] = img.size  # (width, height)
                                print(f"Frame dimensions for {video_group_name}: {img.size}")
                            else:
                                print(f"No frames found in {video_group_name}/video.")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading video metadata for {video_group_name}: {e}")

        # ---------------------------
        # Determine Maximum Frame Dimensions
        # ---------------------------
        if frame_dimensions:
            # Identify the largest frame size based on area (width * height)
            max_width, max_height = max(frame_dimensions.values(), key=lambda x: x[0] * x[1])
            print(f"Maximum frame dimensions determined: {max_width}x{max_height}")
        else:
            print("No frame dimensions found. Exiting function.")
            return

        data_frames = []    # List to hold DataFrames for keypoint data
        resized_flag = False  # Indicates if any frame was resized

        # ===========================
        # Step 3: Extract and Save Frames
        # ===========================
        for video_group, video_filename in video_names.items():
            # Define the output directory for the current video's frames
            video_base_name = os.path.splitext(os.path.basename(video_filename))[0]
            output_dir = os.path.join(labeled_data_dir, video_base_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Processing video group: {video_group}, saving frames to {output_dir}")

            # Check if the 'video' dataset exists within the current group
            if f'{video_group}/video' in hdf_file:
                video_data = hdf_file[f'{video_group}/video'][:]           # Frame image data
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]  # Frame indices
                frame_names = []                                         # List to store saved frame filenames

                # Original frame dimensions for the current video
                original_width, original_height = frame_dimensions[video_group]

                # Iterate over each frame in the video
                for img_bytes, frame_number in zip(video_data, frame_numbers):
                    try:
                        # Convert frame bytes to an image
                        img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))

                        # Resize the image if its dimensions differ from the maximum dimensions
                        if (original_width, original_height) != (max_width, max_height):
                            img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                            resized_flag = True  # Set flag since resizing occurred
                            print(f"Resized frame {frame_number} to {max_width}x{max_height}")

                        # Convert the PIL image to a NumPy array for saving with OpenCV
                        img_array = np.array(img)
                        frame_name = f"img{str(frame_number).zfill(8)}.png"  # e.g., img00000001.png
                        frame_path = os.path.join(output_dir, frame_name)
                        cv2.imwrite(frame_path, img_array)  # Save the frame as a PNG file
                        frame_names.append(frame_name)
                        print(f"Saved frame {frame_number} as {frame_name}")
                    except Exception as e:
                        print(f"Error processing frame {frame_number}: {e}")

                # Save the list of frame filenames to a CSV file for reference
                csv_path = os.path.join(output_dir, "selected_frames.csv")
                try:
                    with open(csv_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        for filename in sorted(frame_names):
                            writer.writerow([filename])
                    print(f"Saved selected frames CSV to {csv_path}")
                except IOError as e:
                    print(f"Error saving selected frames CSV: {e}")
            else:
                print(f"No 'video' dataset found for {video_group}. Skipping frame extraction.")
                continue  # Proceed to the next video group

            # ===========================
            # Step 4: Extract and Save Keypoint Data
            # ===========================
            # Ensure that required datasets exist for keypoint extraction
            if 'frames' in hdf_file and 'points' in hdf_file and 'instances' in hdf_file:
                frames_dataset = hdf_file['frames']
                points_dataset = hdf_file['points']
                instances_dataset = hdf_file['instances']

                # Create a mapping from frame_id to frame_idx for the current video group
                try:
                    video_id = int(video_group.replace('video', ''))
                except ValueError:
                    print(f"Invalid video group name format: {video_group}. Skipping keypoint extraction.")
                    continue

                frame_references = {
                    frame['frame_id']: frame['frame_idx']
                    for frame in frames_dataset
                    if frame['video'] == video_id
                }

                # Create a dictionary to quickly access instances by frame_id
                instances_dict = {inst['frame_id']: inst for inst in instances_dataset}

                data = []  # List to store keypoint data rows

                # Iterate over each frame reference to extract keypoints
                for frame_id, frame_idx in frame_references.items():
                    instance = instances_dict.get(frame_id)
                    if not instance:
                        continue  # Skip if no instance data is found for the frame

                    point_id_start = instance['point_id_start']
                    point_id_end = instance['point_id_end']
                    points = points_dataset[point_id_start:point_id_end]

                    keypoints_flat = []  # Flattened list of keypoint coordinates

                    for kp in points:
                        x, y, vis = kp['x'], kp['y'], kp['visible']

                        # Adjust keypoint coordinates if frames were resized
                        if resized_flag and x is not None and y is not None:
                            x = (x / original_width) * max_width
                            y = (y / original_height) * max_height

                        # Handle invalid or invisible keypoints
                        if x is None or y is None or np.isnan(x) or np.isnan(y) or not vis:
                            keypoints_flat.extend([None, None])
                        else:
                            keypoints_flat.extend([x, y])

                    # Append the frame index and its keypoints to the data list
                    data.append([frame_idx] + keypoints_flat)

                # Parse metadata to retrieve keypoint names and their ordering
                metadata_json = hdf_file['metadata'].attrs.get('json', '{}')
                if metadata_json:
                    try:
                        metadata_dict = json.loads(metadata_json)
                        nodes = metadata_dict.get('nodes', [])
                        skeletons = metadata_dict.get('skeletons', [{}])

                        if not skeletons or 'nodes' not in skeletons[0]:
                            print("Skeleton nodes not found in metadata. Skipping keypoint DataFrame creation.")
                            continue

                        keypoints = [node['name'] for node in nodes]
                        keypoints_ids = [n['id'] for n in skeletons[0].get('nodes', [])]
                        keypoints_ordered = [keypoints[idx] for idx in keypoints_ids]

                        # Define DataFrame columns: 'frame', 'keypoint1_x', 'keypoint1_y', ...
                        columns = ['frame']
                        for kp in keypoints_ordered:
                            columns.extend([f'{kp}_x', f'{kp}_y'])

                        # Create a DataFrame with the extracted keypoint data
                        labels_df = pd.DataFrame(data, columns=columns)

                        # Update the 'frame' column to include the relative path to the frame image
                        labels_df['frame'] = labels_df['frame'].apply(
                            lambda x: f"labeled-data/{video_base_name}/img{str(int(x)).zfill(8)}.png"
                        )

                        # Remove duplicate frames by keeping the first occurrence
                        labels_df = labels_df.groupby('frame', as_index=False).first()

                        # Append the DataFrame to the list of data frames
                        data_frames.append(labels_df)
                        print(f"Extracted keypoint data for video group {video_group}")
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"Error parsing metadata for keypoints: {e}")
                else:
                    print("Metadata JSON not found. Skipping keypoint DataFrame creation.")
            else:
                print("Required datasets ('frames', 'points', 'instances') not found. Skipping keypoint extraction.")

    # ===========================
    # Step 5: Consolidate and Save Keypoint Data
    # ===========================
    if data_frames:
        # Filter out DataFrames that do not contain the 'frame' column or are empty
        non_empty_data_frames = [
            df for df in data_frames
            if 'frame' in df.columns and not df['frame'].isnull().all()
        ]

        if non_empty_data_frames:
            # Combine all non-empty DataFrames into a single DataFrame
            combined_df = pd.concat(non_empty_data_frames, ignore_index=True)
            print(f"Combined DataFrame shape: {combined_df.shape}")

            # Validate that each frame path exists in the filesystem
            combined_df['valid'] = combined_df['frame'].apply(
                lambda path: os.path.exists(os.path.join(base_output_dir, path))
            )
            # Filter out rows with invalid frame paths
            valid_combined_df = combined_df[combined_df['valid']].drop(columns=['valid'])
            print(f"Valid keypoint entries after path verification: {valid_combined_df.shape[0]}")

            if not valid_combined_df.empty:
                # Construct header rows for the final CSV
                scorer_row = ['scorer'] + ['lightning_tracker'] * (len(valid_combined_df.columns) - 1)
                bodyparts_row = ['bodyparts'] + [col.split('_')[0] for col in valid_combined_df.columns[1:]]
                coords_row = ['coords'] + [col.split('_')[1] for col in valid_combined_df.columns[1:]]

                # Create a header DataFrame
                header_df = pd.DataFrame([scorer_row, bodyparts_row, coords_row], columns=valid_combined_df.columns)

                # Combine header and data
                final_df = pd.concat([header_df, valid_combined_df], ignore_index=True)

                # Remove column headers for proper CSV formatting
                final_df.columns = [None] * len(final_df.columns)

                # Define the path for the final consolidated CSV
                final_csv_path = os.path.join(base_output_dir, COLLECTED_DATA_FILENAME)

                try:
                    # Save the final DataFrame to CSV without headers and index
                    final_df.to_csv(final_csv_path, index=False, header=False)
                    print(f"Saved keypoint data to {final_csv_path}")
                except IOError as e:
                    print(f"Error saving keypoint data CSV: {e}")
            else:
                print("No valid frame paths found after verification. Final CSV not created.")
        else:
            print("No non-empty DataFrames with 'frame' column to consolidate.")
    else:
        print("No keypoint data was extracted. Final CSV not created.")


def get_keypoints_from_pkg_slp(file_path: str) -> list:
    keypoints = []

    with h5py.File(file_path, 'r') as hdf_file:
        # Extract instance names from metadata JSON
        metadata_json = hdf_file['metadata'].attrs['json']
        metadata_dict = json.loads(metadata_json)
        nodes = metadata_dict['nodes']
        keypoints = [node['name'] for node in nodes]

    return keypoints


def get_keypoints_from_zipfile(file_path: str, project_type: str = "Lightning Pose") -> list:
    if project_type not in ["DLC", "Lightning Pose"]:
        raise NotImplementedError
    keypoints = []
    with zipfile.ZipFile(file_path) as z:
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


def find_models(
    model_dir: str,
    must_contain_predictions: bool = True,
    must_contain_config: bool = False
) -> list:

    trained_models = []
    # this returns a list of model training days
    dirs_day = os.listdir(model_dir)
    # loop over days and find HH-MM-SS
    for dir_day in dirs_day:
        fullpath1 = os.path.join(model_dir, dir_day)
        dirs_time = os.listdir(fullpath1)
        for dir_time in dirs_time:
            fullpath2 = os.path.join(fullpath1, dir_time)
            if (
                must_contain_predictions
                and not os.path.exists(os.path.join(fullpath2, "predictions.csv"))
            ):
                # skip this model folder if it does not contain predictions.csv file
                continue
            if (
                must_contain_config
                and not os.path.exists(os.path.join(fullpath2, "config.yaml"))
            ):
                # skip this model folder if it does not contain config.yaml file
                continue
            trained_models.append('/'.join(fullpath2.split('/')[-2:]))
    return trained_models
