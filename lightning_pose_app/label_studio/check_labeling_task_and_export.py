"""Check for new labels, export to lightning pose format, export database to FileSystem."""

import argparse
import logging
import os
import pickle
import yaml

from lightning_pose_app import LABELSTUDIO_METADATA_FILENAME, LABELSTUDIO_TASKS_FILENAME
from lightning_pose_app import COLLECTED_DATA_FILENAME
from lightning_pose_app.label_studio.utils import connect_to_label_studio
from lightning_pose_app.label_studio.utils import get_project
from lightning_pose_app.label_studio.utils import LabelStudioJSONProcessor


_logger = logging.getLogger('APP.LABELSTUDIO')

_logger.info("Executing check_labeling_task_and_export.py")

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--proj_dir', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--keypoints_list', type=str)  # assume keypoint names separates by colons
args = parser.parse_args()

# connect to label studio
_logger.debug("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
_logger.debug("Connected to LabelStudio at %s" % args.label_studio_url)

# get current project
metadata_file = os.path.join(args.proj_dir, LABELSTUDIO_METADATA_FILENAME)
try:
    metadata = yaml.safe_load(open(metadata_file, "r"))
except FileNotFoundError:
    _logger.warning(f"Cannot find {metadata_file} in {args.proj_dir}")
    exit()
label_studio_project = get_project(label_studio_client=label_studio_client, id=metadata["id"])
print("Fetched Project ID: %s, Project Title: %s" % (
    label_studio_project.id, label_studio_project.title))

# export the labeled tasks
_logger.debug("Exporting labeled tasks...")
exported_tasks = label_studio_project.export_tasks()
_logger.debug("Exported %i tasks" % len(exported_tasks))

# use our processor to convert into pandas dlc format
if len(exported_tasks) > 0:
    # save to pickle for resuming projects
    _logger.debug("Saving tasks to pickle file")
    pickle.dump(
        exported_tasks,
        open(os.path.join(args.proj_dir, LABELSTUDIO_TASKS_FILENAME), "wb"),
    )
    # save to csv for lightning pose models
    _logger.debug("Saving annotations to csv file")
    processor = LabelStudioJSONProcessor(
        label_studio_json_export=exported_tasks,
        data_dir=args.proj_dir,
        relative_image_dir="",
        keypoint_names=args.keypoints_list.split("/"),
    )
    df = processor()
    df.to_csv(os.path.join(args.proj_dir, COLLECTED_DATA_FILENAME))

# update metadata so app has access to labeling project info
metadata_file = os.path.join(args.proj_dir, LABELSTUDIO_METADATA_FILENAME)
proj_details = yaml.safe_load(open(metadata_file, "r"))
proj_details["n_labeled_tasks"] = len(exported_tasks)
proj_details["n_total_tasks"] = len(label_studio_project.get_tasks())
yaml.safe_dump(proj_details, open(metadata_file, "w"))
