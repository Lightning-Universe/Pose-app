import argparse
import os
import pickle
import yaml

from lai_components.label_studio.utils import connect_to_label_studio
from lai_components.label_studio.utils import get_project
from lai_components.label_studio.utils import LabelStudioJSONProcessor

print("Executing check_labeling_task_and_export.py")

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--proj_dir', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--keypoints_list', type=str)  # assume keypoint names separates by colons
args = parser.parse_args()

# connect to label studio
print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

# get current project
metadata_file = os.path.join(args.proj_dir, "label_studio_metadata.yaml")
try:
    metadata = yaml.safe_load(open(metadata_file, "r"))
except FileNotFoundError:
    print(f"Cannot find label studio files in {args.proj_dir}")
label_studio_project = get_project(label_studio_client=label_studio_client, id=metadata["id"])
print("Fetched Project ID: %s, Project Title: %s" % (
    label_studio_project.id, label_studio_project.title))

# export the labeled tasks
print("Exporting labeled tasks...")
exported_tasks = label_studio_project.export_tasks()
print("Exported %i tasks" % len(exported_tasks))

# use our processor to convert into pandas dlc format
if len(exported_tasks) > 0:
    processor = LabelStudioJSONProcessor(
        label_studio_json_export=exported_tasks,
        data_dir=args.proj_dir,
        relative_image_dir="",
        keypoint_names=args.keypoints_list.split(";"),
    )
    df = processor()
    # print(df)
    # save to csv for lightning pose models
    # print("Saving to csv file")
    df.to_csv(os.path.join(args.proj_dir, "CollectedData.csv"))
    # save to pickle for resuming projects
    print("Saving tasks to pickle file")
    pickle.dump(exported_tasks, open(os.path.join(args.proj_dir, "label_studio_tasks.pkl"), "wb"))

# update metadata so app has access to labeling project info
metadata_file = os.path.join(args.proj_dir, "label_studio_metadata.yaml")
proj_details = yaml.safe_load(open(metadata_file, "r"))
proj_details["n_labeled_tasks"] = len(exported_tasks)
proj_details["n_total_tasks"] = len(label_studio_project.get_tasks())
yaml.safe_dump(proj_details, open(metadata_file, "w"))
