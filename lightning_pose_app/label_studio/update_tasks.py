"""Update tasks after new video frames have been extracted."""

import argparse
import os
import pandas as pd
import yaml

from lightning_pose_app import LABELSTUDIO_METADATA_FILENAME, COLLECTED_DATA_FILENAME
from lightning_pose_app.label_studio.utils import connect_to_label_studio
from lightning_pose_app.label_studio.utils import get_project
from lightning_pose_app.label_studio.utils import get_rel_image_paths_from_idx_files


def get_annotation(
    rel_path: str, labels: pd.DataFrame, dims: dict, task_id: int, project_id: int,
) -> dict:
    task_dict = {
        "completed_by": 1,
        "result": [],
        "was_cancelled": False,  # user skipped the task
        "ground_truth": False,
        "lead_time": 1,  # how much time it took to annotate task
        "last_action": "imported",
        "task": task_id,
        "project": project_id,
        "updated_by": 1,  # last user who updated this annotation
        "parent_prediction": None,
        "parent_annotation": None,
        "last_created_by": None,
    }
    scorer = labels.scorer[0]
    keypoints = labels.bodyparts.unique()
    for keypoint in keypoints:
        idx_x = (labels.scorer == scorer) & (labels.bodyparts == keypoint) & (labels.coords == "x")
        idx_y = (labels.scorer == scorer) & (labels.bodyparts == keypoint) & (labels.coords == "y")
        x_val = labels[idx_x][rel_path].iloc[0]
        y_val = labels[idx_y][rel_path].iloc[0]
        isnan = labels[idx_x][rel_path].isna().iloc[0]
        if not isnan:
            kp_dict = {
                "value": {
                    "x": x_val,
                    "y": y_val,
                    "width": 0.5,
                    "keypointlabels": [keypoint],
                },
                "original_width": dims["width"],
                "original_height": dims["height"],
                "image_rotation": 0,
                "from_name": "kp-1",
                "to_name": "img-1",
                "type": "keypointlabels",
            }
            task_dict["result"].append(kp_dict)
    return task_dict


print("Executing update_tasks.py")

parser = argparse.ArgumentParser()
parser.add_argument("--label_studio_url", type=str)
parser.add_argument("--proj_dir", type=str)
parser.add_argument("--api_key", type=str)
parser.add_argument("--config_file", type=str, default="")
parser.add_argument("--update_from_csv", action="store_true", default=False)
args = parser.parse_args()

print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

# get current project
metadata_file = os.path.join(args.proj_dir, LABELSTUDIO_METADATA_FILENAME)
try:
    metadata = yaml.safe_load(open(metadata_file, "r"))
except FileNotFoundError:
    print(f"Cannot find {metadata_file} in {args.proj_dir}")
    exit()
label_studio_project = get_project(label_studio_client=label_studio_client, id=metadata["id"])
print("Fetched Project ID: %s, Project Title: %s" % (
    label_studio_project.id, label_studio_project.title))

# get tasks that already exist
existing_tasks = label_studio_project.get_tasks()
if len(existing_tasks) > 0:
    existing_imgs = [t["data"]["img"] for t in existing_tasks]
else:
    existing_imgs = []

print("Importing tasks...")
basedir = os.path.relpath(args.proj_dir, os.getcwd())
rel_images = get_rel_image_paths_from_idx_files(args.proj_dir)
print("relative image paths: {}".format(rel_images))
label_studio_prefix = f"data/local-files?d={basedir}/"
# loop over files and add them as dicts to the list, using label studio path format
# ignore files that are already registered as tasks
image_list = []
for r, rel_img in enumerate(rel_images):
    ls_img_path = os.path.join(label_studio_prefix, rel_img)
    if ls_img_path not in existing_imgs:
        image_list.append({"img": ls_img_path})

label_studio_project.import_tasks(image_list)
print("%i Tasks imported." % len(image_list))

# add annotations to tasks when importing from another project (e.g. previous DLC project)
if args.update_from_csv:

    csv_file = os.path.join(args.proj_dir, COLLECTED_DATA_FILENAME)
    df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    config = yaml.safe_load(open(args.config_file, "r"))

    tasks = label_studio_project.get_tasks()
    for task in tasks:
        if len(task["annotations"]) == 0:
            task_id = task["id"]
            rel_img_idx = task["data"]["img"].find("labeled-data")
            rel_img = task["data"]["img"][rel_img_idx:]
            annotation = get_annotation(
                rel_path=rel_img,
                labels=df.loc[rel_img].to_frame().reset_index(),
                dims=config["data"]["image_orig_dims"],
                task_id=task_id,
                project_id=label_studio_project.id,
            )
            label_studio_project.create_annotation(task_id=task_id, **annotation)

    # update metadata so app has access to labeling project info
    metadata_file = os.path.join(args.proj_dir, LABELSTUDIO_METADATA_FILENAME)
    proj_details = yaml.safe_load(open(metadata_file, "r"))
    proj_details["n_labeled_tasks"] = len(image_list)
    proj_details["n_total_tasks"] = len(label_studio_project.get_tasks())
    yaml.safe_dump(proj_details, open(metadata_file, "w"))
