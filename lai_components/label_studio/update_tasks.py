"""Create a label studio project."""

import argparse
import os
import yaml

from lai_components.label_studio.utils import connect_to_label_studio
from lai_components.label_studio.utils import get_project
from lai_components.label_studio.utils import get_rel_image_paths_from_idx_files

print("Executing update_tasks.py")

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--proj_dir', type=str)
parser.add_argument('--api_key', type=str)
args = parser.parse_args()

print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

# get current project
metadata_file = os.path.join(args.proj_dir, "label_studio_metadata.yaml")
try:
    metadata = yaml.safe_load(open(metadata_file, "r"))
except FileNotFoundError:
    print(f"Cannot find label studio files in {args.proj_dir}")
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
for rel_img in rel_images:
    ls_img_path = os.path.join(label_studio_prefix, rel_img)
    if ls_img_path not in existing_imgs:
        image_list.append({"img": ls_img_path})
label_studio_project.import_tasks(image_list)
print("%i Tasks imported." % len(image_list))
