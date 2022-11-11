"""Create a label studio project and add a data source and annotation tasks."""

import os
import argparse
from lai_components.label_studio.utils import (
    connect_to_label_studio, start_project, create_data_source, get_rel_image_paths_from_idx_files)

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--project_name', type=str)
parser.add_argument('--label_config', type=str)
args = parser.parse_args()

basedir = os.path.relpath(args.data_dir, os.getcwd())

print("Executing build_labeling_task.py")

# read label config text file into str
with open(args.label_config, 'r') as f:
    label_config = f.read()

print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

print("Creating LabelStudio project...")
label_studio_project = start_project(
    label_studio_client=label_studio_client,
    title=args.project_name,
    label_config=label_config
)
print("LabelStudio project created.")
print("Project ID: %s" % label_studio_project.id)

# there are other potential args to json, but these are the only ones that are required
json = {
    "path": args.data_dir,
    "project": label_studio_project.id
}

print("Creating LabelStudio data source...")
create_data_source(label_studio_project=label_studio_project, json=json)
print("LabelStudio data source created.")

print("Importing tasks...")
rel_images = get_rel_image_paths_from_idx_files(args.data_dir)
print(rel_images)
label_studio_prefix = f"data/local-files?d={basedir}/"
# loop over the png files in the directory and add them as dicts to the lisr, using labelstudio
# path format
image_list = [{"img": os.path.join(label_studio_prefix, rel_img)} for rel_img in rel_images]
label_studio_project.import_tasks(image_list)
print("%i Tasks imported." % len(image_list))
