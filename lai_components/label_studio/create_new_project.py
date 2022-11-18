"""Create a label studio project."""

import argparse
import datetime
import os
import yaml

from lai_components.label_studio.utils import connect_to_label_studio
from lai_components.label_studio.utils import start_project
from lai_components.label_studio.utils import create_data_source

print("Executing create_new_project.py")

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--proj_dir', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--project_name', type=str)
parser.add_argument('--label_config', type=str)
args = parser.parse_args()

print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

print("Creating LabelStudio project...")
try:
    with open(args.label_config, 'r') as f:
        label_config = f.read()
except FileNotFoundError:
    print(f"Cannot find label studio labeling config at {args.label_config}")
    exit()
label_studio_project = start_project(
    label_studio_client=label_studio_client,
    title=args.project_name,
    label_config=label_config
)
print("LabelStudio project created.")

# save out project details
proj_details = {
    "project_name": args.project_name,
    "id": label_studio_project.id,
    "created_at": str(datetime.datetime.now()),
    "api_key": args.api_key,
    "n_labeled_tasks": 0,
    "n_total_tasks": 0,
}
metadata_file = os.path.join(args.proj_dir, "label_studio_metadata.yaml")
if not os.path.exists(args.proj_dir):
    os.makedirs(args.proj_dir)
yaml.safe_dump(proj_details, open(metadata_file, "w"))

# connect label studio to local data source
print("Creating LabelStudio data source...")
json = {  # there are other args to json, but these are the only ones that are required
    "path": args.proj_dir,
    "project": label_studio_project.id
}
create_data_source(label_studio_project=label_studio_project, json=json)
print("LabelStudio data source created.")
