import os
import argparse
from lai_components.label_studio.utils import connect_to_label_studio, LabelStudioJSONProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--keypoints_list', type=str)  # assume keypoint names separates by colons

args = parser.parse_args()

# print that we're executing this script
print("Executing check_labeling_task_and_export.py")

# connect to label studio
print("Connecting to LabelStudio at %s..." % args.label_studio_url)
label_studio_client = connect_to_label_studio(url=args.label_studio_url, api_key=args.api_key)
print("Connected to LabelStudio at %s" % args.label_studio_url)

# list all projects and get the most recent one
projects = label_studio_client.get_projects()
label_studio_project = projects[0]
print("Fetched Project ID: %s, Project Title: %s" % (label_studio_project.id, label_studio_project.title))

# export the labeled tasks
print("Exporting labeled tasks...")
exported_tasks = label_studio_project.export_tasks()
print("Exported %i tasks" % len(exported_tasks))

# use our processor to convert into pandas dlc format
if len(exported_tasks) > 0:
    print("Converting to pandas dlc format...")
    processor = LabelStudioJSONProcessor(
        label_studio_json_export=exported_tasks,
        data_dir=args.data_dir,
        relative_image_dir="",
        keypoint_names=args.keypoints_list.split(";"),
    )
    df = processor()

    print("Pandas dlc format:")
    print(df)

    # save to csv in data_dir
    print("Saving to csv...")
    df.to_csv(os.path.join(args.data_dir, "CollectedData.csv"))
