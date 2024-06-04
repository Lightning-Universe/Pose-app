"""Create a label studio project."""

import argparse
import datetime
import logging
import os

import yaml

from lightning_pose_app.label_studio.utils import (
    connect_to_label_studio,
    delete_project,
    get_project,
)

_logger = logging.getLogger('APP.LABELSTUDIO')

_logger.info("Executing delete_project.py")

parser = argparse.ArgumentParser()
parser.add_argument('--label_studio_url', type=str)
parser.add_argument('--proj_dir', type=str)
parser.add_argument('--api_key', type=str)
args = parser.parse_args()

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
_logger.debug("Fetched Project ID: %s, Project Title: %s" % (
    label_studio_project.id, label_studio_project.title))

# delete project from label studio database
_logger.info("Deleting LabelStudio project...")
status = delete_project(label_studio_client=label_studio_client, id=metadata["id"])
_logger.info(f"LabelStudio project deleted. {status}")
