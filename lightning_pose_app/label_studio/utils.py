"""Utility functions for label studio"""

from label_studio_sdk import Client
import logging
import numpy as np
import os
import pandas as pd
import time
from typing import Any, Tuple, List, Dict

from lightning_pose_app import LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME


_logger = logging.getLogger('APP.LABELSTUDIO')

MAX_CONNECT_ATTEMPTS = 30


# define a decorator that retries to run a function MAX_CONNECT_ATTEMPTS times
def retry(func):
    """Retry calling the decorated function MAX_CONNECT_ATTEMPTS times"""
    def wrapper(*args, **kwargs):
        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except:
                _logger.debug(f"Could not execute {func.__name__}, retrying in one second...")
                attempts += 1
                time.sleep(1)
                if attempts > MAX_CONNECT_ATTEMPTS:
                    raise Exception("Could not execute {} after {} attempts".format(
                        func.__name__, MAX_CONNECT_ATTEMPTS))
    return wrapper


# -------------------------
# LabelStudio API wrappers
#--------------------------

@retry
def connect_to_label_studio(url, api_key):
    # Connect to Label Studio
    return Client(url=url, api_key=api_key)


@retry
def start_project(label_studio_client, title, label_config):
    project = label_studio_client.start_project(
        title=title,
        label_config=label_config)
    return project


@retry
def get_project(label_studio_client, id):
    project = label_studio_client.get_project(id=id)
    return project


@retry
def create_data_source(label_studio_project, json):
    label_studio_project.make_request('POST', '/api/storages/localfiles', json=json)


# -------------------------
# data handling utils
#--------------------------

class LabelStudioJSONProcessor:
    """ methods to process the .json output from labelstudio """

    def __init__(self, label_studio_json_export: List[dict], data_dir: str,
                 relative_image_dir: str, keypoint_names: List[str]):
        self.label_studio_json_export = label_studio_json_export
        self.data_dir = data_dir
        self.relative_image_dir = relative_image_dir
        self.keypoint_names = keypoint_names

    @property
    def absolute_image_dir(self) -> str:
        return os.path.join(self.data_dir, self.relative_image_dir)

    def get_relative_image_paths(self) -> List[str]:
        """Paths within data_dir"""
        relative_image_paths = [self.get_relative_image_path(annotated_example) for
                                annotated_example in self.label_studio_json_export]
        return relative_image_paths

    def get_relative_image_path(self, annotated_example) -> str:
        """Paths within data_dir"""
        path = os.path.join(
            self.relative_image_dir, annotated_example["data"]["img"].split("=")[-1])
        relative_image_path = path[path.find(LABELED_DATA_DIR):]
        return relative_image_path

    def get_absolute_image_paths(self) -> List[str]:
        """Absolute paths on the machine"""
        abs_image_paths = [
            os.path.join(self.absolute_image_dir, annotated_example["data"]["img"].split("=")[-1])
            for annotated_example in self.label_studio_json_export]
        return abs_image_paths

    def get_keypoint_names(self) -> List[str]:
        return self.keypoint_names

    @staticmethod
    def make_dlc_pandas_index(model_name: str, keypoint_names: List[str]) -> pd.MultiIndex:
        pdindex = pd.MultiIndex.from_product(
            [["%s_tracker" % model_name], keypoint_names, ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        )
        return pdindex

    def build_zeros_dataframe(self) -> pd.DataFrame:
        """Build a dataframe with the keypoint names as columns and the image paths as the index"""
        keypoint_names = self.get_keypoint_names()
        image_paths = self.get_relative_image_paths()
        # build the hierarchical index
        pdindex = self.make_dlc_pandas_index("lightning", keypoint_names)
        df = pd.DataFrame(np.nan * np.zeros((len(image_paths), int(len(keypoint_names) * 2))),
                          columns=pdindex, index=image_paths)
        df.sort_index(inplace=True)
        return df

    # have one method per a single keypoint, one per frame, and one across frames.
    @staticmethod
    def get_pixel_coordinates_per_image(result: Dict[str, Any]) -> Tuple[float, float]:
        """Convert result dict of an image to a dict with img name, keypoint name, x,y vals."""
        if 'original_width' not in result or 'original_height' not in result:
            # we need these to resize images
            return None

        # includes the keypoint name and x,y values, relative to the image size
        value: Dict[str, Any] = result['value']
        width, height = result['original_width'], result['original_height']

        if all([key in value for key in ['x', 'y']]):  # if both x and y are in the dict
            return width * value['x'] / 100.0, \
                   height * value['y'] / 100.0

    def __call__(self) -> pd.DataFrame:
        """Build a dataframe with the keypoint names as columns and the image paths as the index"""
        df = self.build_zeros_dataframe()
        # fill the dataframe with the keypoint coordinates
        for i, example in enumerate(self.label_studio_json_export):  # loop over the images
            relative_image_path = self.get_relative_image_path(example)
            annotation = example["annotations"][
                0]  # assume there is only one annotator per kp and image
            for result in annotation["result"]:  # loop over the keypoints
                pixel_coordinates = self.get_pixel_coordinates_per_image(result)
                if pixel_coordinates is None:
                    continue
                x, y = pixel_coordinates
                keypoint_name = result['value']['keypointlabels'][0]
                # note, if we have multiple annotations of the same keypoint and image, the last
                # one overrides the previous ones in the dataframe.
                df.loc[relative_image_path, ("lightning_tracker", keypoint_name, "x")] = x
                df.loc[relative_image_path, ("lightning_tracker", keypoint_name, "y")] = y
        return df


# a function that gets relative image paths from csv files inside of a base dir
def get_rel_image_paths_from_idx_files(basedir: str) -> List[str]:
    img_list = []
    for root, dirs, files in os.walk(basedir):
        if LABELED_DATA_DIR not in root:
            # make sure we only look in the labeled data directory
            # if we do not do this we risk uploading info from temp dirs too
            continue
        for file in files:
            if file == SELECTED_FRAMES_FILENAME:
                abspath = os.path.join(root, file)
                img_files = np.genfromtxt(abspath, delimiter=',', dtype=str)
                for img_file in img_files:
                    img_list.append(os.path.relpath(os.path.join(root, img_file), start=basedir))
    return img_list


def build_xml(bodypart_names: list) -> str:
    """Builds the XML file for Label Studio"""
    # 25 colors
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gray",
            "black", "white", "cyan", "magenta", "lime", "maroon", "olive", "navy", "teal",
            "aqua", "fuchsia", "silver", "gold", "indigo", "violet", "coral"]
    # replicate just to be safe
    colors = colors + colors + colors + colors
    colors_to_use = colors[:len(bodypart_names)]  # practically ignoring colors
    view_str = "<!--Basic keypoint image labeling configuration for multiple regions-->"
    view_str += "\n<View>"
    view_str += "\n<Header value=\"Select keypoint name with the cursor/number button, " \
                "then click on the image.\"/>"
    view_str += "\n<Text name=\"text1\" value=\"Important: Click Submit after you have labeled " \
                "all visible keypoints in this image.\"/>"
    view_str += "\n<Text name=\"text2\" value=\"Also useful: Press H for hand tool, " \
                "CTRL+ to zoom in and CTRL- to zoom out\"/>"
    view_str += "\n  <KeyPointLabels name=\"kp-1\" toName=\"img-1\" strokeWidth=\"3\">"  # indent 2
    for keypoint, color in zip(bodypart_names, colors_to_use):
        view_str += f"\n    <Label value=\"{keypoint}\" />"  # indent 4
    view_str += "\n  </KeyPointLabels>"  # indent 2
    view_str += "\n  <Image name=\"img-1\" value=\"$img\" />"  # indent 2
    view_str += "\n</View>"  # indent 0
    return view_str


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
                    "x": x_val / dims["width"] * 100.0,  # percentage of image width
                    "y": y_val / dims["height"] * 100.0,  # percentage of image height
                    "width": 0.5,  # point size by percentage of image size
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
