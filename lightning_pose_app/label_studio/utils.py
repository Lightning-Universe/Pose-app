"""This module has to be called within the label-studio venv"""

import time
from label_studio_sdk import Client  # currently in its own venv, conflicting with lightning
import pandas as pd
import numpy as np
import os
from typing import Any, Tuple, List, Dict

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
                print("Could not execute {}, retrying in one second...".format(func.__name__))
                attempts += 1
                time.sleep(1)
                if attempts > MAX_CONNECT_ATTEMPTS:
                    raise Exception("Could not execute {} after {} attempts".format(
                        func.__name__, MAX_CONNECT_ATTEMPTS))
    return wrapper


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
        relative_image_path = path[path.find('labeled-data'):]
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

        value: Dict[str, Any] = result[
            'value']  # includes the keypoint name and x,y values, relative to the image size
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
        for file in files:
            if file.endswith(".csv"):
                abspath = os.path.join(root, file)
                img_files = np.genfromtxt(abspath, delimiter=',', dtype=str)
                for img_file in img_files:
                    img_list.append(os.path.relpath(os.path.join(root, img_file), start=basedir))
    return img_list
