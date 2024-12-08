"""Create a label studio configuration xml file."""

import argparse
import logging
import os

_logger = logging.getLogger('APP.LABELSTUDIO')


def build_xml(bodypart_names):
    """Builds the XML file for Label Studio"""
    # 25 unique colors
    colors = [
        "red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gray",
        "black", "white", "cyan", "magenta", "lime", "maroon", "olive", "navy", "teal",
        "aqua", "fuchsia", "silver", "gold", "indigo", "violet", "coral",
    ]
    # 34 unique hotkeys
    hotkeys = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
        "a", "s", "d", "f", "g", "j", "k", "l",
        "z", "x", "c", "b", "n", "m",
    ]
    # replicate just to be safe
    colors = colors + colors + colors + colors
    hotkeys = hotkeys + hotkeys
    colors_to_use = colors[:len(bodypart_names)]  # practically ignoring colors
    hotkeys_to_use = hotkeys[:len(bodypart_names)]  # note: if >34 kps, hotkeys overwrite
    view_str = "<!--Basic keypoint image labeling configuration for multiple regions-->"
    view_str += "\n<View>"
    view_str += "\n<Header value=\"Select keypoint name with the cursor/number button, " \
                "then click on the image.\"/>"
    view_str += "\n<Text name=\"text1\" value=\"Save annotations: " \
                "click Submit (or CTRL+ENTER)\"/>"
    view_str += "\n<Text name=\"text2\" value=\"Manipulate image: press H for hand tool, " \
                "\nCTRL+ to zoom in and CTRL- to zoom out\"/>"
    view_str += "\n<Text name=\"text3\" value=\"Next frame: SHIFT+DOWN, then SHIFT+RIGHT\"/>"
    view_str += "\n<Text name=\"text4\" value=\"To copy keypoints to another frame: hold CTRL " \
                "and select all keypoints; CTRL+c to copy; move to new frame; CTRL+v to paste\"/>"
    view_str += "\n  <KeyPointLabels name=\"kp-1\" toName=\"img-1\" strokeWidth=\"3\">"  # indent 2
    for keypoint, color, hotkey in zip(bodypart_names, colors_to_use, hotkeys_to_use):
        # indent 4
        view_str += f"\n    " \
                    f"<Label value=\"{keypoint}\" background=\"{color}\" hotkey=\"{hotkey}\" />"
    view_str += "\n  </KeyPointLabels>"  # indent 2
    view_str += "\n  <Image name=\"img-1\" value=\"$img\" />"  # indent 2
    view_str += "\n</View>"  # indent 0
    return view_str


_logger.info("Executing create_labeling_config.py")

parser = argparse.ArgumentParser()
parser.add_argument("--proj_dir", type=str)
parser.add_argument("--filename", type=str)
parser.add_argument("--keypoints_list", type=str)
args = parser.parse_args()

xml_str = build_xml(args.keypoints_list.split("/"))

config_file = os.path.join(args.proj_dir, args.filename)
if not os.path.exists(args.proj_dir):
    os.makedirs(args.proj_dir)
with open(config_file, 'wt') as outfile:
    outfile.write(xml_str)
