"""Create a label studio configuration xml file."""

import argparse
import os


def build_xml(bodypart_names):
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


print("Executing create_labeling_config.py")

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
