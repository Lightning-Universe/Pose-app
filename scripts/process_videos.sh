#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 model_dir video_dir compute_metrics label_video_full label_video_snippet"
    exit 1
fi

# Assign arguments to variables
model_dir=$1            # absolute path to model directory (usually YY-MM-DD/HH-MM-SS format)
video_dir=$2            # absolute path to directory of videos to be processed
compute_metrics=$3      # bool, true to compute metrics on each video
label_video_full=$4     # bool, true to create full video overlaid w/ predictions
label_video_snippet=$5  # bool, true to create 30 sec video snippet overlaid w/ predictions

# Check if the provided directory exists
if [ ! -d "$video_dir" ]; then
    echo "Directory $video_dir not found"
    exit 1
fi

# Loop through all mp4 and avi files in the directory
for video_file in "$video_dir"/*.mp4 "$video_dir"/*.avi; do
    if [ -f "$video_file" ]; then
        echo "Processing file: $video_file"
        python scripts/process_videos.py --model_dir "$model_dir" --video_file "$video_file" --compute_metrics "$compute_metrics" --label_video_full "$label_video_full" --label_video_snippet "$label_video_snippet"
    fi
done
