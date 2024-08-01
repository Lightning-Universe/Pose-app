#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 model_dir_list video_dir compute_metrics label_video_full label_video_snippet compute_eks eks_save_dir"
    exit 1
fi

# Assign arguments to variables
model_dir_list=$1       # list of model directories separated by colons (:)
video_dir=$2            # absolute path to directory of videos to be processed
compute_metrics=$3      # bool, true to compute metrics on each video
label_video_full=$4     # bool, true to create full video overlaid w/ predictions
label_video_snippet=$5  # bool, true to create 30 sec video snippet overlaid w/ predictions
compute_eks=$6          # bool, true to run EKS on the outputs of the listed models
eks_save_dir=$7         # absolute path to directory to save eks outputs

# Check if the provided video directory exists
if [ ! -d "$video_dir" ]; then
    echo "Directory $video_dir not found"
    exit 1
fi

# Split the model directory list into an array
IFS=':' read -r -a model_dirs <<< "$model_dir_list"

# Loop through each model directory and call the process_videos.sh script
for model_dir in "${model_dirs[@]}"; do
    if [ ! -d "$model_dir" ]; then
        echo "Model directory $model_dir not found"
        exit 1
    fi
    echo "Processing videos with model: $model_dir"
    bash scripts/process_videos.sh "$model_dir" "$video_dir" "$compute_metrics" "$label_video_full" "$label_video_snippet"
done

# If compute_eks flag is true, loop through each video file and call the run_eks.py script
if [ "$compute_eks" = "True" ]; then
    echo "Running Ensemble Kalman Smoother (EKS) on each video"
    for video_file in "$video_dir"/*.mp4 "$video_dir"/*.avi; do
        if [ -f "$video_file" ]; then
            echo "Processing file: $video_file with EKS"
            python scripts/run_eks.py --model_dir_list "$model_dir_list" --video_file "$video_file" --compute_metrics "$compute_metrics" --label_video_full "$label_video_full" --label_video_snippet "$label_video_snippet" --eks_save_dir "$eks_save_dir"
        fi
    done
fi
