# Command line scripts

These scripts are provided to accelerate inference using the command line instead of the app.

To run any of these scripts, first move into the `Pose-app` directory:

```shell
cd Pose-app
```

## process_videos.sh

This script processes all videos in a given directory using a specific model.

Run using `bash`:

```shell
bash scripts/process_videos.sh /path/to/model_dir /path/to/video_dir compute_metrics_flag label_video_full_flag label_video_snippet_flag
```

The positional arguments are as follows:
* `/path/to/model_dir`: absolute path to the top-level model directory 
(usually /path/to/YYYY-MM-DD/HH-MM-SS)
* `/path/to/video_dir`: absolute path to a directory containing mp4 or avi files.
The model will run inference on every video in the directory.
* `compute_metrics_flag`: if `True`, all available metrics will be computed on each video
(temporal norm, Pose PCA error, etc.)
* `label_video_full_flag`: if `True`, a separate mp4 file will be created that plots the inference
results on top of the original video, for each video. 
This is not recommended for very long videos.
* `label_video_snippet_flag`: if `True`, a separate mp4 file will be created that plots the 
inference results on top of a 30 second clip of the original video that contains the most 
movement in the predictions.

As a concrete example, if a model is stored in `/home/user/my_model`, the videos are stored in
`/home/user/data/my_videos`, you want to compute metrics, and label a video snippet, the command
would be

```shell
bash scripts/process_videos.sh /home/user/my_model /home/user/data/my_videos True False True
```

## process_videos_ensemble.sh

This script will process all videos in a given directory with multiple models, and then optionally
run the Ensemble Kalman Smoother (EKS) post-processor on the results.

Run using `bash`:

```shell
bash scripts/process_videos_ensemble.sh /path/to/model_dir_0:/path/to/model_dir_1:...:/path/to/model_dir_n /path/to/video_dir compute_metrics_flag label_video_full_flag label_video_snippet_flag compute_eks_flag /path/to/eks_save_dir
```

The positional arguments align with those of `process_videos.sh`, with some differences
* the first input should be a list of model directories separated by a colon (`:`), *not* separated
by spaces
* `compute_eks_flag`: if `True`, run EKS on the outputs of the listed models. 
The previous three flags for computing metrics and label videos will be applied to the EKS outputs
as well, where applicable.
* `/path/to/eks_save_dir`: absolute path to directory to save eks outputs
