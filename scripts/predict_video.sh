# add missing video predictions where missing
# cd lightning-pose
# bash ../scripts/predict_video.sh outputs

predict() {
if [ ! -f $dir/test_vid_*.csv ]; then
  dir=$1
  hydra_output_name=$(basename "$(dirname "$dir")")/$(basename "$dir")
  python scripts/predict_new_vids.py eval.hydra_paths=["$hydra_output_name"] \
  eval.test_videos_directory=$(pwd)/toy_datasets/toymouseRunningData/unlabeled_videos \
  eval.saved_vid_preds_dir="$(pwd)/outputs/$hydra_output_name/"
fi
}

for f in $(find $1 -name predictions.csv -print); do 
  dir=$(dirname $f)
  ls $dir
  predict $dir
done
