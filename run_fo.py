import fiftyone as fo
import fiftyone.zoo as foz
import fire

def run(dataset_name:str = "quickstart",address=None, port=None):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

python scripts/create_fiftyone_dataset.py \
eval.fiftyone.dataset_to_create="images" \
eval.fiftyone.dataset_name=$DATASET_NAME \
eval.fiftyone.build_speed="fast" \
eval.hydra_paths=["/home/jovyan/lightning-pose"] \
eval.fiftyone.model_display_names=["rslee_test"] \
eval.fiftyone.launch_app_from_script=True

if __name__ == '__main__':
  fire.Fire(run)

