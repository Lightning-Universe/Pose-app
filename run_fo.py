import fiftyone as fo
import fiftyone.zoo as foz
import fire 

def run(dataset_name:str = "quickstart",address=None, port=None):
  dataset = foz.load_zoo_dataset(dataset_name)
  session = fo.launch_app(dataset, address=address, port=port)
  session.wait()

if __name__ == '__main__':
  fire.Fire(run)

