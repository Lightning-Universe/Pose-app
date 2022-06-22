# test hydra outdir
#   python hydra_out_dir.py hydra.run.dir="alt/me"
# does not change the output dir
#   python hydra_out_dir.py +hydra.output_subdir="alt/me"
#   python hydra_out_dir.py +hydra.run.out="alt/me"

import hydra
import os
from omegaconf import DictConfig

@hydra.main()
def my_app(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

if __name__ == "__main__":
    my_app()

"""
$ python my_app.py
Working directory : /home/omry/dev/hydra/outputs/2019-09-25/15-16-17

$ python my_app.py
Working directory : /home/omry/dev/hydra/outputs/2019-09-25/15-16-19
"""