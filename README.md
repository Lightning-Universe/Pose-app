# Lightning Pose App

App for:
* Annotating keypoints on images
* Training a model to predict keypoints (after configuring it)
* Predicting keypoints on images and videos
* Looking at diagnostics via Tensorboard and FiftyOne
* More to come! (Deploying for new videos, active learning, etc.)

## Installation
For now, the installation assumes a local editable installation of `lightning` and `lightning-pose` 

### Conda environment

Create a `conda` environment and `cd` into it:
```bash
conda create --yes --name lit-app python=3.8
conda activate lit-app
```

### Install `lightning` (beta)
Following the instructions here:

```bash
git clone https://github.com/PyTorchLightning/lightning
cd lightning
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_frontend.py
```

- check for lightning version of 0.0.45

```
lightning --version
```

### Download lightning-pose-app
NOTE: requirements.txt has lightning-pose requirements.  this allows the app to run in the cloud.
```bash
git clone https://github.com/PyTorchLightning/lightning-pose-app
cd lightning-pose-app
git checkout rslee-prototype
# TODO: torch and numpy are in requirements.txt, but pip cannt find it. so install first before the rest
python -m pip install torch numpy
python -m pip install -r requirements.txt
```

NOTE: 

Ignore the following error for now.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
aiobotocore 2.1.2 requires botocore<1.23.25,>=1.23.24, but you have botocore 1.26.5 which is incompatible.
```


### Install `lightning-pose`
NOTE:  The requirements are copied into lightning-pose-app.  no need to redo the pip install

```bash
git clone https://github.com/danbider/lightning-pose
cd lightning-pose
python -m pip install -e .
```

### Locally

In order to run the application locally, run the following commands

```bash
cd lightning-pose-app
lightning run app app.py
```

The following can be resolved with `rm -rf ~/.fiftyone`

```
{"t":{"$date":"2022-05-23T14:42:45.150Z"},"s":"I",  "c":"CONTROL",  "id":20697,   "ctx":"main","msg":"Renamed existing log file","attr":{"oldLogPath":"/Users/robertlee/.fiftyone/var/lib/mongo/log/mongo.log","newLogPath":"/Users/robertlee/.fiftyone/var/lib/mongo/log/mongo.log.2022-05-23T14-42-45"}}
Subprocess ['/opt/miniconda3/envs/lai/lib/python3.8/site-packages/fiftyone/db/bin/mongod', '--dbpath', '/Users/robertlee/.fiftyone/var/lib/mongo', '--logpath', '/Users/robertlee/.fiftyone/var/lib/mongo/log/mongo.log', '--port', '0', '--nounixsocket'] exited with error 100:
```


### On GPU
```
USE_GPU=1 lightning run app app.py --cloud --name lightning-pose
```
