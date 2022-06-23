# Lightning Pose App

App for:
* Annotating keypoints on images
* Training a model to predict keypoints (after configuring it)
* Predicting keypoints on images and videos
* Looking at diagnostics via Tensorboard and FiftyOne
* More to come! (Deploying for new videos, active learning, etc.)

## Prerequsites

For now, the installation assumes 
- Grid Session GPU instance 
- Microsoft Visual Studio Code on laptop
- local editable installation of `lightning-pose`
- Use `python -m pip`, which is the best practice when using virtual env like `conda`.
- DO NOT USE `pip`.  Some modules may not install correctly.

# From laptop

open a terminal

## create grid session 

```
grid session create --instance_type g4dn.xlarge 
```

make sure session ssh is setup for VSC
```
grid session ssh GRID_SESSION_NAME "exit"
```

open VSC

- From the VSC, connect to GRID_SESSION_NAME

# Inside GRID_SESSION_NAME from VSC

## create conda env inside

- create lai
```bash
cd ~
conda create --yes --name lai python=3.8
conda activate lai
# mandatory step to pull the dependencies from extra-index-url
python -m pip install lightning
```

- record versions and git hash
```
git rev-parse HEAD
lightning --version
python --version
```

### Download lightning-pose-app and put lightning-pose inside it

NOTE: requirements.txt has lightning-pose requirements.  this allows the app to run in the cloud.

```bash
cd ~
git clone https://github.com/PyTorchLightning/lightning-pose-app
cd lightning-pose-app
git checkout rslee-prototype

git clone https://github.com/danbider/lightning-pose lightning_pose
# TODO: torch and numpy are in requirements.txt, but pip cannt find it. so install first before the rest
python -m pip install -r requirements.txt -e lightning-pose/.
```

NOTE: 

Ignore the following error for now.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
aiobotocore 2.1.2 requires botocore<1.23.25,>=1.23.24, but you have botocore 1.26.5 which is incompatible.
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
