# Lightning Pose App

This repo contains browser-based GUIs that facilitate the deveopment of a pose estimation project.

We provide three different apps:
* `demo_app.py`: using provided example data, train and evaluate pose estimation models
* `labeling_app.py`: stand-alone labeling app where you can upload videos, extract frames, and annotate keypoints on extracted frames using LabelStudio
* `app.py`: [UNDER CONSTRUCTION!] full app that includes labeling, training, and evaluation

The following instructions detail how to install the app locally for development purposes.
Note these instructions are only for Linux machines; we will update the documentation for macOS 
shortly.

## Environment setup

First, check to see if you have ffmpeg installed:
```bash
ffmpeg -version
```
If not, install ffmpeg:
```bash
sudo apt install ffmpeg
```

Next, create a conda environment:

```bash
cd ~
conda create --yes --name lai python=3.8
conda activate lai
```

Next, install the `Pose-app` repo:

```bash
cd ~
git clone --recursive https://github.com/Lightning-Universe/Pose-app
cd Pose-app
pip install -e .
```

There are additional installation steps based on which apps you would like to run.

#### Demo app and full app
Install the `lightning-pose` repo:
```bash
cd ~/Pose-app/lightning-pose
pip install -e .
```

If you are using Ubuntu 22.04 or newer, you'll need an additional update for the Fiftyone package:
```bash
pip install fiftyone-db-ubuntu2204
```

#### Labling app and full app
Install `LabelStudio` in a virtual environment. From `~/Pose-app`, run
```bash
virtualenv ~/venv-label-studio
source ~/venv-label-studio/bin/activate; sudo apt-get install libpq-dev; deactivate
source ~/venv-label-studio/bin/activate; conda install libffi==3.3; deactivate
source ~/venv-label-studio/bin/activate; pip install -e .; deactivate
source ~/venv-label-studio/bin/activate; pip install label-studio label-studio-sdk; deactivate
```

(note that `libffi` install may downgrade python, but this is fine.)

Test:
```bash
source ~/venv-label-studio/bin/activate; label-studio version; deactivate
```

## Run the apps

#### Running locally
Run any of the above three apps from the command line:
```bash
cd ~/Pose-app
lightning run app <app_name.py>
```

If you need to increase the file size limit for uploading videos, set the following environment
variable (in units of MB). We recommend using videos less than 500MB for best performance.
```bash
lightning run app <app_name.py> --env STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
```

#### Running on cloud
Running the app on the cloud is easy!
```bash
lightning run app <app_name.py> --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```
