# Lightning Pose App

This is an _experimental_ version of our Lightning App for:

- Annotating keypoints on images
- Training a pose estimation model to predict keypoints
- Predicting keypoints on images and videos
- Looking at diagnostics via Tensorboard and FiftyOne
- More to come! (Deploying for new videos, active learning, etc.)

Our app of reference is `demo_app.py`, which is a demo of the above functionality.

The following instructions detail how to install the app locally for development purposes.

## Environment setup

First, create a conda environment:

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
cd lightning-pose
pip install -e .
```

## Run the app on the cloud

Once the environment has been set up, running the app on the cloud is easy! Launch with the
following command:

```bash
lightning run app demo_app.py --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

## Run the app locally

Running the app locally requires a few additional installs.

First, install the `lightning-pose` repo in editable mode (the code has already been downloaded):

```bash
cd ~/Pose-app/lightning-pose
pip install -e .
```

Install LabelStudio:
from `~/Pose-app`, run

```bash
virtualenv ~/venv-label-studio
source ~/venv-label-studio/bin/activate; sudo apt-get install libpq-dev; deactivate
source ~/venv-label-studio/bin/activate; conda install libffi==3.3; deactivate
source ~/venv-label-studio/bin/activate; pip install label-studio label-studio-sdk; deactivate
source ~/venv-label-studio/bin/activate; pip install -e .; deactivate
```

(note that `libffi` install may downgrade python, but this is fine.)

Test

```bash
source ~/venv-label-studio/bin/activate; label-studio version; deactivate
```

In order to run the application locally, run the following commands:

```bash
cd ~/Pose-app
lightning run app demo_app.py
```
