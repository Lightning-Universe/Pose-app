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

If you are planning to run the app on a local computer or from a Lightning Studio, you 
additionally need to install LabelStudio. From `~/Pose-app`, run
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

We provide three different apps:
* `demo_app.py`: using provided example data, you can train and evaluate models
* `labeling_app.py`: stand-alone labeling app where you can upload videos, extract frames, and label using LabelStudio
* `app.py`: full app that includes labeling, training, and evaluation

### Running locally
Run any of the above three apps from the command line:
```bash
cd ~/Pose-app
lightning run app <app_name.py>
```

If you need to increase the file size limit for uploading videos, set the following environment
variable (in units of MB, so 500 is equivalent to 500MB). We recommend using videos less than 500MB
for best performance.
```
lightning run app <app_name.py> --env STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
```

### Running on cloud
Running the app on the cloud is easy!
```bash
lightning run app <app_name.py> --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

