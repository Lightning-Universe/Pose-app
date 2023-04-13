# Lightning Pose App

Lightning App for:
* Annotating keypoints on images
* Training a pose estimation model to predict keypoints
* Predicting keypoints on images and videos
* Looking at diagnostics via Tensorboard and FiftyOne
* More to come! (Deploying for new videos, active learning, etc.)

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
git clone https://github.com/Lightning-Universe/Pose-app
cd Pose-app
pip install -e .
```

Finally, install the `lightning-pose` repo _inside_ the `Pose-app` repo:

```bash
cd ~/Pose-app
git clone https://github.com/danbider/lightning-pose
cd lightning-pose
pip install -e .
```

## Run the app on the cloud
Once the environment has been set up, running the app on the cloud is easy! Launch with the
following command:
```bash
lightning run app app.py --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

## Run the app locally
Running the app locally requires a bit of extra work, since we'll need to install some additional
packages and set up a virtual environement in order to mirror what happens on the cloud when 
machines are requisitioned.

Install LabelStudio:
<!-- ```bash
(lai) $ virtualenv ~/venv-label-studio 
(lai) $ source ~/venv-label-studio/bin/activate; which python; python -m pip install label-studio label-studio-sdk; deactivate
(lai) $ source ~/venv-label-studio/bin/activate; python -m pip install -e .; deactivate -->
<!-- ``` -->

```bash
sudo apt-get install libpq-dev
conda install libffi==3.3 # it may downgrade python, but this is fine
pip install label-studio label-studio-sdk
```
Test
```bash
label-studio version
```

<!-- ```bash
sudo apt-get install libpq-dev
(lai) $ virtualenv ~/venv-label-studio 
(lai) $ source ~/venv-label-studio/bin/activate; which python; python -m pip install label-studio label-studio-sdk; deactivate
(lai) $ source ~/venv-label-studio/bin/activate; python -m pip install -e .; deactivate
``` -->

<!-- Test:
```bash
(lai) $ source ~/venv-label-studio/bin/activate; label-studio version; deactivate
``` -->

In order to run the application locally, run the following commands:

```bash
cd ~/Pose-app
lightning run app app.py
```
