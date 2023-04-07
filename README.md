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
$ cd ~
$ conda create --yes --name lai python=3.9
$ conda activate lai
```

Next, install the `Pose-app` repo:
```bash
(lai) $ cd ~
(lai) $ git clone https://github.com/Lightning-Universe/Pose-app
(lai) $ cd Pose-app
(lai) $ python -m pip install -r requirements.txt
(lai) $ python -m pip install -e .
```
(Note the use of `python -m pip` rather than just `pip`, this is important!)

Finally, install the `lightning-pose` repo _inside_ the `Pose-app` repo:
```bash
(lai) $ cd ~/Pose-app
(lai) $ git clone https://github.com/danbider/lightning-pose
(lai) $ cd lightning-pose
(lai) $ python -m pip install -r requirements.txt
```

## Run the app on the cloud
Once the environment has been set up, running the app on the cloud is easy! Launch with the
following command:
```bash
(lai) $ lightning run app app.py --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

## Run the app locally
Running the app locally requires a bit of extra work, since we'll need to install some additional
packages and set up a virtual environement in order to mirror what happens on the cloud when 
machines are requisitioned.

Install tensorboard:
```bash
(lai) $ python -m pip install tensorboard
```

Install LabelStudio virtual environment:
```bash
(lai) $ virtualenv ~/venv-label-studio 
(lai) $ source ~/venv-label-studio/bin/activate; which python; python -m pip install label-studio label-studio-sdk; deactivate
(lai) $ source ~/venv-label-studio/bin/activate; python -m pip install -e .; deactivate

```
Test:
```bash
(lai) $ source ~/venv-label-studio/bin/activate; label-studio version; deactivate
```

In order to run the application locally, run the following commands:

```bash
(lai) $ cd ~/Pose-app
(lai) $ lightning run app app.py
```
