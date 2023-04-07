# Lightning Pose App

Lightning App for:
* Annotating keypoints on images
* Training a pose estimation model to predict keypoints
* Predicting keypoints on images and videos
* Looking at diagnostics via Tensorboard and FiftyOne
* More to come! (Deploying for new videos, active learning, etc.)

The following instructions detail how to install the app locally for development purposes.

## Create grid session 
Note: if you are developing from a local, non-Grid machine, skip this section and go to 
`Environment setup`.

From a laptop, create a new Grid session:
```bash
$ grid session create --instance_type g4dn.xlarge 
```

Make sure session ssh is setup for VSC:
```
$ grid session ssh GRID_SESSION_NAME "exit"
```

Open VSC, and connect to GRID_SESSION_NAME. Complete the following steps within 
GRID_SESSION_NAME from VSC.

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

Record versions and git hash:
```bash
(lai) $ lightning --version
(lai) $ python --version
```

## Run the app on the cloud
Once the environment has been set up, running the app on the cloud is easy! Launch with the
following command:
```bash
(lai) $ lightning run app app.py --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

## Run the app locally
Running the app locally requires a bit of extra work, since we'll need to set up additional 
environments in order to mirror what happens on the cloud when machines are requisitioned;
this involves setting up various virtual environments.

#### Tensorflow
Install:
```bash
(lai) $ virtualenv ~/venv-tensorboard 
(lai) $ source ~/venv-tensorboard/bin/activate; which python; python -m pip install tensorflow tensorboard; deactivate
```
Test:
```bash
(lai) $ source ~/venv-tensorboard/bin/activate; tensorboard --logdir .; deactivate
```

#### LabelStudio
Install:
```bash
(lai) $ virtualenv ~/venv-label-studio 
(lai) $ source ~/venv-label-studio/bin/activate; which python; python -m pip install label-studio label-studio-sdk; deactivate
(lai) $ source ~/venv-label-studio/bin/activate; python -m pip install -e .; deactivate

```
Test:
```bash
(lai) $ source ~/venv-label-studio/bin/activate; label-studio version; deactivate
```

#### Run the app!

In order to run the application locally, run the following commands:

```bash
(lai) $ cd ~/Pose-app
(lai) $ lightning run app app.py
```
