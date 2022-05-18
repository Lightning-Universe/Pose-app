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
cd lightning.beta
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_frontend.py
```

- check forli lightning version of 0.0.43

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



### Install `lightning-pose`
NOTE:  The requirements are copied into lightning-pose-app.  no need to redo the pip install

```bash
git clone https://github.com/danbider/lightning-pose
cd lightning-pose
```

### Locally

In order to run the application locally, run the following commands

```bash
cd lightning-pose-app
lightning run app app.py
```

### On GPU

```
USE_GPU=1 lightning run app app.py --cloud --name lightning-pose
```
