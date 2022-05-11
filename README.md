# Lightning Pose App

App for:
* Annotating keypoints on images
* Training a model to predict keypoints (after configuring it)
* Predicting keypoints on images and videos
* Looking at diagnostics via Tensorboard and FiftyOne
* More to come! (Deploying for new videos, active learning, etc.)

## Installation
For now, the installation assumes a local editable installation of `lightning` and `lightning-pose` (the latter skips the `DALI` installation).
### Conda environment
Create a `conda` environment and activate it
```bash
conda create --name lit-app python=3.8
conda activate lit-app
```

### Install `lightning-pose`
We clone lightning and install it with its dependencies in editable mode
```bash
git clone https://github.com/danbider/lightning-pose
cd lightning-pose
```
NOTE: we have two options for installing dependencies. On a remote instance with linux and CUDA 11, do the usual 
```bash
pip install -r requirements.txt
```
Which will install DALI. For local testing:
```bash
pip install -e .
```

### Install `lightning` (beta)
Following the instructions here:

```bash
git clone https://github.com/PyTorchLightning/lightning.beta.git
```
Move into folder
```bash
cd lightning.beta
```
Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

Download the `lightning` UI:
```bash
python scripts/download_frontend.py
```

### Locally

In order to run the application locally, run the following commands

```bash
git clone https://github.com/PyTorchLightning/lightning-pose-app.git
cd lightning-pose-app
pip install -r requirements.txt
lightning run app app.py
```

### Cloud

In order to run the application cloud, run the following commands

### On CPU

```
lightning run app app.py --cloud
```

### On GPU

```
USE_GPU=1 lightning run app app.py --cloud
```
