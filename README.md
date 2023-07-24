# Lightning Pose App

Preprint: [Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling, and cloud-native open-source tools](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1)

This repo contains browser-based GUIs that facilitate the deveopment of a pose estimation project.

We provide three different apps:
* `demo_app.py`: using provided example data, train and evaluate pose estimation models
* `labeling_app.py`: stand-alone labeling app where you can upload videos, extract frames, and annotate keypoints on extracted frames using LabelStudio
* `app.py`: full app that includes labeling, training, and evaluation

The following instructions detail how to install the app locally for development purposes.
Note these instructions are only for Linux machines; we will update the documentation for macOS 
shortly.

#### Requirements
Your (potentially remote) machine has a Linux operating system, 
at least one GPU and **CUDA 11.0-12.x** installed. 
This is a requirement for **NVIDIA DALI**.

#### Community
[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)

The Lightning Pose app is primarily maintained by 
[Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University)
and 
[Dan Biderman](https://dan-biderman.netlify.app) (Columbia University). 
Come chat with us in Discord.

## Installation

First, check to see if you have ffmpeg installed by typing the following into the terminal:
```bash
ffmpeg -version
```
If not, install ffmpeg:
```bash
sudo apt install ffmpeg
```

Next, create a conda environment:

```bash
conda create --yes --name lai python=3.8
conda activate lai
```

Next, move to your home directory (or wherever you would like to download the code) 
and install the `Pose-app` repo:
```bash
cd ~
git clone --recursive https://github.com/Lightning-Universe/Pose-app
cd Pose-app
pip install -e .
```

There are additional installation steps based on which apps you would like to run.

#### Demo app and full app
Both the demo app and full app require access to a GPU machine for training and inference. See the
[lightning pose requirements](https://github.com/danbider/lightning-pose#requirements) 
to ensure you have the correct computational resources.

Install the `lightning-pose` repo:
```bash
cd ~/Pose-app/lightning-pose
pip install -e .
```

If you are using Ubuntu 22.04 or newer, you'll need an additional update for the Fiftyone package:
```bash
pip install fiftyone-db-ubuntu2204
```

#### Labeling app and full app
The labeling app and full app require the installation of Label Studio for data annotation. 
Because Label Studio and Lightning have conflicting dependencies, you will need to install 
Label Studio in a virtual environment. 

From `~/Pose-app`, run
```bash
virtualenv ~/venv-label-studio
source ~/venv-label-studio/bin/activate; sudo apt-get install libpq-dev; deactivate
source ~/venv-label-studio/bin/activate; conda install libffi==3.3; deactivate
source ~/venv-label-studio/bin/activate; pip install -e .; deactivate
source ~/venv-label-studio/bin/activate; pip install label-studio label-studio-sdk; deactivate
```

(note that `libffi` install may downgrade python, but this is fine.)

Test the installation of Label Studio:
```bash
source ~/venv-label-studio/bin/activate; label-studio version; deactivate
```
If the installation was successful you will not see any error messages by running the above command.

## Run the apps

#### Running locally
Run any of the above three apps from the command line:
```bash
cd ~/Pose-app
lightning run app <app_name.py>
```

#### Running on cloud
More info coming soon!

[comment]: <> (Running the app on the cloud is easy!)
[comment]: <> (```bash)
[comment]: <> (lightning run app <app_name.py> --cloud --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video)
[comment]: <> (```)

Your data - videos, labels, models, etc. - will all be stored in a project directory detailed 
[here](docs/directory_structure.md).

## FAQs

* **Can I import a pose estimation project in another format into the app?** 
We currently support conversion from DLC projects into Lightning Pose projects
(if you would like support for another format, 
please [open an issue](https://github.com/Lightning-Universe/Pose-app/issues)). 
First, make sure that all videos in your DLC project are actual files, not symbolic links. 
Symlinked videos in the DLC project will not be uploaded. 
Next, compress your DLC project into a zip file - you will upload this into the app later.
Finally, when you run the labeling app or full app select "Create new project from source" and
upload your zip file. As of 07/2023 context datasets are not automatically created upon import; if
this is a feature you would like to see, 
please [open an issue](https://github.com/Lightning-Universe/Pose-app/issues). 
* **How do I increase the file upload size limit?**
Set the following environment variable (in units of MB);
 we recommend using videos less than 500MB for best performance.
```bash
lightning run app <app_name.py> --env STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
```
