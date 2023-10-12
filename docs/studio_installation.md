To install the app in a fresh Lightning Studio environment:
```bash
sudo apt install ffmpeg
conda install python=3.8
git clone --recursive https://github.com/Lightning-Universe/Pose-app
cd Pose-app
pip install -e .
pip install -r requirements_litpose.txt -e lightning-pose
sudo apt-get install libpq-dev
conda install libffi==3.3
pip install label-studio==1.9.1 label-studio-sdk==0.0.32
```