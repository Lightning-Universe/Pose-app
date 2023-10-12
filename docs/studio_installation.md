To install the app in a fresh Lightning Studio environment:
```bash
sudo apt install ffmpeg
conda install python=3.8
git clone --recursive https://github.com/Lightning-Universe/Pose-app
cd Pose-app
pip install -e .
pip install -r requirements_litpose.txt -e lightning-pose
```