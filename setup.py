#!/usr/bin/env python

from pathlib import Path
from setuptools import find_packages, setup


def read(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "lightning[app]==2.2.5",
    "ensemble-kalman-smoother==2.0.1",
    "numpy",
    "opencv-python-headless",
    "pandas",
    "scikit-learn",
    "streamlit",
    "streamlit-ace",
    "streamlit_autorefresh",
    "tables",
    "tqdm",
    "unvicorn==0.34.0",
    "watchdog",
    "google-auth-oauthlib",
    "label-studio==1.13.1",
    "label-studio-sdk==1.0.5",  # freeze for compatibility with label-studio
]

# additional requirements
extras_require = {
    "dev": {
        "flake8",
        "isort",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx-rtd-dark-mode",
        "sphinx-automodapi",
        "sphinx-copybutton",
        "sphinx-design",
        "sphinxcontrib.youtube",
    },
}

setup(
    name="lightning-pose-app",
    version=get_version(Path("lightning_pose_app").joinpath("__init__.py")),
    description="lightning app for lightning pose repo",
    long_description=long_description,
    author="Dan Biderman and Matt Whiteway and Robert Lee",
    author_email="danbider@gmail.com",
    url="https://github.com/Lightning-Universe/Pose-app",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    package_data={},
)
