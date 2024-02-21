#!/usr/bin/env python

from setuptools import find_packages, setup

VERSION = "1.0.0"

# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "lightning[app]==2.1.3",
    "numpy",
    "opencv-python",
    "pandas",
    "scikit-learn",
    "streamlit",
    "streamlit-ace",
    "streamlit_autorefresh",
    "tables",
    "tqdm",
    "watchdog",
    "google-auth-oauthlib==0.7.1",  # freeze this for compatibility between tensorboard and label-studio
    "label-studio==1.9.1",
    "label-studio-sdk==0.0.32",
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
    },
}

setup(
    name="lightning-pose-app",
    version=VERSION,
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
