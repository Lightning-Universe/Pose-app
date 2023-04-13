#!/usr/bin/env python

from setuptools import find_packages, setup

VERSION = "0.0.1"  # was previously None

# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "streamlit",
    "streamlit_autorefresh",
    "watchdog",
    "streamlit-ace",
    "virtualenv",
]

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
