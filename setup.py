#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="lightning-pose-app",
    version="0.0.1",
    description="lightning app for lightning pose repo",
    author="Lee, Biderman, Whiteway",
    author_email="",
    url="https://github.com/Lightning-AI/lightning-pose-app",
    install_requires=[],


    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['nginx-8080.conf']},
)
