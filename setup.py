#!/usr/bin/env python
import os
from io import open

from setuptools import find_packages, setup

_PARENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
_LONG_DESCRIPTION = open(os.path.join(_PARENT_DIRECTORY, "README.md"), encoding="utf-8").read()
_INSTALL_REQUIRES = open(os.path.join(_PARENT_DIRECTORY, "requirements.txt")).read().splitlines()


def main():
    setup(
        name="stereo_rectification",
        version="0.1.0",
        description="uncalibrated stereo rectification",
        long_description=_LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="xmba15",
        url="https://github.com/xmba15/uncalibrated_stereo_rectification.git",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        packages=find_packages(exclude=["tests"]),
        install_requires=_INSTALL_REQUIRES,
    )


if __name__ == "__main__":
    main()
