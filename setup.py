#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="m05-gr03-2022",
    version="1.0.0",
    description="Basic example of a Reproducible Research Project in Python for the M05 course",
    url="https://github.com/Chxresubles/m05_miniprojects",
    license="BSD-2",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["wine_quality = wine_quality.algorithm:main",
                                      "boston_house_prices = boston_house_prices.algorithm:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
