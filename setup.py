#!/usr/bin/env python3
# from distutils.core import setup
from setuptools import setup, find_packages

setup(name="sublens",
      description="Subhalo lensing and analysis framework",
      packages = find_packages(),
      install_requires = ["numpy>=0.11",
                          "scipy>=0.17",
                          "astropy>=1.1",
                          "pandas>=0.18",
                          "matplotlib>=1.5",
                          "hankel",
                          "kmeans_radec",
                          "cython>=0.23"],

      author="Tamas Norbert Varga",
      author_email="vargat@gmail.com",
      version="0.1")
