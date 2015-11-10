#!/usr/bin/env python3
# from distutils.core import setup
from setuptools import setup, find_packages

setup(name="sublens",
      description="Subhalo lensing and analysis framework",
      # packages=['sublens'],
      packages = find_packages(),
      author="Tamas Norbert Varga",
      author_email="vargat@gmail.com",
      version="0.1")

