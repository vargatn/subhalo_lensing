"""
Data handling routines


Contents:
----------

* magtools:
    - single band k-correction using red galaxy template
    - distance modulus wrapping astropy
     (Tested to work well for DES redmagic galaxies)

* rotate:
    - Exact Spherical rotations
    - Great circle distance
    - cart to polar transformation and the reverse

* shearops:
    - calculate bin edges, centers and areas
    - Handle Raw profiles read from file
    - Creat stacked profiles
    - add and subtract profiles with their proper JackKnife covariance
    - Measure cross-covariance of several profiles

* troughs:
    - Handle trough maps
    - query targets in the vicinity of troughs
    - create matched cluster -- trough coordinate tables

"""

# from .shearops import *
# from .rotate import *
# from .magtools import *
# from .troughs import *
