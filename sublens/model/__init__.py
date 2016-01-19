"""
Calculates analytic shear profile model


Contents:
---------

* single profile modelling

* ensemble modelling:   For a set of lenses reproduces the ensememble profile
                        given the parametric profile of the individual lenses

* mock object:  Calculate the profile based  on the parameters, or distr. of
                parameters
"""

# from .profiles import *
# from .misc import *

# import .profile
from astropy.cosmology import FlatLambdaCDM


def default_cosmo():
    cosmo0 = {
        'H0': 70,
        'Om0': 0.3
    }
    return FlatLambdaCDM(**cosmo0)