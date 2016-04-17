"""
Calculates shear profile model


Contents:
---------

TBA
"""

from astropy.cosmology import FlatLambdaCDM


def default_cosmo():
    """Default cosmology"""
    cosmo0 = {
        'H0': 70,
        'Om0': 0.3
    }
    return FlatLambdaCDM(**cosmo0)