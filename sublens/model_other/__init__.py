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

# TODO write pure python code for offcentered halo

# TODO write cython code for offcentered halo

# TODO write numba code for offcentered halo