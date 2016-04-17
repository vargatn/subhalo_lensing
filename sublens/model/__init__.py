

from astropy.cosmology import FlatLambdaCDM

# from ..model import profiles
# from ..model import astroconvert
# from ..model import pyprojector


def default_cosmo():
    """Default cosmology"""
    cosmo0 = {
        'H0': 70,
        'Om0': 0.3
    }
    return FlatLambdaCDM(**cosmo0)