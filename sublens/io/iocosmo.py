"""
Default cosmology and cosmology params
"""

from astropy.cosmology import FlatLambdaCDM


def default_cosmo():
    """Default cosmology"""
    cosmo0 = {
        'H0': 70,
        'Om0': 0.3
    }
    return FlatLambdaCDM(**cosmo0)


def get_cosmopars(cosmo):
    """Gets parameters of the used cosmology"""
    assert isinstance(cosmo, FlatLambdaCDM)
    cosmo_pars = {
        'H0': cosmo.H0,
        'Om0': cosmo.Om0,
        'Tcmb0': cosmo.Tcmb0,
        'Neff': cosmo.Neff,
        'm_nu': cosmo.m_nu,
        'Ob0': cosmo.Ob0,
    }
    return cosmo_pars
