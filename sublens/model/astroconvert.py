"""
Conversions and scaling relations related to astrophysisc/cosmology
"""

import math
import astropy.units as units
import astropy.cosmology as cosmology


def nfw_params(cosmo, m200c, c200c, z, *args, **kwargs):
    """
    Calculates the parameters of an NFW halo based on the given cosmo

    values should be relative to the *critical* density at z

    :param cosmo: FlatLambdaCDM cosmology object to be used
    :param m200: M_200c in units of MSol
    :param c200: c_200c
    :param z: cosmological redshift, must be somewhat larger than 0.
    :return (rs, rho_s, r200), in units of Mpc and MSol
    """
    err_msg = "Must define a valid astropy FlatLambdaCDM object"
    assert isinstance(cosmo, cosmology.FlatLambdaCDM)

    m200c *= units.solMass

    cdens = cosmo.critical_density(z).to(units.solMass / units.Mpc ** 3)
    r200 = (3. / 4. * m200c / (200. * cdens) / math.pi) ** (1. / 3.)
    rs = r200 / c200c
    dc = 200. / 3. * (c200c ** 3.) /\
         (math.log(1. + c200c) - c200c / (1. + c200c))
    rho_s = dc * cdens

    rs = rs.to(units.Mpc)

    return (rs.value, rho_s.value), ('rs', 'rho_s')