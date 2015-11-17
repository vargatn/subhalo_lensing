import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import math


def default_cosmo():
    cosmo0 = {
        'H0': 100,
        'Om0': 0.3
    }
    return FlatLambdaCDM(**cosmo0)


def nfw_pars(m200, c200, z, cosmo=None):
    """
    Calculates parameters for NFW

    :param m200: Mass
    :param c200: concentration
    :param z: redshift
    :return: scale radius, \rho_s
    """

    if cosmo is None:
        cosmo = default_cosmo()

    assert isinstance(cosmo, FlatLambdaCDM)

    cdens = cosmo.critical_density(z).to(u.solMass / u.Mpc**3).value

    r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
    rs = r200 / c200
    dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200 - c200 / (1. + c200)))
    rho_s = dc * cdens

    return rs, rho_s, r200
