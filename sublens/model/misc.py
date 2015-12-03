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

    m200 *= u.solMass

    cdens = cosmo.critical_density(z).to(u.solMass / u.Mpc**3)


    # print("cdens = {:.5e}".format(cdens))

    # cdens = cosmo.critical_density(z).to(u.solMass / u.pc**3)
    #    print("{:.5e}".format(m200))
    # r200 = (m200 * cdens)
    r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
    rs = r200 / c200
    dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200) - c200 / (1. + c200))
    # print("dc = ", dc)
    rho_s = dc * cdens
    # rho_s = rho_s.to(u.solMass / u.pc**3)
    # print("rho_s = {:.5e}".format(rho_s))

    rs = rs.to(u.Mpc)
    # print("rs = {:.5e}".format(rs))
    r200 = r200.to(u.Mpc)


    return rs.value, rho_s.value, r200.value