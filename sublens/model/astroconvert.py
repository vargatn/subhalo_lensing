"""
Conversions and scaling relations related to astrophysisc/cosmology
"""

import math
import astropy.units as units
import astropy.cosmology as cosmology


def nfw_params(cosmo, m200c, c200c, z, mlog10=True, *args, **kwargs):
    """
    Calculates the parameters of an NFW halo based on the given cosmo

    values should be relative to the *critical* density at z

    :param cosmo: FlatLambdaCDM cosmology object to be used
    :param m200c: M_200c in units of MSol
    :param c200c: c_200c
    :param z: cosmological redshift, must be somewhat larger than 0.
    :param mlog10: treat m200c as a logarithmic value
    :return (rs, rho_s, r200), in units of Mpc and MSol
    """
    err_msg = "Must define a valid astropy FlatLambdaCDM object"
    assert isinstance(cosmo, cosmology.FlatLambdaCDM)

    if mlog10:
        m200c = 10. ** m200c

    m200c *= units.solMass

    cdens = cosmo.critical_density(z).to(units.solMass / units.Mpc ** 3)
    r200 = (3. / 4. * m200c / (200. * cdens) / math.pi) ** (1. / 3.)
    rs = r200 / c200c
    dc = 200. / 3. * (c200c ** 3.) /\
         (math.log(1. + c200c) - c200c / (1. + c200c))
    rho_s = dc * cdens

    rs = rs.to(units.Mpc)

    return (rs.value, rho_s.value), ('rs', 'rho_s')


class ConvertorBase(object):
    def __init__(self):
        self.requires = []
        self.provides = []

    def convert(self, ftab):
        raise NotImplementedError


class DuffyCScale(ConvertorBase):
    def __init__(self):
        """
        Calculates NFW concentration based on M200 and redshift following
         Duffy et al. 2008
        """
        super().__init__()
        self.requires = sorted(['m200c', 'z'])
        self.provides = sorted(['c200c'])

        self.mpivot = 2. * 1e12  # Msun / h100
        self.a200 = 6.71
        self.b200 = -0.091
        self.c200 = -0.44

    def convert(self, ftab):
        carr = self.a200 * (ftab[:, 0] / self.mpivot) ** self.b200 *\
               (1. + ftab[:, 1]) ** self.c200
        return carr

