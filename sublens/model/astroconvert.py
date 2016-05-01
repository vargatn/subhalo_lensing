"""
Conversions and scaling relations related to astrophysisc/cosmology
"""

import numpy as np
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


def lm200_rykoff_orig(l, **kwargs):
    # FIXME this is a throwaway function
    """
    Matches redmapper richness to cluster m200

    :param l: richness
    :return: M200 [M_sun / h100]
    """
    mpivot = 1e14 # msun / h100
    m200 = np.exp(1.48 + 1.06 * np.log(l / 60.)) * mpivot
    return m200


def fabrice_mlum_scaleing(rlum, h=1.0):
    # FIXME this is a throwaway function
    """Some rough scaling for r band galaxy mass..."""
    Lpivot = 1.6e10 / h**2.
    Mpivot = 18.6e11 / h
    nu = 1.05

    value = (rlum / Lpivot)**nu * Mpivot
    return value


class ConvertorBase(object):
    def __init__(self):
        self.requires = []
        self.provides = []

        self.pivots0 = {}
        self.pivots = {}

        self.exponents = {}
        self.exponents0 = {}

    def _get_indices(self, parnames):
        indices = np.arange(len(self.requires))
        if parnames is not None:
            indices = np.array([np.where(req == np.array(parnames))[0][0]
                                for req in self.requires])
        if len(set(indices)) != len(indices):
            raise NameError("duplicate parameter name specified!")

        return indices

    def convert(self, ftab):
        raise NotImplementedError

    def update(self, new_pivots=None, new_exp=None, **kwargs):
        if new_pivots is not None and isinstance(new_pivots, dict):
            self.pivots = new_pivots
        if new_exp is not None and isinstance(new_exp, dict):
            self.exponents = new_exp

    def reset(self, *args, **kwargs):
        self.pivots = self.pivots0.copy()
        self.exponents = self.exponents0.copy()


# TODO documentation!!!
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

    def convert(self, ftab, parnames=None):

        indices = self._get_indices(parnames)

        marr = ftab[:, indices[0]]
        zarr = ftab[:, indices[1]]

        carr = self.a200 * (10.**marr / self.mpivot) ** self.b200 *\
               (1. + zarr) ** self.c200
        return carr[:, np.newaxis]


class SimpleSubhaloMapper(ConvertorBase):
    def __init__(self):
        super().__init__()

        self.requires = sorted(['r-Lum', 'dist'])
        self.provides = sorted(['m200c', 'c200c'])

        self.pivots0 = {
            'lum_pivot': 10.514,
            'm_pivot': 12.4,
            'c_pivot': 6.67,
        }

        self.exponents0 = {
            'lexp': 1.05,
            'rexp_m': 0.0,
            'rexp_c': 0.0,
            'mexp': -0.092,
        }

        self.pivots = self.pivots0.copy()
        self.exponents = self.exponents0.copy()

    def __str__(self):
        return "SimpleSubhaloMapper"

    def convert(self, ftab, parnames=None):
        indices = self._get_indices(parnames)
        rlum_arr = 10. ** ftab[:, indices[0]]
        r_arr = ftab[:, indices[1]]

        mpivot = 10. ** self.pivots['m_pivot']
        lpivot = 10. ** self.pivots['lum_pivot']

        marr = mpivot * (rlum_arr / lpivot) ** self.exponents['lexp']\
               * r_arr ** self.exponents['rexp_m']

        carr = self.pivots['c_pivot'] *\
               (marr / mpivot) ** self.exponents['mexp'] *\
               r_arr ** self.exponents['rexp_c']

        restab = np.vstack((carr, np.log10(marr))).T
        return restab


class SimpleParentMapper(ConvertorBase):
    def __init__(self):
        super().__init__()

        self.requires = sorted(['lambda'])
        self.provides = sorted(['m200c', 'c200c'])

        self.pivots0 = {
            'm_pivot': 14.0,
            'lamb_pivot': 60.0,
            'mc_pivot': 12.4,
            'c_pivot': 6.67,
        }

        self.exponents0 = {
            'lexp0': 1.48,
            'lexp1': 1.06,
            'mexp': -0.092,
        }

        self.pivots = self.pivots0.copy()
        self.exponents = self.exponents0.copy()

    def __str__(self):
        return "SimpleParentMapper"

    def convert(self, ftab, parnames=None):
        indices = self._get_indices(parnames)

        mpivot = 10. ** self.pivots['m_pivot']
        lamb = ftab[:, indices[0]]

        marr= np.exp(self.exponents['lexp0'] + self.exponents['lexp1'] *
                      np.log(lamb / self.pivots['lamb_pivot'])) * mpivot

        carr =  self.pivots['c_pivot'] * \
                (marr / mpivot) ** self.exponents['mexp']

        restab = np.vstack((carr, np.log10(marr))).T
        return restab




