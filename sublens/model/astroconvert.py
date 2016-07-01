"""
Conversions and scaling relations related to astrophysisc/cosmology
"""

import math
import numpy as np
import astropy.units as units


def nfw_params(cosmo, z, mlike, clike, delta=200., dens="crit", mlog10=True,
               *args, **kwargs):
    """
    Calculates the parameters of an NFW halo based on the given cosmology

    :param cosmo: FlatLambdaCDM cosmology object to be used

    :param z: redshift of object

    :param mlike: mass like parameter

    :param clike: concentraton like parameter

    :param delta: overdensity value e.g.: 200, 500 etc...

    :param dens: density type: crit or mean

    :param mlog10: if mass is given in log10 base

    :return: rs, rho_s), in units of Mpc and MSol
    """
    if mlog10:
        mlike = 10. ** mlike
    mlike *= units.solMass

    if dens == "crit":
        denslike = cosmo.critical_density(z).to(units.solMass / units.Mpc ** 3)
    elif dens == "mean":
        denslike = cosmo.critical_density(z).to(units.solMass /
                                                units.Mpc ** 3) * cosmo.Om(z)

    dc = delta / 3. * (clike ** 3.) / \
         (math.log(1. + clike) - clike / (1. + clike))
    rho_s = dc * denslike

    rlike = (3. / 4. * mlike / (delta * denslike) / math.pi) ** (1. / 3.)
    rs = rlike / clike
    rs = rs.to(units.Mpc)

    return (rs.value, rho_s.value), ('rs', 'rho_s')


class ConvertorBase(object):
    """parent class for parameter conversions"""
    def __init__(self):
        """
        Converts the specified parameter names and the specified data table
        into an other sert of physical parameters

        This is just the base class, does not do anything
        """
        self.requires = []
        self.provides = []

        self.pivots0 = {}
        self.pivots = {}

        self.exponents = {}
        self.exponents0 = {}

    def _get_indices(self, parnames):
        """Gets indices for the data table, based on paramneter names"""
        indices = np.arange(len(self.requires))
        if parnames is not None:
            indices = np.array([np.where(req == np.array(parnames))[0][0]
                                for req in self.requires])
        if len(set(indices)) != len(indices):
            raise NameError("duplicate parameter name specified!")

        return indices

    def convert(self, ftab):
        raise NotImplementedError

    def update(self, **kwargs):
        """Updates the meta-parameters (exponents) which are used
         for the conversion"""
        pivots = {}
        for k in set(self.pivots0.keys()).intersection(set(kwargs.keys())):
            self.pivots.update({k: kwargs[k]})
        exponents = {}
        for k in set(self.exponents0.keys()).intersection(set(kwargs.keys())):
                self.exponents.update({k: kwargs[k]})

    def reset(self, *args, **kwargs):
        """resets meta-parameters and exponents to default"""
        self.pivots = self.pivots0.copy()
        self.exponents = self.exponents0.copy()


class DuffyCScale(ConvertorBase):
    def __init__(self):
        """
        Calculates NFW concentration based on M200 and redshift following
         Duffy et al. 2008

        requires:
            m200c, z

        provides:
            c200c
        """
        super().__init__()
        self.requires = sorted(['m200c', 'z'])
        self.provides = sorted(['c200c'])

        self.mpivot = 2. * 1e12  # Msun / h100
        self.a200 = 6.71
        self.b200 = -0.091
        self.c200 = -0.44

    def convert(self, ftab, parnames=None, point=False):
        """
        Performs conversion

        :param ftab: np 2D array containing the  parameters, must be sorted if
            parnames are not specified

        :param parnames: list of parameter names for each column of the input
                ftab table

        :return: np.array of c200c values
        """
        indices = self._get_indices(parnames)

        marr = ftab[:, indices[0]]
        zarr = ftab[:, indices[1]]

        carr = self.a200 * (10.**marr / self.mpivot) ** self.b200 *\
               (1. + zarr) ** self.c200
        return carr[:, np.newaxis]


class SimpleSubhaloMapper(ConvertorBase):
    def __init__(self):
        """
        Conversion tool for subhalo paramteres

        requires:
        -------------
            'dist', 'rlum', 'z'

        provides:
        -------------
            'c200c', 'm200c', 'z'
        """
        super().__init__()

        self.requires = sorted(['dist', 'rlum', 'z'])
        self.provides = sorted(['c200c', 'm200c', 'z'])

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
        z_arr = ftab[:, indices[2]]

        mpivot = 10. ** self.pivots['m_pivot']
        lpivot = self.pivots['lum_pivot']

        marr = mpivot * (rlum_arr / lpivot) ** self.exponents['lexp']\
               * r_arr ** self.exponents['rexp_m']

        carr = self.pivots['c_pivot'] *\
               (marr / mpivot) ** self.exponents['mexp'] *\
               r_arr ** self.exponents['rexp_c']

        restab = np.vstack((carr, np.log10(marr), z_arr)).T
        return restab


class SimpleParentMapper(ConvertorBase):
    def __init__(self):
        """
        Conversion tool for subhalo paramteres

        requires:
        -------------
            'dist', 'lamb', 'z'

        provides:
        -------------
            'dist', 'm200c', 'z'
        """
        super().__init__()

        self.requires = sorted(['dist', 'lamb', 'z'])
        self.provides = sorted(['dist', 'm200c', 'z'])

        self.pivots0 = {
            'm_pivot_p': 13.8,
            'lamb_pivot': 60.0,
        }

        self.exponents0 = {
            'lexp0': 1.48,
            'lexp1': 1.06,
        }

        self.pivots = self.pivots0.copy()
        self.exponents = self.exponents0.copy()

    def __str__(self):
        return "SimpleParentMapper"

    def convert(self, ftab, parnames=None):
        indices = self._get_indices(parnames)

        mpivot = 10. ** self.pivots['m_pivot_p']

        dist_arr = ftab[:, indices[0]]
        lamb = 10. ** ftab[:, indices[1]]
        z_arr = ftab[:, indices[2]]

        marr= np.exp(self.exponents['lexp0'] + self.exponents['lexp1'] *
                      np.log(lamb / self.pivots['lamb_pivot'])) * mpivot

        restab = np.vstack((dist_arr / 0.7, np.log10(marr * 0.7), z_arr)).T
        return restab

# -----------------------------------------------------------------------------
# Older functions for reference


def lm200_rykoff_orig(l, **kwargs):
    """
    Matches redmapper richness to cluster m200

    :param l: richness
    :return: M200 [M_sun / h100]
    """
    mpivot = 1e14 * 0.7# msun / h100
    # m200 = np.exp(1.48 + 1.06 * np.log(l / 60.)) * mpivot
    m200 = mpivot * np.exp(1.48) * (l / 60.)**1.06
    return m200


def fabrice_mlum_scaleing(rlum, h=1.0):
    """Some rough scaling for r band galaxy mass..."""
    Lpivot = 1.6e10 / h**2.
    Mpivot = 18.6e11 / h
    nu = 1.05

    value = (rlum / Lpivot)**nu * Mpivot
    return value


def cscale_duffy(m200=0.0, z=0.0, **kwargs):
    """
    Calculates NFW concentration based on M200 and redshift
    I use the halo definition with r200 from Duffy et al. 2008
    h is propagated through
    :param m200: nfw mass [M_sun / h100]
    :param z: halo redshift
    :return: concentration
    """
    mpivot = 2. * 1e12  # Msun / h100
    a200 = 6.71
    b200 = -0.091
    c200 = -0.44
    c = a200 * (m200 / mpivot) ** b200 * (1. + z) ** c200
    return c
