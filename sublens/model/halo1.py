import time
import math
import numpy as np
from ..model.misc import nfw_pars
import jdnfw.nfw as jd_nfw
from ..model.misc import default_cosmo
from ..model.autoscale import cscale_duffy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from scipy import integrate as integr


class CentHaloNFW(object):
    def __init__(self, cosmo=None, cscale=cscale_duffy):
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.
        self.cscale = cscale

    def nfw_params(self, m200, c200, z):
        m200 *= u.solMass

        cdens = self.cosmo.critical_density(z).to(u.solMass / u.Mpc**3)
        r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
        rs = r200 / c200
        dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200) - c200 / (1. + c200))
        rho_s = dc * cdens

        rs = rs.to(u.Mpc)
        r200 = r200.to(u.Mpc)

        return rs.value, rho_s.value, r200.value

    def ds_point(self, m , z, r):
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.nfw_params(m, c, z)

        ds = self.nfw_shear_t(r, rs, rho_s) / r / 1e12

        return r, ds

    def ds(self, m, z, rr):

        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.nfw_params(m, c, z)

        ds = np.array([self.nfw_shear_t(r, rs, rho_s) / r
                   for i, r in enumerate(rr)])

        ds /= 1e12

        return rr, ds

    @staticmethod
    def nfw_shear_t(r, rs, rho_s):
        """
        RAW centered NFW halo

        Equation from:
        Gravitational Lensing by NFW Halos
        Wright, Candace Oaxaca; Brainerd, Tereasa G.
        http://adsabs.harvard.edu/abs/2000ApJ...534...34W

        :return: value of tangential shear_old at distance x
        """

        x = r / rs

        shear = 0

        if 0. < x < 1.:
            shear = rs * rho_s * ((8. * math.atanh(math.sqrt((1. - x) / (1. + x))) / (
                x ** 2. * math.sqrt(1. - x ** 2.)) +
                            4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1.) +
                            (4. * math.atanh(math.sqrt((1. - x) / (1. + x)))) / (
                                (x ** 2. - 1.) * math.sqrt(1. - x ** 2.))))

        elif x == 1.:
            shear = rs * rho_s *(10. / 3. + 4. * math.log(1. / 2.))

        elif x > 1.:
            shear = rs * rho_s *((8. * math.atan(math.sqrt((x - 1.) / (1. + x))) / (
                x ** 2. * math.sqrt(x ** 2. - 1.)) +
                            4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1) +
                            (4. * math.atan(math.sqrt((x - 1.) / (1. + x)))) / (
                                (x ** 2. - 1.) ** (3. / 2.))))

        return shear * r

# def nfw_prof_noint(c200, m200, z, edges):
#     rs, rho_s, r200 = nfw_pars(m200, c200, z)
#
#     areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
#                       for i, edge in enumerate(edges[:-1])])
#     cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
#                      (edges[i + 1] ** 2. - edges[i] ** 2.)
#                      for i, edge in enumerate(edges[:-1])])
#
#     ds = np.array([nfw_shear_t(cen, rs, rho_s) / cen
#                    for i, cen in enumerate(cens)])
#
#     ds = ds / 1e12
#
#     return cens, ds



