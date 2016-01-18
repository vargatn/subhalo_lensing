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

from ..model.halo1 import CentHaloNFW
from ..model.halo2 import SecHalo
from ..model.halo2 import SecHaloOld

from ..model.haloc import Halo


class HGen(object):
    def __init__(self, pscont, cscale=cscale_duffy, cosmo=None):

        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.
        self.cscale = cscale

        # primary halo object
        self.ph = CentHaloNFW(cosmo)

        # secondary halo object
        self.sh = SecHalo(pscont, cosmo=cosmo)

    def cen_ds(self, m, z, rr):
        assert z > 0.05, 'Two halo term only works for z > 0'

        rarr, ds2 = self.sh.ds(m, z, rr)
        rarr, ds1 = self.ph.ds(m, z, rr)

        return rr, ds1 + ds2

    def ocen_ds(self, m, z, rr, dist=0.):
        """offcentered halo"""

        # creating interpolating grid for 2-halo term
        orr = np.logspace(-3, 2, num=200)
        ifunc2 = self.sh.interp_ds(m, z, orr, rmult=True)

        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.ph.nfw_params(m, c, z)

        # ds = np.array([self._ocen_prof(0.0, r, rs, rho_s, ifunc2) for r in rr])
        ds = np.array([self.offc_int(r, rs, rho_s, ifunc2, dist) for r in rr])
        return rr, ds

    def _ocen_prof(self, phi, r, rs, rho_s, ifunc2, dist=0.):
        dist2 = dist * dist
        assert r > 0.
        rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
        rr = math.sqrt(rr2)

        # rr = r
        term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. - 2. * dist * math.cos(phi) - r)) / rr2
        term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2

        ph_shear_tcent = self.ph.nfw_shear_t(rr, rs, rho_s) / 1e12
        sh_shear_tcent = ifunc2(rr)

        # print(ph_shear_tcent, sh_shear_tcent)
        shear_tcent = ph_shear_tcent + sh_shear_tcent

        shear_t = shear_tcent * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
        return shear_t


    def offc_int(self, r, rs, rho_s, ifunc2, dist=0.):
        """Integrates in a circle at r"""

        # print(r)
        intres = i
    ntegr.quad(self._ocen_prof, -math.pi, math.pi,
                             args=(r, rs, rho_s, ifunc2, dist))

        phmean = intres[0]
        circumference = 2. * math.pi * r

        return phmean / circumference

    # def offc_ring(self):


# def nfw_offc_dblint(phi, r, rs, rho_s, dist=0.0, *args, **kwargs):
#     """tangential profile of Off-Centered NFW halo"""
#
#     dist2 = dist * dist
#     assert r > 0.
#     rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
#     assert rr2 > 0.
#     rr = math.sqrt(rr2)
#
#     term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. -
#                           2. * dist * math.cos(phi) - r)) / rr2
#     term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2
#
#     par = rho_s * rs / 1e12
#     x = rr / rs
#
#     shear_t_cent = nfw_shear_t(x=x, par=par) * r
#     shear_t = shear_t_cent * (term1 * math.cos(2. * phi) +
#                               term2 * math.sin(2. * phi))
#
#     return shear_t