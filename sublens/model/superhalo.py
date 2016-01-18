import time
import math
import scipy
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import scipy.interpolate as interp
import scipy.integrate as integr

from ..model.misc import nfw_pars
from ..model.misc import default_cosmo
from ..model.autoscale import cscale_duffy

class SecHalo:
    def __init__(self, pscont, cosmo=None):
        """
        Creates object for 2-halo term
        """
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo

        # power spectrum container
        self.pscont = pscont
        self.w2i = W2int(pscont, cosmo=cosmo)

    def ds(self, m, z, rr, mode='nearest'):
        ps = self.pscont.getspec(z, mode=mode)
        mb = MatterBias(ps)
        bias = mb.bias(m, z)

        resarr = self.w2i.getw2(rr, z)

        return rr, resarr[1] * bias

    def interp_ds(self, m, z, rr, kind="linear", rmult=False):
        """creates interpolator function for the profile based on rr evals"""
        rarr, ds = self.ds(m, z, rr)

        if rmult:
            ds *= rr

        ifunc = interp.interp1d(rarr, ds, kind=kind, fill_value=0.0,
                                bounds_error=False)

        return ifunc


class W2int(object):
    def __init__(self, pcont, cosmo=None):
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.pcont = pcont

    @staticmethod
    def _warg(l, theta, pfunc, z, da):
        val = l / (2. * np.pi) * scipy.special.jv(2, l * theta) * pfunc(l / ((1 + z) * da))
        return val

    def getw2(self, rarr, z):
        """Single two halo term"""
        # cosmological quantities
        da = self.cosmo.angular_diameter_distance(z).value
        cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3.).value
        Om0 = self.cosmo.Om0

        # prefactor before the integral
        prefac = cdens0 * Om0 / da**2

        # power spectrum at redshift z
        ps = self.pcont.getspec(z)

        thetarr = rarr / da
        resarr = np.zeros(shape=thetarr.shape)

        for i, th in enumerate(thetarr):
            resarr[i] = integr.romberg(self._warg, 0, 10**(2.5 - np.log10(th)),
                                       args=(th, ps.specfunc, z, da))
        resarr = resarr / 1e12
        return rarr, resarr * prefac


class MatterBias(object):
    def __init__(self, ps, cosmo=None):
        self.dc = 1.686

        if cosmo is None:
            cosmo =  default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.
        self.cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3)

        # power spectrum container
        self.ps = ps

    def nu(self, mass, z):
        """peak heiht"""
        rval = self.rbias(mass, z)
        nn = self.dc / self.ps.sigma(rval)
        return nn

    def bias(self, mass, z, delta=200):
        """linear matter bias based on halo mass and redshift"""
        nu = self.nu(mass / self.h, z)
        b = self.fbias(nu, delta=delta)
        return b

    def rbias(self, mass, z):
        """Lagrangian scale of the halo"""
        m = mass * u.solMass
        cdens = self.cosmo.critical_density(z).to(u.solMass / u.Mpc**3)
        rlag = ((3. * m) / (4. * np.pi * cdens * self.cosmo.Om(z))) ** (1. / 3.)
        return rlag.value

    def fbias(self, nu, delta=200.):
        """Calculates bias as a function of nu"""
        y = np.log10(delta)

        A = 1. + 0.24 * y * np.exp(-1. * ( 4. / y) ** 4.)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-1. * (4. / y) ** 4.)
        c = 2.4

        valarr = 1. - A * nu**a / (nu**a + self.dc**a) + B * nu**b + C * nu**c
        return valarr