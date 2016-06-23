"""
2-Halo term formulas
"""

import scipy.interpolate as interp
from scipy.integrate import quad
import numpy as np
import hankel
import astropy.units as u


class H2calc(object):
    """Linear 2-halo term"""
    def __init__(self, z, spectra, cosmo, scalar_ind=0.96):
        """
        Linear 2-halo term formula with a single multiplicative bias parameter
        (the bias is not included here, only the integral!)

        \Delta\Sigma(r,z) = b * \int_0^\infty  dk  k / (2 p\i) J_2(r k) P(k, z)

        :param z: redshift

        :param spectra: linear matter power spectra, like the one from CAMB

        :param cosmo: astropy cosmology object

        :param scalar_ind: scalar index of the Power Spectrum
        """
        self.z = z
        self.spectra = spectra
        self.cosmo = cosmo
        self.scalar_ind = scalar_ind
        self.pkfunc = self.finterp()

    def finterp(self):
        """Interpolates the CAMB output power spectra into a callable func."""
        karr = self.spectra[:, 0]
        parr = self.spectra[:, 1]
        return interp.UnivariateSpline(karr, parr, s=0, ext="zeros")

    def fmaker(self, rr):
        """
        Creates a callable and "continous" function out of a power spectra,
        with the x-axis given in physical radii

        :param rr: physical scale (Mpc)

        :return: function
        """
        pkfunc = self.pkfunc
        karr = self.spectra[:, 0]
        lower_norm = (pkfunc(karr[1]) / karr[1] ** self.scalar_ind)
        upper_norm = pkfunc(karr[-2]) / (karr[-2] ** (self.scalar_ind - 4)
                                         * np.log(karr[-2]) ** 2)
        scalar_ind = self.scalar_ind

        def ufunclike(x):
            kval = x / rr
            if kval > karr[-2]:
                res = kval ** (scalar_ind - 4) * np.log(
                    kval) ** 2. * upper_norm
            elif kval < karr[1]:
                res = kval ** scalar_ind * lower_norm
            else:
                res = pkfunc(kval)
            return res * kval
        return np.vectorize(ufunclike)

    def prefac(self):
        """calculates prefactor for the integral"""
        pref = self.cosmo.critical_density(self.z) * \
               self.cosmo.Om(self.z)
        pref = pref.to(u.solMass / u.Mpc ** 3)
        return pref.value

    def wint(self, rr, nn=100, hh=0.03):
        """Hankel integrator"""
        fpow = self.fmaker(rr)
        h = hankel.HankelTransform(nu=2, N=nn, h=hh)
        val = np.array(h.transform(fpow)) / rr / (2. * np.pi)
        return val

    def wval(self, rr, nn=100, hh=0.03):
        """Vectorized Hankel integrator with the appropriate prefactors"""
        intres = self.wint(rr, nn=nn, hh=hh)[0]
        return intres * self.prefac()

    def _wval(self, rr, nn=100, hh=0.03):
        intres = self.wint(rr, nn=nn, hh=hh)[0] * rr * 2. * np.pi
        return intres

    def wring(self, r0, r1, nn=100, hh=0.03):
        dsum = np.array(quad(self._wval, r0, r1, args=(nn, hh), epsabs=0,
                        epsrel=1e-3))
        aring = np.pi * (r1 ** 2. - r0 ** 2.)
        return dsum[0] / aring * self.prefac()


