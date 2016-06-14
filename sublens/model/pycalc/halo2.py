import scipy.interpolate as interp
import numpy as np
import math
import hankel
import astropy.units as u


class W2calc(object):
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

    def fmaker(self, rr):
        """
        Creates a callable and "continous" function out of a power spectra,
        with the x-axis given in physical radii

        :param rr: physical scale (Mpc)
        :return: function
        """
        karr = self.spectra[:, 0]
        parr = self.spectra[:, 1]
        pkfunc = interp.interp1d(karr, parr, bounds_error=True,
                                 fill_value=np.nan)
        lower_norm = (pkfunc(karr[1]) / karr[1] ** self.scalar_ind)
        upper_norm = pkfunc(karr[-2]) / (karr[-2] ** (self.scalar_ind - 4)
                                         * np.log(karr[-2]) ** 2)

        scalar_ind = self.scalar_ind
        def ufunclike(x):
            kval = x / rr
            res = 0.0
            if kval > karr[-2]:
                res = kval ** (scalar_ind - 4) * np.log(
                    kval) ** 2. * upper_norm
            elif kval < karr[1]:
                res = kval ** scalar_ind * lower_norm
            else:
                res = pkfunc(kval)
            return res * kval

        return np.vectorize(ufunclike)

    def wint(self, rr, nn=100, hh=0.03):
        """Hankel integrator"""
        fpow = self.fmaker(rr)
        h = hankel.HankelTransform(nu=2, N=nn, h=hh)
        val = h.transform(fpow) / rr / (2. * np.pi)
        return val

    def warr(self, rarr, nn=100, hh=0.03):
        """Vectorized Hankel integrator with the appropriate prefactors"""
        intres = np.array([self.wint(rr, nn=nn, hh=hh) for rr in rarr])
        prefac = self.cosmo.critical_density(self.z) *\
                 self.cosmo.Om(self.z)
        prefac = prefac.to(u.solMass/ u.Mpc**3)

        return intres * prefac














