import scipy.interpolate as interp
import numpy as np
import math
import hankel
import astropy.units as u

# TODO document this

class W2calc(object):
    def __init__(self, z, spectra, cosmo, scalar_ind=0.96):
        self.z = z
        self.spectra = spectra
        self.cosmo = cosmo
        self.scalar_ind = scalar_ind

    def fmaker(self, rr):
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
        fpow = self.fmaker(rr)
        h = hankel.HankelTransform(nu=2, N=nn, h=hh)
        val = h.transform(fpow) / rr / (2. * np.pi)
        return val

    def warr(self, rarr, nn=100, hh=0.03):
        intres = np.array([self.wint(rr, nn=nn, hh=hh) for rr in rarr])
        prefac = self.cosmo.critical_density(self.z) *\
                 self.cosmo.Om(self.z)
        prefac = prefac.to(u.solMass/ u.Mpc**3)

        return intres * prefac














