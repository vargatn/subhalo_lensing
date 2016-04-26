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

    def fmaker(self, theta):
        karr = self.spectra[:, 0]
        parr = self.spectra[:, 1]
        pkfunc = interp.interp1d(karr, parr, bounds_error=True, fill_value=np.nan)
        lower_norm = (pkfunc(karr[1]) / karr[1]**self.scalar_ind)
        upper_norm = pkfunc(karr[-2]) / (karr[-2]**(self.scalar_ind - 4)
                                     * np.log(karr[-2])**2)

        da = self.cosmo.angular_diameter_distance(self.z).value
        kfactor = (1. + self.z) * da
        scalar_ind = self.scalar_ind

        def ufunclike(x):
            kval = x / theta / kfactor
            res = 0.0
            if kval > karr[-2]:
                res = kval ** (scalar_ind - 4) * np.log(kval)**2. * upper_norm
            elif kval < karr[1]:
                res = kval ** scalar_ind * lower_norm
            else:
                res = pkfunc(kval)
            return res * x
        return np.vectorize(ufunclike)

    def wint(self, theta, N=100, h=0.03):
        fpow = self.fmaker(theta)
        h = hankel.HankelTransform(nu=2, N=N, h=h)
        val = h.transform(fpow) / theta**2. / (2. * math.pi)

        # prefactor before the integral
        da = self.cosmo.angular_diameter_distance(self.z).value
        cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3.).value
        Om0 = self.cosmo.Om0
        prefac = cdens0 * Om0 / da**2 / (1. + self.z) ** 3.

        return val * prefac

    def warr(self, rarr, N=100, h=0.03):
        da = self.cosmo.angular_diameter_distance(self.z).value
        thetarr = rarr / da
        return np.array([self.wint(theta, N=N, h=h) for theta in thetarr])