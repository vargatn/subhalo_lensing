# TODO implement here the direct summation calculation

import scipy
import astropy.units as u
from ..model.misc import default_cosmo
import numpy as np
import scipy.interpolate as interp

class PowerSpec:
    def __init__(self, prefix, suffix, zvals, loc='./', scalar_ind=0.96):
        """
        Loads linear matter power speclist from CAMB output

        :param prefix:
        :param suffix:
        :param zvals:
        :param loc:
        """
        self.scalar_ind = scalar_ind
        self.zvals = np.array(zvals)
        self.fnames = [prefix + "{:.1f}.".format(zval) + suffix
                       for zval in zvals]

        self.speclist = [np.loadtxt(name) for name in self.fnames]

    def _make_interpolator(self):
        pass
        # interp.interp1d()


    def _interpolate(self, z):
        """
        Interpolates power spectrum to the specified redshift

        ONLY INTERPOLATES DATA points

        Raises error if z is outside the redshift range

        :return: power spectrum at specified redshift
        """
        pass

    def spec(self, z, nearest=True):
        if nearest:
            specind = np.argmin((z - self.zvals)**2.)
            spectra = self.speclist[specind]
        else:
            raise NotImplementedError

        karr = spectra[:, 0]
        parr = spectra[:, 1]

        pkfunc = interp.interp1d(karr, parr)
        lower_norm = (pkfunc(karr[1]) / karr[1]**0.96)
        upper_norm = pkfunc(karr[-2]) / (karr[-2]**(self.scalar_ind - 4)
                                         * np.log(karr[-2])**2)

        def ufunclike(kval):
            if (kval > karr[-2]):
                return kval ** (self.scalar_ind - 4) * np.log(kval)**2. \
                       * upper_norm
            elif (kval < karr[1]):
                return kval ** self.scalar_ind * lower_norm
            else:
                return pkfunc(kval)

        return ufunclike

class SecHalo:
    def __init__(self, pspectra):
        """

        :param z: redshift
        :param pspectra: PowerSpectrum object
        """
        # default cosmology
        self.cosmo = default_cosmo()
        self.cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3)

        # power spectrum container
        self.pspectra = pspectra

    @staticmethod
    def _lfunc(l, theta, pfunc, kfactor):
        """The inside of the Hankel transformation"""
        val = l / (2. * np.pi) * scipy.special.jv(2, l * theta) *\
              pfunc(l / kfactor)
        return val

    def dsigma_nb(self, z, edges, **kwargs):
        """
        2-halo DSigma WITHOUT BIAS

        :param z: redshift
        :param theta: angular separation from main halo n radian
        :return:
        """
        cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                             (edges[i + 1] ** 2. - edges[i] ** 2.)
                             for i, edge in enumerate(edges[:-1])])
        # Angular size distance
        da = self.cosmo.angular_diameter_distance(z).to(u.Mpc)
        thetarr = cens / da.value

        print(thetarr)
        # to convert l to k
        kfactor = ((1. + z) * da.value)

        # linear matter power spectrum at required redshift
        pfunc = self.pspectra.spec(z)

        prefactor = (self.cdens0 * self.cosmo.Om0 / da**2.).value

        sumarr = np.array([self._sumfunc(self._lfunc, (theta, pfunc, kfactor),
                                         **kwargs) for theta in thetarr])
        return cens, sumarr * prefactor / 1e12

    @staticmethod
    def _sumfunc(func, args, lmin=0, lmax=1.5e5, dl=1):
        val = 0.0
        l = 0
        for l in np.arange(lmin, lmax, dl):
            val += func(l, *args)
        return val


class MatterBias:
    def __init__(self, pspectra):
        self.cosmo = default_cosmo()
        self.cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3)

        # self.cosmo.
        # power spectrum container
        self.pspectra = pspectra

    @staticmethod
    def bbks(k):
        h = 0.7
        ob = 0.02156 /h /h
        oc = 0.12544 /h /h
        om = ob + oc
        ns = 0.96
        q = k / (om * h * np.exp(-1. * ob * (1. + np.sqrt(2. * h) / om)))
        tk = np.log(1. + 2.34 * q) / (2.34 * q) * (1. + 3.89 * q + (16.1 * q)**2. +
                                                   (5.46 * q)**3. + (6.71 * q)**4.)\
                                                  **(-1. / 4.)
        return tk**2. * k**ns


    def shat(self, k):
        R = 8.
        wk8 = 3. / k**3. / R**3. * (-1. * k * R * np.cos(k * R) + np.sin(k * R))
        pk = self.bbks(k)

        return k**2. / (2. * scipy.pi**2.) * pk * wk8**2.

    def ref_sigma8(self):
        self.refs8 = 0.79








    def sigma8(self):
        """Consistency check"""
        z = 0.0
        pfunc = self.pspectra.spec(z)

        rr = 8.0

        def shat(k):
            return k**2. / (2. * scipy.pi**2.) * pfunc(k) * self.sph_tophat(k, rr)

        intres = scipy.integrate.quad(shat, 0.0, np.inf)

        return intres


    @staticmethod
    def sph_tophat(k, rr):
        wkr = 3. / k**3. / rr**3. *\
              (-1. * k * rr * np.cos(k * rr) + np.sin(k * rr))
        return wkr

    def sigma(self, rr, z):
        pfunc = self.pspectra.spec(z)

        intres = scipy.integrate.quad(self._ifunc, 0.0, np.inf, args=(rr, pfunc))
        print(intres)
        sigm = np.sqrt(intres[0]) / (2. * scipy.pi**2.)
        print(sigm)


    def _ifuncarr(self, karr, rr, z):
        pfunc = self.pspectra.spec(z)
        valarr = np.array([self._ifunc(k, rr, pfunc) for k in karr])
        return valarr

    def _ifunc(self, k, rr, pfunc):
        val = k**2. * pfunc(k) * self.sph_tophat(k, rr)**2.
        return val


