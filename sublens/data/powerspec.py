import numpy as np
import scipy.integrate as integr
import scipy.interpolate as interp

class PowerSpecContainer(object):
    def __init__(self, prefix, suffix, zvals, loc='./', scalar_ind=0.96, verbose=False):
        """
        Loads linear matter power spec lists from CAMB output

        :param prefix:
        :param suffix:
        :param zvals:
        :param loc:
        """

        self.scalar_ind = scalar_ind
        self.zvals = np.array(zvals)

        assert np.min(self.zvals**2.) < 1e-5, 'z=0 must be included'

        self.fnames = [prefix + "{:.1f}.".format(zval) + suffix
                       for zval in zvals]

        if verbose:
            print(self.fnames)

        self.speclist = [np.loadtxt(name) for name in self.fnames]

    def getspec0(self):
        specind = np.argmin((0.0 - self.zvals)**2.)
        spectra = self.speclist[specind]
        return spectra

    def getspec(self, z, mode='nearest', h=0.7, s8=0.79):
        if mode == "nearest":
            specind = np.argmin((z - self.zvals)**2.)
            spectra = self.speclist[specind]
        else:
            raise NotImplementedError

        spectra0 = self.getspec0()
        ps0 = PowerSpec(spectra0[:, 0], spectra0[:, 1], 0.0)
        s80 = ps0.sigma(8. / h)
        fscale = (s8 / s80) ** 2.

        return PowerSpec(spectra[:, 0], spectra[:, 1] * fscale, z)


class PowerSpec(object):
    def __init__(self, karr, parr, z, descr="", scalar_ind=0.96, h=0.7, s8=0.79):
        """
        Creates callable power spectrum at z

        :param karr: k values
        :param parr: power spectrum value
        :param z: redshift of the power spec
        """
        self.karr = karr
        self.parr = parr
        self.z = z
        self.scalar_ind= scalar_ind
        self.h = h

        self.specfunc = self._specmaker(self.karr, self.parr)

    def spec(self, kvals):
        """evaluates power spectrum at kvals"""
        kk = np.array(kvals)
        pp = np.array([self.specfunc(k) for k in kk])
        return pp

    def _specmaker(self, karr, parr):
        """inter and extrapolates a CAMB output power spectrum"""
        pkfunc = interp.interp1d(karr, parr)
        lower_norm = (pkfunc(karr[1]) / karr[1]**0.96)
        upper_norm = pkfunc(karr[-2]) /\
                     (karr[-2]**(self.scalar_ind - 4) *
                      np.log(karr[-2])**2)

        def ufunclike(kval):
            if (kval > karr[-2]):
                return kval ** (self.scalar_ind - 4) * np.log(kval)**2. \
                       * upper_norm
            elif (kval < karr[1]):
                return kval ** self.scalar_ind * lower_norm
            else:
                return pkfunc(kval)

        return ufunclike

    @staticmethod
    def sph_tophat(kr):
        """Fourier transform of the tophat function"""
        wkr = 3. / kr ** 3. * (-1. * kr * np.cos(kr) + np.sin(kr))
        return wkr

    def _sigmarg(self, k, rr, func):
        """argument of the sigma^2 integral"""
        val = 1. / (2. * np.pi ** 2.) * func(k) * k ** 2. *\
              self.sph_tophat(k * rr) ** 2.
        return val

    def _sigma(self, rr, func, kmin=1e-8, kmax=1e3):
        """Amplitude of fluctuations on the scale of rr [Mpc]"""
        sigma2 = integr.quad(self._sigmarg, kmin, kmax,
                             args=(rr, func))
        return np.sqrt(sigma2)[0]

    def sigma(self, rr, kmin=1e-8, kmax=1e3):
        """Amplitude of fluctuations on the scale of rr [Mpc]"""
        sigma2 = integr.quad(self._sigmarg, kmin, kmax,
                             args=(rr, self.specfunc))
        return np.sqrt(sigma2)[0]