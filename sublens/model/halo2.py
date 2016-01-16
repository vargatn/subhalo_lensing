
import scipy
import astropy.units as u
from ..model.misc import default_cosmo
import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integr
import pickle


# class SecHalo:

class SecHalo:

    def __init__(self, pscont, w2log, cosmo=None):
        """
        Creates object for 2-halo term
        """
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo

        # power spectrum container
        self.pscont = pscont
        self.w2log = w2log

    def ds(self, m, z, mode='nearest'):
        """Delta sigma profile of 2-halo term"""
        ps = self.pscont.getspec(z, mode=mode)
        mb = MatterBias(ps)
        bias = mb.bias(m, z)

        if mode=='nearest':
            ind = np.argmin((self.w2log['zarr'] - z)**2.)
        else:
            raise NotImplementedError

        return self.w2log['rarr'], self.w2log['restab'][ind, :] * bias


class W2calc(object):
    def __init__(self, rarr, zarr, pcont, cosmo=None):
        """
        2-halo term without the bias

        Calculates the W_2 integral and the corresponding prefactors
        """
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.rarr = rarr
        self.zarr = zarr
        self.pcont = pcont

        self.darr = np.array([self.cosmo.angular_diameter_distance(z).value
                              for z in zarr])
        self.thetatab = np.array([[r/da for r in self.rarr]
                                  for da in self.darr]) # radian

        self.restab = None

    @staticmethod
    def _warg(l, theta, pfunc, z, da):
        val = l / (2. * np.pi) * scipy.special.jv(2, l * theta) * pfunc(l / ((1 + z) * da))
        return val

    def calc(self, sname=None):
        """
        creates the actual data table (tested only z = 0.1 - 1.0)
        """
        restab = np.ones(shape=self.thetatab.shape) * -1.

        cdens0 = self.cosmo.critical_density0.to(u.solMass / u.Mpc**3.)
        Om0 = self.cosmo.Om0

        for i, row in enumerate(self.thetatab):
            z = self.zarr[i]
            ps = self.pcont.getspec(z)
            da = self.darr[i]
            for j, th in enumerate(row):
                restab[i, j] = integr.romberg(self._warg, 0,
                                              10**(2.5 - np.log10(th)),
                                              args=(th, ps.specfunc, z, da))
            prefac = cdens0 * Om0 / da**2
            restab[i, :] *= prefac
            restab[i, :] /= 1e12

        self.restab = restab
        return self.restab

    def getlog(self, tag=""):
        log_dict = {
            "restab": self.restab,
            "zarr": self.zarr,
            "rarr": self.rarr,
            "tag": tag,
        }
        return log_dict

    def write(self, sname, tag=""):
        """saves the calculated thetatable"""

        log_dict = {
            "restab": self.restab,
            "zarr": self.zarr,
            "rarr": self.rarr,
            "tag": tag,
        }
        pickle.dump(log_dict, open(sname, 'wb'))

    @staticmethod
    def read(oname):
        log_dict = pickle.load(open(oname, 'rb'))
        return log_dict


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


class PowerSpec(object):
    def __init__(self, karr, parr, z, descr="", scalar_ind=0.96, h=0.7, s8=0.79):
        """
        Creates power spectrum

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
        # self.specfunc = self._s8corr(s8) # this has the desired sigma at 8 Mpc/

    def spec(self, kvals):
        """evaluates power spectrum at kvals"""
        kk = np.array(kvals)
        pp = np.array([self.specfunc(k) for k in kk])
        return pp

    # def _s8corr(self, s8=0.79):
    #     """rescales the power spectrum to the desired sigma8 at 8 Mpc/h"""
    #
    #     specfunc0 = self._specmaker(self.karr, self.parr)
    #
    #     s80 = self._sigma(8. / self.h, specfunc0)
    #
    #     fscale = (s8 / s80) ** 2.
    #
    #     return self._specmaker(self.karr, self.parr * fscale)

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


class PowerSpecContainer(object):
    def __init__(self, prefix, suffix, zvals, loc='./', scalar_ind=0.96):
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


# class PowerSpec2:
#     def __init__(self, prefix, suffix, zvals, loc='./', scalar_ind=0.96):
#         """
#         DEPRECATED
#         Loads linear matter power speclist from CAMB output
#
#         :param prefix:
#         :param suffix:
#         :param zvals:
#         :param loc:
#         """
#         self.scalar_ind = scalar_ind
#         self.zvals = np.array(zvals)
#         self.fnames = [prefix + "{:.1f}.".format(zval) + suffix
#                        for zval in zvals]
#
#         self.speclist = [np.loadtxt(name) for name in self.fnames]
#
#     def _make_interpolator(self):
#         pass
#         # interp.interp1d()
#
#     def _interpolate(self, z):
#         """
#         Interpolates power spectrum to the specified redshift
#
#         ONLY INTERPOLATES DATA points
#
#         Raises error if z is outside the redshift range
#
#         :return: power spectrum at specified redshift
#         """
#         pass
#
#     def spec(self, z, nearest=True):
#         if nearest:
#             specind = np.argmin((z - self.zvals)**2.)
#             spectra = self.speclist[specind]
#         else:
#             raise NotImplementedError
#
#         karr = spectra[:, 0]
#         parr = spectra[:, 1]
#
#         pkfunc = interp.interp1d(karr, parr)
#         lower_norm = (pkfunc(karr[1]) / karr[1]**0.96)
#         upper_norm = pkfunc(karr[-2]) / (karr[-2]**(self.scalar_ind - 4)
#                                          * np.log(karr[-2])**2)
#
#         def ufunclike(kval):
#             if (kval > karr[-2]):
#                 return kval ** (self.scalar_ind - 4) * np.log(kval)**2. \
#                        * upper_norm
#             elif (kval < karr[1]):
#                 return kval ** self.scalar_ind * lower_norm
#             else:
#                 return pkfunc(kval)
#
#         return ufunclike





