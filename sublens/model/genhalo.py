import time
import math
import scipy
import numpy as np
import astropy.units as u
import scipy.interpolate as interp
import scipy.integrate as integr

from sublens import default_cosmo
from ..model.autoscale import cscale_duffy


class Halo(object):
    def __init(self, component):
        self.ncomp = len(component)
        self.comp = component

        self.m = None
        # self.

    def cen_ds_curve(self, m, z, rr):
        pass

    def ocen_ds_ring(self):
        pass

    def ocen_ds_circ(self):
        pass

    def prep_int(self, m, z, dist=0.0):
        pass

    def _ds2d(self, phi, r, dist, rs, rho_s):
        assert r > 0.
        # creating transformation variables
        dist2 = dist * dist
        rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
        rr = math.sqrt(rr2)
        term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. - 2. * dist * math.cos(phi) - r)) / rr2
        term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2



class HaloComponent(object):
    def __init__(self, cosmo=None):
        """Parent object for single DM halo components"""
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        # setting up parameters
        self.ifunc = lambda x: 0.0
        self.iinit = False
        
    def prep_ds(self, *args, **kwargs):
        self.iinit = True

    def _ds(self, *args, **kwargs):
        pass

    def dsarr(self, *args, **kwargs):
        pass


class NFW(HaloComponent):
    def __init__(self, cosmo=None, cscale=cscale_duffy):
        super(NFW, self).__init__(cosmo=cosmo)

        self.h = self.cosmo.H0.value / 100.
        self.cscale = cscale

        self.m = None
        self.z = None
        self.c = None
        self.rs = None
        self.rho_s = None
        self.r200 = None

    def prep_ds(self, m, z, **kwargs):
        self.c = self.cscale(m / self.h, z)
        self.rs, self.rho_s, self.r200 = self._nfw_params(self.cosmo, m,
                                                          self.c, z)
        self.iinit = True

    def _ds(self, rr):
        """Intended for point evaluations"""
        assert self.iinit, 'profile not initiated'
        return self._nfw_shear_t(rr, self.rs, self.rho_s) / 1e12

    def dsarr(self, m, z, rr, *args, **kwargs):
        """Evaluates the \Delta\Sigma profile at the specified rr values"""
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self._nfw_params(self.cosmo, m, c, z)

        ds = np.array([self._nfw_shear_t(r, rs, rho_s)
                       for i, r in enumerate(rr)])
        ds /= 1e12

        return rr, ds

    @staticmethod
    def _nfw_params(cosmo, m200, c200, z):
        """Calculates the parameters of an NFW halo based on the given cosmo"""
        m200 *= u.solMass

        cdens = cosmo.critical_density(z).to(u.solMass / u.Mpc**3)
        r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
        rs = r200 / c200
        dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200) - c200 / (1. + c200))
        rho_s = dc * cdens

        rs = rs.to(u.Mpc)
        r200 = r200.to(u.Mpc)

        return rs.value, rho_s.value, r200.value

    @staticmethod
    def _nfw_shear_t(r, rs, rho_s):
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
        return shear


class TwoHalo(HaloComponent):
    def __init__(self, pscont, cosmo=None):
        super(TwoHalo, self).__init__(cosmo=cosmo)
        self.pscont = pscont
        self.w2i = W2int(pscont, cosmo=cosmo)

        self.m = None
        self.z = None
        self.rr = None
        self.ds = None

    def prep_ds(self, m, z, rr="default", kind="linear", **kwargs):
        """creates interpolator function for the profile based on rr evals"""

        if rr == "default":
            rr = np.logspace(-3, 2, num=50)
        rarr, ds = self.dsarr(m, z, rr)

        # saving prepped parameters
        self.m = m
        self.z = z
        self.rr = rr
        self.ds = ds

        # creating interpolation
        ifunc = interp.interp1d(rarr, ds, kind=kind, fill_value=0.0,
                                bounds_error=False)
        self.ifunc = ifunc
        self.iinit = True

    def _ds(self, r, *args, **kwargs):
        assert self.iinit, 'interpolating func un-initialized'
        return self.ifunc(r)

    def dsarr(self, m, z, rr, mode='nearest'):
        """Calculates deltasigma profile for the 2-halo term"""
        ps = self.pscont.getspec(z, mode=mode)
        mb = MatterBias(ps)
        bias = mb.bias(m, z)

        resarr = self.w2i.getw2(rr, z)
        return rr, resarr[1] * bias


def get_edges(mn=0.02, mx=2.0, bins=14):
    """
    Calculates logarithmic bin edges, bin centers and bin areas

    :param mn: lower included limit
    :param mx: upper included limit
    :param bins: number of bins
    """
    edges = np.logspace(np.log10(mn), np.log10(mx), bins + 1, endpoint=True)
    mids = edges[:-1] + np.diff(edges) / 2.

    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, val in enumerate(edges[:-1])])

    return mids, edges, areas


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