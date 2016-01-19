import time
import math
import scipy
import numpy as np
import astropy.units as u
import scipy.interpolate as interp
import scipy.integrate as integr

# FIXME document this module

from ..model.misc import default_cosmo
from ..model.autoscale import cscale_duffy


class SuperHalo(object):
    def __init__(self, pscont, cscale=cscale_duffy, cosmo=None):

        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.
        self.cscale = cscale

        # primary halo object
        self.ph = NFW(cosmo)

        # secondary halo object
        self.sh = TwoHalo(pscont, cosmo=cosmo)

    def cen_ds_curve(self, m, z, rr):
        assert z > 0.05, 'Two halo term only works for z > 0'

        rarr, ds2 = self.sh.ds(m, z, rr)
        rarr, ds1 = self.ph.ds(m, z, rr)

        return rr, ds1 + ds2

    def ocen_ds_ring(self, edges, dist, m, z, interp_grid="default"):
        """integrates the ds at a ring between the edges"""
        # getting nfw parameters
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.ph.nfw_params(m, c, z)

        t0 = time.time()
        # creating 2-halo parameters
        if interp_grid == "default":
            interp_grid = np.logspace(-3, 2, num=50)
        ifunc = self.sh.interp_ds(m, z, interp_grid)
        t1 = time.time()
        print(t1 - t0, ' s')

        # area of rings between edges
        areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                          for i, val in enumerate(edges[:-1])])

        # boundary functions for the rings
        def gfun(val):
            return 0.

        def hfun(val):
            return 2 * math.pi

        t0 = time.time()
        # calculating the integral
        ds_sum = np.zeros(shape=(len(edges)-1, 2))

        for i, edge in enumerate(edges[:-1]):
            print(i)
            ds_sum[i] = integr.dblquad(self._ds2d, edges[i], edges[i+1],
                                       gfun, hfun,
                                       args=(dist, ifunc, rs, rho_s),
                                       epsabs=1.49e-4)[0]
        t1 = time.time()
        print(t1 - t0, ' s')

        return ds_sum / areas[:, np.newaxis]

    def ocen_ds_circ(self, rarr, dist, m, z, interp_grid="default" ):
        """integrates the ds at a ring at r"""
        # getting nfw parameters
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.ph.nfw_params(m, c, z)

        t0 = time.time()
        # creating 2-halo parameters
        if interp_grid == "default":
            interp_grid = np.logspace(-3, 2, num=50)
        ifunc = self.sh.interp_ds(m, z, interp_grid)
        t1 = time.time()
        print(t1 - t0, ' s')

        # calculating circumference
        circarr = 2. * math.pi * rarr

        t0 = time.time()
        # calculating the integral
        dst_sum = np.array([integr.quad(self._ds2d, -math.pi, math.pi,
                                        args=(r, dist, ifunc, rs, rho_s)) for r in rarr])
        t1 = time.time()
        print(t1 - t0, ' s')

        return dst_sum / circarr[:, np.newaxis]

    def ocen_ds(self, pararr, dist, m, z, interp_grid="default"):
        """simple diagnostic function"""

        # getting nfw parameters
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.ph.nfw_params(m, c, z)

        t0 = time.time()
        # creating 2-halo parameters
        if interp_grid == "default":
            interp_grid = np.logspace(-3, 2, num=50)
        ifunc = self.sh.interp_ds(m, z, interp_grid)
        t1 = time.time()
        print(t1 - t0, ' s')

        t0 = time.time()
        dstarr = np.array([self._ds2d(phi, r, dist, ifunc, rs, rho_s) for (phi, r) in pararr])
        t1 = time.time()
        print(t1 - t0, ' s')

        return dstarr

    def _ds2d(self, phi, r, dist, ifunc, rs, rho_s):
        assert r > 0.
        # creating transformation variables
        dist2 = dist * dist
        rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
        rr = math.sqrt(rr2)
        term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. - 2. * dist * math.cos(phi) - r)) / rr2
        term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2

        # creating centered profile at distance r
        ds_p = self.ph.nfw_shear_t(rr, rs, rho_s) / 1e12 * r  # 1-halo
        ds_s = ifunc(rr) * r  # 2-halo
        dst_cen = ds_p + ds_s

        dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
        return dst


class NFW(object):
    def __init__(self, cosmo=None, cscale=cscale_duffy):
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.
        self.cscale = cscale

    def nfw_params(self, m200, c200, z):
        m200 *= u.solMass

        cdens = self.cosmo.critical_density(z).to(u.solMass / u.Mpc**3)
        r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
        rs = r200 / c200
        dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200) - c200 / (1. + c200))
        rho_s = dc * cdens

        rs = rs.to(u.Mpc)
        r200 = r200.to(u.Mpc)

        return rs.value, rho_s.value, r200.value

    def ds_point(self, m , z, r):
        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.nfw_params(m, c, z)

        ds = self.nfw_shear_t(r, rs, rho_s) / 1e12

        return r, ds

    def ds(self, m, z, rr):

        c = self.cscale(m / self.h, z)
        rs, rho_s, r200 = self.nfw_params(m, c, z)

        ds = np.array([self.nfw_shear_t(r, rs, rho_s)
                   for i, r in enumerate(rr)])

        ds /= 1e12

        return rr, ds

    @staticmethod
    def nfw_shear_t(r, rs, rho_s):
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


class TwoHalo:
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

        self.ifunc = None

    def ds(self, m, z, rr, mode='nearest'):
        """Calculates deltasigma profile for the 2-halo term"""
        ps = self.pscont.getspec(z, mode=mode)
        mb = MatterBias(ps)
        bias = mb.bias(m, z)

        resarr = self.w2i.getw2(rr, z)
        return rr, resarr[1] * bias

    def interp_ds(self, m, z, rr, kind="linear"):
        """creates interpolator function for the profile based on rr evals"""
        rarr, ds = self.ds(m, z, rr)
        ifunc = interp.interp1d(rarr, ds, kind=kind, fill_value=0.0,
                                bounds_error=False)

        self.ifunc = ifunc
        return self.ifunc


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