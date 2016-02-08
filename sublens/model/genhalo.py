import time
import math
import scipy
import pickle
import numpy as np
import astropy.units as u
import multiprocessing as mp
import scipy.interpolate as interp
import scipy.integrate as integr

from sublens import default_cosmo
from ..model.autoscale import cscale_duffy


def hloader(name):
    hlog = pickle.load(open(name, "rb"))
    return hlog['names'], hlog['pars'], hlog['rvals'], hlog['profs'], hlog
    # srvals2 = slog2['rvals']


def parmaker(**kwargs):
    # gets names of parameters
    par_names = sorted(kwargs.keys())
    # gets edges for parameters
    edges = [kwargs[key] for key in par_names]

    grid_list = np.meshgrid(*edges, indexing='ij')
    flatgrid = np.array([grid.flatten() for grid in grid_list]).T

    log_dict = {
        'names': par_names,
        'pars': flatgrid,
        'pcens': kwargs,
    }
    return log_dict


class Halo(object):
    def __init__(self, component):
        self.ncomp = len(component)
        self.comp = component

        self.pnames = None
        self.pars = None
        # lis
        self.pcens = None

        self.mode = None
        self.rvals = None
        self.profs = None

    def write_log(self, sname='./profs.p'):
        ldict = {
            'comp': self.comp,
            'names': self.pnames,
            'pars': self.pars,
            'pcens': self.pcens,
            'profs': self.profs,
            'rvals': self.rvals,
            'mode': self.mode,
        }
        pickle.dump(ldict, open(sname, 'wb'))

    @staticmethod
    def read_log(lname):
        ldict = pickle.load(open(lname, 'rb'))
        # comp = ldict['comp']
        names = ldict['names']
        pars = ldict['pars']
        profs = ldict['profs']
        return names, pars, profs

    def multi_func(self, rvals, logdict, n_multi=1, mode='circ', **kwargs):
        self.pnames = logdict['names']
        self.pars = logdict['pars']
        self.pcens = logdict['pcens']

        # creating list of settings
        npar = len(self.pars)
        setlist = []
        for i in range(npar):
            settings = {'rvals': rvals,}
            for j, key in enumerate(self.pnames):
                settings.update({key: self.pars[i, j]})
            setlist.append(settings)

        # multiprocessing threads
        pool = mp.Pool(processes=n_multi)

        self.mode = mode
        self.rvals = rvals

        if mode == 'circ':
            profs = np.array(pool.map(self._ocen_ds_circ_pool, setlist))
        elif mode == "ring":
            profs = np.array(pool.map(self._ocen_ds_ring_pool, setlist))
        else:
            raise NotImplementedError

        self.profs = profs
        return self.profs

    def cen_ds_curve(self, rr, m, z=0.5, *args, **kwargs):
        # preparing halo components
        [comp.prep_ds(m=m, z=z, **kwargs) for comp in self.comp]
        # obtaining individual profiles
        dsarr = np.sum(np.array([comp.dsarr(m=m, z=z, rr=rr)[1]
                                 for comp in self.comp]), axis=0)
        return dsarr

    def _ocen_ds_ring_pool(self, settings):
        return self.alter_ring(**settings)

    # def ocen_ds_ring(self, rvals, m=1e12, z=0.5, dist=0.0, **kwargs):
    #     # area of rings between edges
    #     areas = np.array([np.pi * (rvals[i + 1] ** 2. - rvals[i] ** 2.)
    #                       for i, val in enumerate(rvals[:-1])])
    #     # preparing halo components
    #     [comp.prep_ds(m=m, z=z, **kwargs) for comp in self.comp]
    #
    #     def gfun(val):
    #         return -1. *math.pi
    #
    #     def hfun(val):
    #         return math.pi
    #
    #     ds_sum = np.zeros(shape=(len(rvals)-1, 2))
    #     for i, edge in enumerate(rvals[:-1]):
    #         ds_sum[i] = integr.dblquad(self._ds2d, rvals[i], rvals[i+1],
    #                                    gfun, hfun,
    #                                    args=(dist, self.comp),
    #                                    epsabs=1.49e-8, epsrel=1.49e-8)[0]
    #
    #     return ds_sum / areas[:, np.newaxis]

    def alter_ring(self, rvals, m=1e12, z=0.5, dist=0.0, verbose=False, split=True, deps=0.01, **kwargs):
        [comp.prep_ds(m=m, z=z, **kwargs) for comp in self.comp]

        points = None
        ds_sum = np.zeros(shape=(len(rvals)-1, 2))

        time00 = time.time()
        for i, edge in enumerate(rvals[:-1]):
            if verbose:
                print('ring ',i)
                print(rvals[i], rvals[i + 1], ' Mpc')

            time0 = time.time()
            if split * (dist < rvals[i+1]) * (dist > rvals[i]):
                if verbose:
                    print('dist = ', dist, ' Mpc')
                    print('doing the splitting')

                dssum0 = np.array(integr.quad(self._alter_rc, rvals[i], dist, args=(dist,), points=points))
                dssum1 = np.array(integr.quad(self._alter_rc, dist, rvals[i+1], args=(dist,), points=points))
                # print(dssum0 + dssum1)
                ds_sum[i] = dssum0 + dssum1
            else:
                ds_sum[i] = integr.quad(self._alter_rc, rvals[i], rvals[i+1], args=(dist,), points=points)
            # print(ds_sum[i])
            ds_sum[i] /= (rvals[i+1] - rvals[i])


            time1 = time.time()
            if verbose:
                print(time1 - time0, ' s')
        time11 = time.time()
        if verbose:
            print(time11 - time00, ' s')
        return ds_sum

    def _alter_rc(self, r, dist):
        circ = 2. * math.pi * r
        # evaluating function
        dst = integr.quad(self._ds2d, -math.pi, math.pi, args=(r, dist, self.comp), points=(0.0,))
        return dst[0] / circ

    def _ocen_ds_circ_pool(self, settings):
        return self.ocen_ds_circ(**settings)

    def ocen_ds_circ(self, rvals, m=1e12, z=0.5, dist=0.0, **kwargs):
        # calculating circumference
        circarr = 2. * math.pi * rvals
        # preparing halo components
        [comp.prep_ds(m=m, z=z, **kwargs) for comp in self.comp]

        # evaluating function
        dst_sum = np.array([integr.quad(self._ds2d, -math.pi, math.pi,
                                        args=(r, dist, self.comp), points=(0.0,)) for r in rvals])
        return dst_sum / circarr[:, np.newaxis]

    def prep_int(self, m, z, dist=0.0):
        pass

    @staticmethod
    def _ds2d(phi, r, dist, comps):
        assert r > 0.
        # creating transformation variables
        dist2 = dist * dist
        rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
        rr = math.sqrt(rr2)
        term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. - 2. * dist * math.cos(phi) - r)) / rr2
        term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2

        dst_cen = np.sum([comp._ds(rr) * r for comp in comps])

        dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
        return dst


class HaloComponent(object):
    def __init__(self, cosmo=None):
        """Parent object for single DM halo components"""
        # default cosmology
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        # setting up parameters
        # self.ifunc = lambda x: 0.0

        # self.ifunc = ifunc
        self.iinit = False

    @staticmethod
    def ifunc(x):
        return 0.0

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
        return self._nfw_shear_t(rr, self.rs, self.rho_s)

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


class NFW_gen(NFW):
    def prep_ds(self, m, z, **kwargs):
        self.rs, self.rho_s, self.r200 = self._nfw_params(self.cosmo, m,
                                                          kwargs['c'], z)
        self.iinit = True

    def dsarr(self, m, z, rr, *args, **kwargs):
        """Evaluates the \Delta\Sigma profile at the specified rr values"""
        rs, rho_s, r200 = self._nfw_params(self.cosmo, m, kwargs['c'], z)

        ds = np.array([self._nfw_shear_t(r, rs, rho_s)
                       for i, r in enumerate(rr)])
        ds /= 1e12

        return rr, ds

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
        print('pre interp')
        # creating interpolation
        ifunc = interp.interp1d(rarr, ds, kind=kind, fill_value=0.0,
                                bounds_error=False)
        print('post interp')
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