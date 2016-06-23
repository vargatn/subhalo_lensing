""""
MCMC base and likelihoods
"""

import numpy as np


def mcmc(like, names, par0, stepsize, limits, extra_pars=None, nstep=10,
         seed=None, rng=None, verbose_step=None, covstep=None, **kwargs):

    """
    Simple single walker MCMC function with fixed number of steps (no stopping
    criteria), but with adaptive step size

    :param like: Likelihood object

    :param names: list of parameter names

    :param par0: initial values of parameters (list)

    :param stepsize: initial stepsize for parameters

    :param limits: hard parameter interval edges

    :param extra_pars: dict of additional parameters to pass to likelihood

    :param nstep: number of steps to make

    :param seed: random seed to use

    :param rng: RNG instance to use, default is np.random

    :param verbose_step: step numbers to write (multiples of this)

    :param covstep: number of steps between refreshing the cov matrix

    :param kwargs: additional parameters to pass to likelihood

    :return: (parameter_values, chi2_values, deltasigma, other_deltasigmas)
    """
    if rng is None:
        rng = np.random.RandomState(seed=seed)

    if extra_pars is None:
        extra_pars = dict({})

    parvec0 = np.array(par0)
    parvec = np.array([parvec0])

    dpar = dict(zip(names, par0))
    dpar.update(extra_pars)
    chi = like.get_like(**dpar)
    chivec = np.array([chi])

    ds_other_cont = []
    ds_other = like.mvec_others
    ds_other_cont.append(ds_other)
    dsvec = np.array([like.mvec])

    stepsize = np.array(stepsize.copy())
    # step covariance matrix
    if len(stepsize) == len(par0) and len(stepsize.shape) == 1:
        stepmatr = np.diag(stepsize)
    else:
        stepmatr = stepsize

    # going through the steps
    for i in np.arange(nstep):
        if verbose_step is not None and i % verbose_step == 0:
            print(i)
        if covstep is not None and i > 0 and i % covstep == 0:
            stepmatr = np.cov(parvec.T)
        ds = dsvec[-1]
        par = parvec[-1]

        # making random step
        dpar = rng.multivariate_normal(mean=np.zeros(par.shape), cov=stepmatr)
        parnew = par + dpar

        # checking for parameter ranges
        for i, name in enumerate(names):
            if (parnew[i] < limits[i][0]) | (parnew[i] > limits[i][1]):
                parnew = par

        # calculating chi2 value for new position
        dpar = dict(zip(names, parnew))
        dpar.update(extra_pars)
        chinew = like.get_like(**dpar)
        dsnew = like.mvec
        # obtaining reference random number
        ref = rng.uniform()
        #
        # checking for accepting new step
        if np.exp(-1. * (chinew - chi)) > ref:
            par = parnew
            chi = chinew
            ds = dsnew
            ds_other = like.mvec_others

        # saving data
        parvec = np.vstack((parvec, par))
        chivec = np.vstack((chivec, chi))
        dsvec = np.vstack((dsvec, ds))
        ds_other_cont.append(ds_other)

    return parvec, chivec, dsvec, ds_other_cont


class LikelihoodBase(object):
    """Base class for all Likelihoods"""
    def __init__(self, *args, **kwargs):
        self.mvec_others = None
        pass

    def get_like(self, *args, **kwargs):
        raise NotImplementedError


class SimpleLikelihood(LikelihoodBase):
    """
    The old fashioned example likelihood
    """
    def __init__(self, profobj, obs_profile, mode="point", fit_range=None):
        """
        Example likelihood with a single component

        :param profobj: DeltaSigmaProfile object instance

        :param obs_profile: dictionary of observed profile

        :param mode: "point" or "r_edges"

        :param fit_range: radial range to use for the fit
        """
        super().__init__()
        self.mode = mode

        self.profobj = profobj
        self.obs_profile = obs_profile
        self.requires = profobj.requires

        self.rr = self.obs_profile['rr']
        self.mvec = np.zeros(shape=self.rr.shape)
        self.fit_range = fit_range

    def set_mode(self, mode):
        """set prediction mode"""
        self.mode = mode

    def get_like(self, **kwargs):
        """Evaluates likelihood. Parameters should be specified bz keywords"""
        self.profobj.prepare(**kwargs)

        if self.mode == "edge":
            rvals = self.obs_profile['r_edges']
            self.profobj.rbin_deltasigma(rvals)
        else:
            rvals = self.obs_profile['rr']
            self.profobj.deltasigma(rvals)

        if self.fit_range is None:
            self.fit_range = (0., np.inf)
        index = np.where((self.fit_range[0] <= self.obs_profile['rr']) *
                         (self.fit_range[1] > self.obs_profile['rr']))[0]

        self.mvec = self.profobj.ds
        dvec = self.obs_profile['ds'][index]

        cov = self.obs_profile['cov'][index, :][:, index]
        diff = (dvec - self.mvec[index])
        cinv = np.linalg.inv(cov)
        chisq = float(np.dot(diff.T, np.dot(cinv, diff)))
        return chisq

# TODO document this properly
class EnsembleLikelihood(LikelihoodBase):
    def __init__(self, mappers, tables, obs_profile, obs_params, etol=5e-3,
                 fit_range=None):
        """
        To be used with profiles from lookuptables

        :param mappers:

        :param tables:

        :param obs_profile:

        :param obs_params:

        :param etol:
        """
        super().__init__()

        self.fit_range = fit_range

        self.mappers = mappers
        self.tables = tables
        if len(self.mappers) != len(self.tables):
            raise TypeError('mappers and tables must be iterable!!')

        self.ncomponents = len(mappers)

        self.obs_profile = obs_profile
        self.obs_params = obs_params

        self.rr = self.obs_profile['rr']
        self.mvec = np.zeros(shape=self.rr.shape)

        self.model_pars = (None,) * len(self.mappers)
        self.model_ds = np.zeros(shape=(len(self.mappers), len(self.rr)))

        # checking that the radial values are the same for all tables
        for tab in self.tables:
            if np.max(np.abs(tab.rr - self.tables[0].rr)) > etol:
                raise ValueError('radial values do not agree between the' +
                                 ' tables')

        # checking that the radial values are the same for the observed prof
        # and the tables
        if np.max(np.abs(self.obs_profile['rr'] - self.tables[0].rr)) > etol:
            raise ValueError('radial values do not agree between profile' +
                             ' and tables')

        self.req_params = set([])
        [self.req_params.update(set(mpp.requires)) for mpp in self.mappers]
        self.req_params = sorted(self.req_params)

        if set(self.req_params) != set(self.obs_params['par_names']):
            raise ValueError('observed parmeters are incompatible with' +
                             'the mappers')

        # check required paramters of the mappers
        if set(self.req_params) <= set(obs_params):
            raise ValueError('some parameters are missing for the mappers')

        # checking that at the required parameters can be provided by
        #  the mappers
        for mpp, tbl in zip (self.mappers, self.tables):
            if set(mpp.provides) != set(tbl.haspars):
                raise ValueError('tables require unspecified paramters')

        self.meta_pivots = [mpp.pivots0.keys() for mpp in self.mappers]
        self.meta_exps = [mpp.exponents0.keys() for mpp in self.mappers]

    def reset_mappers(self):
        [mpp.reset() for mpp in self.mappers]

    def get_like(self, **kwargs):

        rcens = self.obs_profile0['rr']
        if self.fit_range is None:
            self.fit_range = (0., np.inf)
        index = np.where((self.fit_range[0] <= rcens) *
                         (self.fit_range[1] > rcens))[0]

        # updates all mappers
        for mpp in self.mappers:
            mpp.update(**kwargs)

        # gets model parameters
        self.model_pars = []
        for mpp in self.mappers:
            mpars = mpp.convert(self.obs_params['par_values'],
                                parnames=self.obs_params['par_names'])
            self.model_pars.append(mpars)

        self.mvec = np.zeros(shape=self.rr.shape)

        # gets prediciton from all tables
        self.model_ds = []
        for i, tab in enumerate(self.tables):
            ds = tab.combine_profile(sample=self.model_pars[i],
                                     parnames=self.mappers[i].provides)[1]
            self.model_ds.append(ds)

        self.model_ds = np.array(self.model_ds)
        self.mvec = np.sum(self.model_ds, axis=0)

        dvec = self.obs_profile['ds']
        cov = self.obs_profile['cov']
        diff = (dvec - self.mvec)
        cinv = np.linalg.inv(cov)
        chisq = float(np.dot(diff.T, np.dot(cinv, diff)))
        return chisq


class SuperCompositeHaloLikelihood(LikelihoodBase):
    """
    Some frankenstein monster of a likelihood... Aaaaargh...
    """
    def __init__(self, profobj, table, obs_params, obs_profile0, obs_profile1,
                 mode='point', fit_range=None):
        """Uses two profiles, one calculated, and the other from a lookup"""
        super().__init__()
        self.mode = mode

        self.profobj = profobj
        self.table = table
        self.obs_params = obs_params
        self.obs_profile0 = obs_profile0
        self.obs_profile1 = obs_profile1

        self.rr = self.obs_profile0['rr']
        self.mvec0 = np.zeros(shape=self.rr.shape)
        self.mvec1 = np.zeros(shape=self.rr.shape)
        self.mvec2 = np.zeros(shape=self.rr.shape)

        self.mvec = np.zeros(shape=self.rr.shape)
        self.fit_range = fit_range

        self.zind = np.where('z'==np.array(self.obs_params['par_names']))[0]
        self.zmean = np.mean(self.obs_params['par_values'][:, self.zind])

        self.distind = np.where('dist' == np.array(
            self.obs_params['par_names']))[0]

        self.values = self.obs_params['par_values']

    def get_like(self, msub, csub, mparent, d0=0.0):

        redges = self.obs_profile0['r_edges']
        rcens = self.obs_profile0['rr']

        if self.fit_range is None:
            self.fit_range = (0., np.inf)
        index = np.where((self.fit_range[0] <= rcens) *
                         (self.fit_range[1] > rcens))[0]

        # getting first likelihood
        self.profobj.prepare(m200c=msub, c200c=csub, z=self.zmean)

        self.profobj.rbin_deltasigma(redges)
        self.mvec0 = self.profobj.ds

        mparnames = ['dist', 'm200c', 'z']

        massarr = np.ones(len(self.values)) * mparent
        zarr = np.ones(len(self.values)) * self.zmean
        darr = (self.values[:, self.distind]/ 0.7) - d0
        mpars = np.hstack((darr, massarr[:, np.newaxis], zarr[:, np.newaxis]))
        mrr, self.mvec1 = self.table.combine_profile(mpars,
                                                     mparnames, checknan=True)

        self.mvec = self.mvec0 + self.mvec1

        dvec = self.obs_profile0['ds'][index]

        cov = self.obs_profile0['cov'][index, :][:, index]
        diff = (dvec - self.mvec[index])
        cinv = np.linalg.inv(cov)
        chisq0 = float(np.dot(diff.T, np.dot(cinv, diff)))

        # getting second likelihood
        massarr = np.ones(len(self.values)) * mparent
        zarr = np.ones(len(self.values)) * self.zmean
        darr = (self.values[:, self.distind]/ 0.7) + d0
        mpars = np.hstack((darr, massarr[:, np.newaxis], zarr[:, np.newaxis]))
        mrr, self.mvec2 = self.table.combine_profile(mpars,
                                                     mparnames, checknan=True)

        self.mvec_others = [self.mvec0, self.mvec1, self.mvec2]

        dvec = self.obs_profile1['ds'][index]

        cov = self.obs_profile1['cov'][index, :][:, index]
        diff = (dvec - self.mvec2[index])
        cinv = np.linalg.inv(cov)
        chisq1 = float(np.dot(diff.T, np.dot(cinv, diff)))

        return chisq0 + chisq1

