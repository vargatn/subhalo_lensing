""""
MCMC base and likelihoods
"""

import numpy as np


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
        if verbose_step is not None and i%verbose_step == 0:
            print(i)
        if covstep is not None and i > 0 and i%covstep == 0:
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

