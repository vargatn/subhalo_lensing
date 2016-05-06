""""
MCMC base and likelihoods
"""

import numpy as np

# TODO implement this module


class LikelihoodBase(object):
    def __init__(self, *args, **kwargs):
        self.mvec_others = None
        pass

    def get_like(self, *args, **kwargs):
        raise NotImplementedError


class SimpleLikelihood(LikelihoodBase):
    """
    The old fashioned single bin likelihood
    """
    def __init__(self, profobj, obs_profile, mode='point', fit_range=None):
        super().__init__()
        self.mode = mode

        self.profobj = profobj
        self.obs_profile = obs_profile
        self.requires = profobj.requires

        self.rr = self.obs_profile['rr']
        self.mvec = np.zeros(shape=self.rr.shape)
        self.fit_range = fit_range

    def set_mode(self, mode):
        self.mode = mode

    def get_like(self, **kwargs):
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
        # print(index)
        # print(index.shape)
        self.mvec = self.profobj.ds
        dvec = self.obs_profile['ds'][index]
        # print(dvec)
        cov = self.obs_profile['cov'][index, :][:, index]
        diff = (dvec - self.mvec[index])
        cinv = np.linalg.inv(cov)
        chisq = float(np.dot(diff.T, np.dot(cinv, diff)))
        return chisq


# class CombinedLikelihood(LikelihoodBase):
#     def __init__(self, profobj, table, ):


class EnsembleLikelihood(LikelihoodBase):
    def __init__(self, mappers, tables, obs_profile, obs_params, etol=5e-3):
        super().__init__()

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


def mcmc(like, names, par0, stepsize, limits, extra_pars=None, nstep=10,
         seed=None, rng=None, verbose_step=None, **kwargs):
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

    # step covariance matrix
    stepmatr = np.diag(stepsize)

    # going through the steps
    for i in np.arange(nstep):
        if verbose_step is not None and i%verbose_step == 0:
            print(i)
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
        # checkong for accepting new step
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


def bgetter(res):
    return res[2][np.argmin(res[1])]



