""""
MCMC base and likelihoods
"""

import numpy as np

# TODO implement this module


class LikelihoodBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def get_like(self, *args, **kwargs):
        raise NotImplementedError


class SimpleLikelihood(LikelihoodBase):
    """
    The old fashioned single bin likelihood
    """
    def __init__(self, profobj, obs_profile, mode='point'):
        super().__init__()
        self.mode = mode

        self.profobj = profobj
        self.obs_profile = obs_profile
        self.requires = profobj.requires

        self.rr = self.obs_profile['rr']
        self.mvec = None

    def set_mode(self, mode):
        self.mode = mode

    def get_like(self, mode="point", **kwargs):
        self.profobj.prepare(**kwargs)

        if mode == "edge":
            rvals = self.obs_profile['r_edges']
            self.profobj.rbin_deltasigma(rvals)
        else:
            rvals = self.obs_profile['rr']
            self.profobj.deltasigma(rvals)

        self.mvec = self.profobj.ds

        dvec = self.obs_profile['ds']
        cov = self.obs_profile['cov']
        diff = (dvec - self.mvec)
        cinv = np.linalg.inv(cov)
        chisq = float(np.dot(diff.T, np.dot(cinv, diff)))
        return chisq


class EnsembleLikelihood(LikelihoodBase):
    def __init__(self, mappers, tables, obs_profile, obs_params):
        super().__init__()

        self.mappers = mappers
        self.tables = tables
        if len(self.mappers) == len(self.tables):
            raise TypeError('mappers and tables must be iterable!!')

        self.ncomponents = len(mappers)

        self.obs_profile = obs_profile
        self.obs_params = obs_params

        # checking that the radial values are the same for all tables
        for tab in self.tables:
            if np.abs(tab.rr - self.tables[0].rr) > 1e-5:
                raise ValueError('radial values do not agree between the' +
                                 ' tables')

        # checking that the radial values are the same for the observed prof
        # and the tables
        if np.abs(self.obs_profile['rr'] - self.tables[0].rr) > 1e-5:
            raise ValueError('radial values do not agree between profile' +
                             ' and tables')

        self.req_params = sorted(set(np.array([mpp.requires
                                               for mpp in self.mappers]
                                              ).flatten()))
        # check required paramters of the mappers
        if set(self.req_params) <= set(obs_params):
            raise ValueError('some parameters are missing for the mappers')

        # checking that at the required parameters can be provided by
        #  the mappers
        # if set(self.mappers.provides)




