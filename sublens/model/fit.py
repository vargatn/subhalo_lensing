""""
MCMC base and likelihoods
"""

import numpy as np

# TODO implement this module


class LikelihoodBase(object):
    def __init__(self, mode='point'):
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def get_like(self, *args, **kwargs):
        raise NotImplementedError


class SimpleLikelihood(LikelihoodBase):
    """
    The old fashioned single bin likelihood
    """
    def __init__(self, profobj, obs_profile, mode='point'):
        super().__init__(mode=mode)

        self.profobj = profobj
        self.obs_profile = obs_profile
        self.requires = profobj.requires

        self.rr = self.obs_profile['rr']
        self.mvec = None

    def get_like(self, mode="point", **kwargs):
        self.profobj.prepare(**kwargs)

        if mode == "edge":
            rvals = self.obs_profile['r_edges']
            self.profobj.rbin_deltasigma(rvals)
        else:
            rvals = self.obs_profile['rr']
            self.profobj.deltasigma(rvals)
        # print(rvals)

        self.mvec = self.profobj.ds

        dvec = self.obs_profile['ds']
        cov = self.obs_profile['cov']
        diff = (dvec - self.mvec)
        cinv = np.linalg.inv(cov)
        chisq = float(np.dot(diff.T, np.dot(cinv, diff)))
        return chisq


class EnsembleLikelihood(LikelihoodBase):
    def __init__(self):
        super().__init__()
