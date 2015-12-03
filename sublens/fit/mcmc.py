"""
MCMC chain an likelihood
"""

import numpy as np


class Likelihood:
    def __init__(self, profmaker, modelprof):

        # checking if everything is prepared
        assert profmaker.dst is not None
        assert profmaker.dsx is not None
        assert profmaker.dst_cov is not None
        assert profmaker.dsx_cov is not None

        assert modelprof.ref_list is not None


        self.profmaker = profmaker
        self.modelprof = modelprof

        self.dst = profmaker.dst
        self.dsx = profmaker.dsx
        self.dst_cov = profmaker.dst_cov
        self.dsx_cov = profmaker.dsx_cov

        self.cdett = np.linalg.det(self.dst_cov)
        self.cdetx = np.linalg.det(self.dsx_cov)

        self.cinvt = np.linalg.inv(self.dst_cov)
        self.cinvx = np.linalg.inv(self.dsx_cov)


    def like(self, pars):
        """calculates model based on scaling and evaluates chi2"""
        self.modelprof.scale_frame(pars)
        self.modelprof.collapse_profiles()

        model = self.modelprof.model

        delta = self.dst - model

        chi2 = 1. / 2. * np.dot(delta.T, np.dot(self.cinvt, delta))

        return chi2


class MCMC:
    def __init__(self, like):
        self.like = like



    def randomstep(self):
        ndim = len(self.usind)

        step = np.zeros(shape=self.dpar.shape)
        cov = np.diag(self.dpar[self.usind])

        step[self.usind] = np.random.multivariate_normal(np.zeros(ndim), cov)
        return step



    def run(self, par0, dpar, nstep=100, seed=5):
        """perform a Metropolis Hastings run"""

        self.nstep = nstep
        self.seed = seed
        self.dpar = np.array(dpar)

        # setting seed for rng
        np.random.seed(seed)

        self.params = np.array([par0])
        chi2_old = self.like.like(par0)

        self.chi2s = np.array([chi2_old])
        par_old = par0

        print(self.params.shape)
        # checking how many dimensions are explored

        self.usind = np.array(np.where(np.isnan(dpar) == 0)[0])

        for i in np.arange(nstep):
            step = self.randomstep()
            new_par = par_old + step

            new_chi2  = self.like.like(new_par)

            ref = np.random.uniform()

            if np.exp(-1. * (new_chi2 - chi2_old)) > ref:
                par_old = new_par
                chi2_old = new_chi2



            self.chi2s = np.concatenate((self.chi2s, [chi2_old]))
            self.params = np.vstack((self.params, new_par))

        return self.params, self.chi2s








