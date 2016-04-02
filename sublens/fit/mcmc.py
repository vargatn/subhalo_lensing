"""
MCMC chain an likelihood
"""

import numpy as np


def llike(hh, rvals, dvec, dcov, mode='cen', range=None, **kwargs):
    """
    Evaluates log-likelihood

    Uses centered non-bin-averaged DeltaSigma profile

    :param hh: Halo object
    :param rvals: radius values to use
    :param dvec: data vector
    :param dcov: data covariance matrix
    :param kwargs: arguements passed to
    :return: chi2 value
    """

    if mode == 'cen':
        model = hh.cen_ds_curve(rvals, **kwargs)
    else:
        raise NotImplementedError

    diff = (dvec - model)

    if range is None:
        range = (0., np.inf)
    index = np.where((range[0] <= rvals) * (range[1] > rvals))

    cinv = np.linalg.inv(dcov[index, index])
    chisq = float(np.dot(diff[index].T, np.dot(cinv, diff[index])))

    return chisq, model



