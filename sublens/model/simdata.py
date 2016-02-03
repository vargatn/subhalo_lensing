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




def snoisemaker(dvec, areas, nlevel=1.0, seed=13141):
    """Creates shape noise dominated diagonal covarainace matrix"""
    if seed is not None:
        np.random.seed(seed)
    covm = np.eye(15) / areas * nlevel
    err = np.sqrt(np.diag(covm))
    offset = np.random.multivariate_normal(np.zeros(shape=err.shape), covm)
    return dvec + offset, err, covm

    # covm =

# class simprof(object):
#     def __init__(self, modeldict):
#
#     def _reweight(self):
#         pass
#
#     def _bindata(self, edges):
#         pass

    # def _
