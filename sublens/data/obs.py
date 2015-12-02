"""
Module to create the observed side of the Modelling likelihood
"""

import math
import numpy as np
from astropy.io import fits
import pickle
import pandas as pd
import kmeans_radec as krd
import bisect as bs



class ProfileMaker:
    """
    Calculates measured shear profile based on the specified subpatches
    """
    def __init__(self, shear_data, *args):
        self.shear_data = shear_data
        self.subs = args
        self.aid = self.all_id() # union of all sub.ids
        self.reid = self.re_ids()

        self.km = None

    def all_id(self):
        """Calculates the union of all subpatches used"""
        aid = set([])
        for sub in self.subs:
            aid = aid.union(set(sub.ids))

        aid = np.array(list(aid))
        return aid

    def re_ids(self):
        """returns list of arrays which indexes aid for the subs"""
        reid = []
        argind = np.argsort(self.aid)
        for sub in self.subs:
            tmp_ids = np.searchsorted(self.aid, sub.ids, sorter=argind)
            reid.append(self.aid[argind][tmp_ids])
        return  reid

    def radec_patches(self, ncen=100, verbose=False):
        """Calculates the specified number of k-means patches on the sky"""
        X = np.vstack((self.shear_data.ra[self.aid],
                       self.shear_data.dec[self.aid])).T
        nsample = X.shape[0] // 2
        km = krd.kmeans_sample(X, ncen=ncen, nsample=nsample, verbose=verbose)

        if not km.converged:
            km.run(X, maxiter=100)

        self.km = km

    def jack_profiles(self):
        pass


class ObsSpace:
    """
    Parameter space for the directly observed parameters

    Contents:
    ----------
    - tracers: list of names of parameter names
    - tdata: data table of tracers in the order of their names
    - ID: id numbers indexing the shear profile data table in ShearData

    - subpatcher
    """
    def __init__(self, tracers, ids, data):
        self.tracers = tracers
        self.data = data
        self.ids = ids
        self.par_ranges = self.pedges()

    @classmethod
    def from_data(cls, shear_data, **kwargs):
        """
        Constructs ObsSpace using existing ShearData

        :param shear_data: instance of ShearData
        :param kwargs: {param_type: key_in_cat}
        """
        tracers = sorted(kwargs.keys())
        assert len(set(tracers)) == len(tracers)

        ids = shear_data.info[:, 0].astype(int
                                           )
        data = np.array([shear_data.cat[kwargs[key]] for key in tracers]).T

        return cls(tracers, ids, data)

    def pedges(self):
        """
        Calculates the cubic edges of the parameter space
        """
        par_ranges = {}
        for i, key in enumerate(self.tracers):
            par_ranges.update({key: (np.min(self.data[:, i]),
                                    np.max(self.data[:, i]))})

        self.par_ranges = par_ranges

    def subpatcher(self, **kwargs):
        """
        Creates a subpatch instance of the current ObsSpace
        :param kwargs: {param_type: (min, max)} None is -inf
        :return: instance of the same class with the shrinked pspace
        """
        params = sorted(kwargs.keys())

        assert set(params) <= set(self.tracers)

        indtab = np.ones(len(self.data)).astype(bool)

        for i, key in enumerate(self.tracers):
            if key in params:
                indtab *= (self.data[:, i] > kwargs[key][0]) *\
                          (self.data[:, i] < kwargs[key][1])

        newdata = self.data[indtab, :]
        newids = self.ids[indtab]

        return ObsSpace(self.tracers, newids, newdata)

