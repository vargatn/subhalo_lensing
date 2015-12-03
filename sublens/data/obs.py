"""
Module to create the observed side of the Modelling likelihood
"""

import numpy as np
import kmeans_radec as krd

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

        self.dst = None
        self.dsx = None

        self.dst_cov = None
        self.dsx_cov = None

    def all_id(self):
        """Calculates the union of all subpatches used"""
        aid = set([])
        for sub in self.subs:
            aid = aid.union(set(sub.ids))

        aid = np.sort(list(aid))
        return aid

    def re_ids(self):
        """
        returns list of arrays which indexes km.labels so that
        km.labels[reid[0]] gives the appropriate labels for sub0
        """
        reid = []
        for sub in self.subs:
            tmp_ids = np.searchsorted(self.aid, sub.ids)
            reid.append(tmp_ids)
        return reid

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
        """
        Calculates jackknife estimate on the datat profiles and cov matrix
        """
        assert self.km is not None

        ncen = len(self.km.centers)
        nsub = len(self.subs)
        dlen = len(self.shear_data.cens)

        # length of data vector = dlen * nsub
        dvec = np.zeros(dlen * nsub)
        # indexing array which check which subpatch is to be used:
        dvind = np.array([np.ones(dlen) * i for i in range(nsub)]).flatten()

        # size of cov matrix = (len(dvec), len(dvec)
        cov = np.zeros(shape=(len(dvec), len(dvec)))

        # creating container for data estimates
        dst_est = np.zeros((ncen, len(dvind)))
        dsx_est = np.zeros((ncen, len(dvind)))

        dsum_jack = np.zeros(shape=dvec.shape)
        dsensum_jack = np.zeros(shape=dvec.shape)
        osum_jack = np.zeros(shape=dvec.shape)
        osensum_jack = np.zeros(shape=dvec.shape)

        # calculating profiles for kmeans subpatches
        for i, lab in enumerate(set(self.km.labels)):
            for j, sub in enumerate(self.subs):
                ind = np.where(self.km.labels[self.reid[j]] != lab)[0]
                patch_ind = np.where(dvind == j)[0]

                dsum_jack[patch_ind] = np.sum(self.shear_data.data[3, ind, :], axis=0)
                dsensum_jack[patch_ind] = np.sum(self.shear_data.data[5, ind, :], axis=0)
                osum_jack[patch_ind] = np.sum(self.shear_data.data[4, ind, :], axis=0)
                osensum_jack[patch_ind] = np.sum(self.shear_data.data[6, ind, :], axis=0)

            dst_est[i, :] = dsum_jack / dsensum_jack
            dsx_est[i, :] = osum_jack / osensum_jack

        dst = np.mean(dst_est, axis=0)
        dsx = np.mean(dsx_est, axis=0)

        dst_cov = np.array([[np.sum((dst_est[:, i] - dst[i]) * (dst_est[:, j] - dst[j])) for j in range(len(dvec))] for i in range(len(dvec))])
        dst_cov *= (ncen - 1.) / ncen

        dsx_cov = np.array([[np.sum((dsx_est[:, i] - dsx[i]) * (dsx_est[:, j] - dsx[j])) for j in range(len(dvec))] for i in range(len(dvec))])
        dsx_cov *= (ncen - 1.) / ncen

        self.dst = dst
        self.dsx = dsx

        self.dst_cov = dst_cov
        self.dsx_cov = dsx_cov

        return dst, dst_cov, dsx, dsx_cov

    def simple_profiles(self):
        """reformats the profiles into a readily plottable version"""
        assert self.dst is not None
        assert self.dsx is not None

        assert self.dst_cov is not None
        assert self.dsx_cov is not None

        assert len(self.dst) == len(self.subs) * len(self.shear_data.cens)
        arr_shape = (len(self.subs), len(self.shear_data.cens))

        dst_arr = self.dst.reshape(arr_shape)
        dsx_arr = self.dsx.reshape(arr_shape)

        dst_err = np.sqrt(np.diag(self.dst_cov).reshape(arr_shape))
        dsx_err = np.sqrt(np.diag(self.dsx_cov).reshape(arr_shape))

        return dst_arr, dst_err, dsx_arr, dsx_err


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

        # self.par_ranges = par_ranges
        return par_ranges

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


