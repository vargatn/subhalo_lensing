"""
Module to create the observed side of the Modelling likelihood
"""

import numpy as np
import kmeans_radec as krd

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


    def forget(self, *args):
        """Forgets the specified tracers"""

        tracers = np.asanyarray(self.tracers)
        keepind = set(range(len(tracers)))
        for arg in args:
            keepind = keepind.intersection(set(np.where(arg != tracers)[0]))

        keepind = list(keepind)
        newdata = self.data[:, keepind]
        return ObsSpace(list(tracers[keepind]), self.ids, newdata)

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


class ProfileMaker:
    """
    Calculates measured shear profile based on the specified subpatches
    """
    def __init__(self, shear_data, *args):
        self.edges = shear_data.edges
        self.sd = shear_data
        self.subs = args
        self.aid = self.all_id() # union of all sub.ids
        self.reid = self.re_ids()

        self.km = None
        self.centers = None

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

    def add_centers(self, centers):
        """Group the objects to the specified centers"""
        self.km = krd.KMeans(centers)
        self.centers = centers

    def make_centers(self, ncen=100, verbose=False):
        """Calculates the specified number of k-means patches on the sky"""
        X = np.vstack((self.sd.ra[self.aid],
                       self.sd.dec[self.aid])).T
        nsample = X.shape[0] // 2
        km = krd.kmeans_sample(X, ncen=ncen, nsample=nsample, verbose=verbose)

        if not km.converged:
            km.run(X, maxiter=100)

        self.km = km
        self.centers = km.centers

    def make_profile(self, sub):
        """
        Calculates a single \Delta\Sigma profiles and covariances.

        Checks for missing values!
        """
        # create profile object
        prof = SingleProfile(self.sd.nbin, len(self.centers))

        # checking bins with zero counts
        prof.nzind = np.nonzero(np.sum(self.sd.data[0, sub.ids, :],
                                            axis=0))[0]

        # calculating radial centers of combined profile
        prof.rr[prof.nzind] = np.sum(self.sd.data[1, sub.ids][:, prof.nzind],
                                     axis=0) /\
                              np.sum(self.sd.data[2, sub.ids][:, prof.nzind],
                                     axis=0)

        # calculating combined profiles
        dsum_jack = np.sum(self.sd.data[3, sub.ids][:, prof.nzind],
                                axis=0)
        dsensum_jack = np.sum(self.sd.data[5, sub.ids][:, prof.nzind],
                                   axis=0)
        prof.dst0[prof.nzind] = dsum_jack / dsensum_jack

        osum_jack = np.sum(self.sd.data[4, sub.ids][:, prof.nzind],
                                axis=0)
        osensum_jack = np.sum(self.sd.data[6, sub.ids][:, prof.nzind],
                                   axis=0)
        prof.dsx0[prof.nzind] = osum_jack / osensum_jack

        # preparing jackknife labels
        assert self.km is not None
        prof.pos = np.vstack((self.sd.ra, self.sd.dec)).T[sub.ids, :]
        prof.labels = self.km.find_nearest(prof.pos).astype(int)
        prof.sub_labels = np.unique(prof.labels)

        # indexes of clusters for subsample i
        prof.indexes = [np.where(prof.labels != ind)[0]
                        for ind in prof.sub_labels]

        # indexes of clusters not in subsample i
        prof.non_indexes = [np.where(prof.labels == ind)[0]
                            for ind in prof.sub_labels]

        # does the p-th subpatch has counts for radial bin r?
        prof.subcounts = np.array([np.sum(self.sd.data[0, ind, :], axis=0)
                                   for ind in prof.indexes])
        prof.hasval = [np.nonzero(arr.astype(bool))[0]
                       for arr in prof.subcounts]

        prof.njk = np.sum(prof.subcounts.astype(bool), axis=0)

        # calculating jackknife subprofiles
        for i, lab in enumerate(prof.sub_labels):
            ind = prof.indexes[i]

            cind = prof.hasval[i]

            dsum_jack = np.sum(self.sd.data[3, ind][:, cind], axis=0)
            dsensum_jack = np.sum(self.sd.data[5, ind][:, cind], axis=0)
            prof.dst_sub[cind, lab] = dsum_jack / dsensum_jack

            osum_jack = np.sum(self.sd.data[4, ind][:, cind], axis=0)
            osensum_jack = np.sum(self.sd.data[6, ind][:, cind], axis=0)
            prof.dsx_sub[cind, lab] = osum_jack / osensum_jack

        # calculating the JK estimate on the mean profile
        for r in range(self.sd.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            if prof.njk[r] > 0:
                subind = prof.sub_labels[np.nonzero(prof.subcounts[:, r])[0]]
                prof.dst[r] = np.sum(prof.dst_sub[r, subind]) / prof.njk[r]
                prof.dsx[r] = np.sum(prof.dsx_sub[r, subind]) / prof.njk[r]

        return prof



class SingleProfile:
    def __init__(self, nbin, ncen):
        """Container for a single profile"""
        self.nbin = nbin # number of radial bins for the profile
        self.ncen = ncen # number of kmeans center for the Jackknife

        self.labels = None
        self.sub_labels = None
        self.indexes = None
        self.non_indexes = None
        self.nzind = None
        self.subcounts = None
        self.hasval = None

        # by default the radial bins are set to -1.0, to indicate missing data
        self.rr = np.ones(self.nbin) * -1.0
        self.dst0 = np.zeros(self.nbin)
        self.dst = np.zeros(self.nbin)

        self.dsx0 = np.zeros(self.nbin)
        self.dsx = np.zeros(self.nbin)

        # containers for the subprofiles in the Jackknife resamplings
        self.dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dsx_sub = np.zeros(shape=(self.nbin, self.ncen))

        # errors anc covariances
        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)

        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)


class ProfileMaker2:
    """
    Calculates measured shear profile based on the specified subpatches
    """
    def __init__(self, shear_data, *args):
        self.edges = shear_data.edges
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





