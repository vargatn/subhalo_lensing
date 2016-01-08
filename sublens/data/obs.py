"""
Module to create the observed side of the Modelling likelihood
"""

import pickle
import numpy as np
import kmeans_radec as krd


class ObsSpace(object):
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

    def get_container(self):
        """returns the easily pickleable container"""
        return SpaceContainer(self.tracers, self.data, self.ids,
                              self.par_ranges)

    @classmethod
    def from_container(cls, cont):
        return cls(cont.tracers, cont.ids, cont.data)


def save_subs(sname, subs, tag=''):
    """pickles the subpatches"""
    contlist = [sub.get_container() for sub in subs]
    logdict = {
        'tag': tag,
        'contlist': contlist,
    }
    pickle.dump(logdict, open(sname, "wb"))


def load_subs(lname):
    """unpickles subpatches"""
    logdict = pickle.load(open(lname, "rb"))
    sublist = [ObsSpace.from_container(cont) for cont in logdict['contlist']]
    return logdict['tag'], sublist


class SpaceContainer(object):
    def __init__(self, tracers, data, ids, par_ranges):
        self.tracers = tracers
        self.data = data
        self.ids = ids
        self.par_ranges = par_ranges


class ProfileMaker(object):
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

        self.proflist = None
        self.cov_t = None
        self.cov_x = None

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

    def all_profiles(self):
        """calculates all \Delta\Sigma profiles and the joint covariance"""

        # caclualting all profiles
        self.proflist = [self.make_profile(sub) for sub in self.subs]

        # creating data vectors for joint covariance
        self.rvec = np.array([prof.rr for prof in self.proflist]).flatten()
        self.dtvec = np.array([prof.dst for prof in self.proflist]).flatten()
        self.dxvec = np.array([prof.dsx for prof in self.proflist]).flatten()

        dlen = len(self.dtvec)
        rlen = self.sd.nbin

        # containers for the matrices
        self.cov_t= np.zeros(shape=(dlen, dlen))
        self.cov_x= np.zeros(shape=(dlen, dlen))

        for i1 in range(dlen):
            p1 = i1 // rlen
            pc1 = self.proflist[p1]
            r1 = i1 % rlen
            for i2 in range(dlen):
                p2 = i2 // rlen
                pc2 = self.proflist[p2]
                r2 = i2 % rlen

                if pc1.njk[r1] > 0 and pc2.njk[r2] > 0:

                    subind1 = pc1.sub_labels[np.nonzero(
                        pc1.subcounts[:, r1])[0]]
                    subind2 = pc2.sub_labels[np.nonzero(
                        pc2.subcounts[:, r2])[0]]
                    subind = list(set(subind1).intersection(set(subind2)))

                    part1_t = (pc1.dst_sub[r1, subind] - pc1.dst[r1])
                    part2_t = (pc2.dst_sub[r2, subind] - pc2.dst[r2])

                    part1_x = (pc1.dsx_sub[r1, subind] - pc1.dsx[r1])
                    part2_x = (pc2.dsx_sub[r2, subind] - pc2.dsx[r2])

                    navail = len(subind)
                    assert(navail >= 1)
                    self.cov_t[i1, i2] = np.sum(part1_t * part2_t) *\
                                         (navail - 1.) / navail
                    self.cov_x[i1, i2] = np.sum(part1_x * part2_x) *\
                                         (navail - 1.) / navail

    def save_profiles(self, sname="profiles.p", tag=''):
        """saves profiles"""
        logdict = {
            'tag': tag,
            'profiles': self.proflist,
            'cov_t': self.cov_t,
            'cov_x': self.cov_x,
        }
        pickle.dump(logdict, open(sname, "wb"))

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
        prof.subcounts = np.array([np.sum(self.sd.data[0, sub.ids[ind], :], axis=0)
                                   for ind in prof.indexes])
        prof.hasval = [np.nonzero(arr.astype(bool))[0]
                       for arr in prof.subcounts]

        prof.njk = np.sum(prof.subcounts.astype(bool), axis=0)

        # calculating jackknife subprofiles
        for i, lab in enumerate(prof.sub_labels):
            ind = prof.indexes[i]

            cind = prof.hasval[i]

            dsum_jack = np.sum(self.sd.data[3, sub.ids[ind]][:, cind], axis=0)
            dsensum_jack = np.sum(self.sd.data[5, sub.ids[ind]][:, cind], axis=0)
            prof.dst_sub[cind, lab] = dsum_jack / dsensum_jack

            osum_jack = np.sum(self.sd.data[4, sub.ids[ind]][:, cind], axis=0)
            osensum_jack = np.sum(self.sd.data[6, sub.ids[ind]][:, cind], axis=0)
            prof.dsx_sub[cind, lab] = osum_jack / osensum_jack

        # calculating the JK estimate on the mean profile
        for r in range(self.sd.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            if prof.njk[r] > 0:
                subind = prof.sub_labels[np.nonzero(prof.subcounts[:, r])[0]]
                prof.dst[r] = np.sum(prof.dst_sub[r, subind]) / prof.njk[r]
                prof.dsx[r] = np.sum(prof.dsx_sub[r, subind]) / prof.njk[r]

        # print(prof.dst_cov.shape)
        # calculating the covariance
        for r1 in range(self.sd.nbin):
            for r2 in range(self.sd.nbin):
                if prof.njk[r1] > 0 and prof.njk[r2] > 0:
                    subind1 = prof.sub_labels[np.nonzero(prof.subcounts[:, r1])[0]]
                    subind2 = prof.sub_labels[np.nonzero(prof.subcounts[:, r2])[0]]
                    subind = list(set(subind1).intersection(set(subind2)))

                    prof.dst_cov[r1, r2] = np.sum((prof.dst_sub[r1, subind] -
                                                   prof.dst[r1]) *
                                                  (prof.dst_sub[r2, subind] -
                                                   prof.dst[r2])) *\
                                           (prof.njk[r] - 1.0) / prof.njk[r]
                    prof.dsx_cov[r1, r2] = np.sum((prof.dsx_sub[r1, subind] -
                                                   prof.dsx[r1]) *
                                                  (prof.dsx_sub[r2, subind] -
                                                   prof.dsx[r2])) *\
                                           (prof.njk[r] - 1.0) / prof.njk[r]

        prof.dst_err = np.sqrt(np.diag(prof.dst_cov))
        prof.dsx_err = np.sqrt(np.diag(prof.dsx_cov))

        return prof

def load_profiles(lname):
    """unpickles profiles"""
    logdict = pickle.load(open(lname, 'rb'))
    return logdict['tag'], logdict['profiles'], logdict['cov_t'],\
           logdict['cov_x']

class SingleProfile(object):
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
        self.njk = None

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








