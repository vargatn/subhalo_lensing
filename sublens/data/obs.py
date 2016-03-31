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


class PosProfile(object):
    def __init__(self, data, raw_pos, centers):
        self.rwdata = data
        self.rwpos = raw_pos
        self.cent = centers
        self.ncen = len(centers)

        # constraining to SPT-E field
        spind = self._get_spte()
        self.data = self.rwdata[spind]
        self.pos = self.rwpos[spind]
        self.num = len(self.pos)
        print('number of objects: ', len(self.pos))

        # Handling xshear output
        self.nbin, self.xinfo, self.xdata = self.xhandler(self.data)

        # using km centers to obtain jackknife subpatches
        self.km = krd.KMeans(self.cent)
        self.labels = self.km.find_nearest(self.pos[:, 1:3]).astype(int)
        self.sub_labels = np.unique(self.labels)

        self.subcounts = None

        # indexes of clusters for subsample i
        self.indexes = [np.where(self.labels != ind)[0]
                        for ind in self.sub_labels]

        # indexes of clusters not in subsample i
        self.non_indexes = [np.where(self.labels == ind)[0]
                            for ind in self.sub_labels]

        # radial bin centers are initialized to -1, to signify missing data
        self.rr = np.ones(self.nbin) * -1.0
        self.dst0 = np.zeros(self.nbin)
        self.dst = np.zeros(self.nbin)

        self.dsx0 = np.zeros(self.nbin)
        self.dsx = np.zeros(self.nbin)

        self.dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)

        self.dsx_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

    def _get_spte(self):
        spte = np.where(self.rwpos[:, 1] > 60.0)[0]
        return spte

    def prof_maker(self):
        self._prof_prep()
        self._prof_calc()

    def _prof_prep(self):

        # checking bins with zero counts
        self.nzind = np.where(np.sum(self.xdata[0, :, :], axis=0) > 0)[0]

        # calculating radial values for data points
        self.rr[self.nzind] = np.sum(self.xdata[1, :, self.nzind], axis=1) /\
                  np.sum(self.xdata[2, :, self.nzind], axis=1)

        # calculating combined profiles
        dsum_jack = np.sum(self.xdata[3, :, self.nzind], axis=1)
        dsensum_jack = np.sum(self.xdata[5, :, self.nzind], axis=1)
        self.dst0[self.nzind] = dsum_jack / dsensum_jack

        osum_jack = np.sum(self.xdata[4, :, self.nzind], axis=1)
        osensum_jack = np.sum(self.xdata[6, :, self.nzind], axis=1)
        self.dsx0[self.nzind] = osum_jack / osensum_jack

        # does the p-th subpatch has counts for radial bin r?
        self.subcounts = np.array([np.sum(self.xdata[0, ind, :], axis=0)
                                   for ind in self.indexes])
        hasval = [np.nonzero(arr.astype(bool))[0] for arr in self.subcounts]

        # calculating jackknife subprofiles
        for i, lab in enumerate(self.sub_labels):
            ind = self.indexes[i]

            cind = hasval[i]

            dsum_jack = np.sum(self.xdata[3, ind][:, cind], axis=0)
            dsensum_jack = np.sum(self.xdata[5, ind][:, cind], axis=0)
            self.dst_sub[cind, lab] = dsum_jack / dsensum_jack

            osum_jack = np.sum(self.xdata[4, ind][:, cind], axis=0)
            osensum_jack = np.sum(self.xdata[6, ind][:, cind], axis=0)
            self.dsx_sub[cind, lab] = osum_jack / osensum_jack

    def _prof_calc(self):
        # calculating the JK estimate on the mean profile
        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            #subind = self.sub_labels[np.nonzero(self.subcounts[:, r])[0]]
            subind = self.sub_labels[np.where(self.subcounts[:, r] > 0)[0]]

            if np.max(self.subcounts[:, r]) == 1:
                self.rr[r] = -1

            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum(self.dst_sub[r, subind]) / njk
                self.dsx[r] = np.sum(self.dsx_sub[r, subind]) / njk
            else:
                self.rr[r] = -1.


        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                # subind1 = self.sub_labels[np.nonzero(self.subcounts[:, r1])[0]]
                subind1 = self.sub_labels[np.where(self.subcounts[:, r1] > 0)[0]]
                # njk1 = len(subind1)
                subind2 = self.sub_labels[np.where(self.subcounts[:, r2] > 0)[0]]
                # njk2 = len(subind2)
                subind = list(set(subind1).intersection(set(subind2)))
                njk = len(subind)

                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum((self.dst_sub[r1, subind] -
                                                   self.dst[r1]) *
                                                  (self.dst_sub[r2, subind] -
                                                   self.dst[r2])) *\
                                           (njk - 1.0) / njk

                    self.dsx_cov[r1, r2] = np.sum((self.dsx_sub[r1, subind] -
                                                   self.dsx[r1]) *
                                                  (self.dsx_sub[r2, subind] -
                                                   self.dsx[r2])) *\
                                           (njk - 1.0) / njk
                elif r1 == r2:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

    def test_err(self):
             # testing the error
        err = np.zeros(self.nbin)

        # print(self.subcounts)

        for r in range(self.nbin):
            print("r==", r)
            subind = self.sub_labels[np.where(self.subcounts[:, r].astype(int) > 1)[0]]

            # if r == :
            # print(self.dst_sub[r, subind])

            njk = len(subind)
            print(njk)
            if njk > 1:
                # print(np.sum((self.dst_sub[r, subind] - self.dst[r])**2.))
                # print((njk - 1.0) / njk)
                # print(np.sum((self.dst_sub[r, subind] - self.dst[r])**2.) * (njk - 1.0) / njk)
                err[r] = np.sum((self.dst_sub[r, subind] - self.dst[r])**2.) * (njk - 1.0) / njk


        return np.sqrt(err)


    def subtract(self, other):

        self.dst = np.zeros(self.nbin)

        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            # subind_a = self.sub_labels[np.nonzero(self.subcounts[:, r])[0]]
            subind_a = self.sub_labels[np.where(self.subcounts[:, r] > 0.)[0]]
            subind_b = other.sub_labels[np.where(other.subcounts[:, r] > 0.)[0]]
            subind = list(set(subind_a).intersection(set(subind_b)))
            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum((self.dst_sub[r, subind] - other.dst_sub[r, subind])) / njk
                self.dsx[r] = np.sum((self.dsx_sub[r, subind] - other.dsx_sub[r, subind])) / njk

        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                subind1a = self.sub_labels[np.where(self.subcounts[:, r1] > 0)[0]]
                subind1b = other.sub_labels[np.where(other.subcounts[:, r1] > 0)[0]]
                subind1 = list(set(subind1a).intersection(set(subind1b)))

                subind2a = self.sub_labels[np.where(self.subcounts[:, r2] > 0)[0]]
                subind2b = other.sub_labels[np.where(other.subcounts[:, r2] > 0)[0]]
                subind2 = list(set(subind2a).intersection(set(subind2b)))

                subind =  list(set(subind1).intersection(set(subind2)))
                njk = len(subind)
                # print(njk)
                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum(((self.dst_sub[r1, subind] - other.dst_sub[r1, subind]) - self.dst[r1]) * ((self.dst_sub[r2, subind] - other.dst_sub[r2, subind]) - self.dst[r2])) * (njk - 1.0) / njk
                    self.dsx_cov[r1, r2] = np.sum(((self.dsx_sub[r1, subind] - other.dsx_sub[r1, subind]) - self.dsx[r1]) * ((self.dsx_sub[r2, subind] - other.dsx_sub[r2, subind]) - self.dsx[r2])) * (njk - 1.0) / njk

                if r1 == r2 and njk <= 1:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

    @staticmethod
    def xhandler(data):
        """
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i", "dsensum_i", "osensum_i"),
        """
        bins = (data.shape[1] - 3) // 7

        # position indexes
        sid = 3
        pos_npair = 0
        pos_rsum = 1
        pos_wsum = 2
        pos_dsum = 3
        pos_osum = 4
        pos_dsensum = 5
        pos_osensum = 6

        gid = data[:, 0]
        weight_tot = data[:, 1]
        tot_pairs = data[:, 2]
        npair = data[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
        rsum = data[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
        wsum = data[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
        dsum = data[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
        osum = data[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
        dsensum = data[:,
                  sid + pos_dsensum * bins: sid + (pos_dsensum + 1) * bins]
        osensum = data[:,
                  sid + pos_osensum * bins: sid + (pos_osensum + 1) * bins]

        info = np.vstack((gid, weight_tot, tot_pairs)).T
        data = np.dstack((npair, rsum, wsum, dsum, osum, dsensum, osensum))
        data = np.transpose(data, axes=(2, 0, 1))

        # checking if loading made sense
        assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

        return bins, info, data

    def subtract2(self, other):
        self.dst = np.zeros(self.nbin)

        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

        tmp_dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        tmp_dsx_sub = np.zeros(shape=(self.nbin, self.ncen))

        tmp_sub_labels = np.array(list(set(self.sub_labels).intersection(set(other.sub_labels))))
        tmp_subcounts = np.zeros((len(tmp_sub_labels), self.nbin))
        # print(self.subcounts)
        for i in range(len(tmp_sub_labels)):
            ind = tmp_sub_labels[i]
            ind1 = np.where(self.sub_labels == ind)[0][0]
            ind2 = np.where(other.sub_labels == ind)[0][0]
            for j in range(self.nbin):
                tmp_subcounts[i, j] = np.min((self.subcounts[ind1, j], other.subcounts[ind2, j]))

        for r in range(self.nbin):
            subind = tmp_sub_labels[np.where(tmp_subcounts[:, r] > 0)[0]]
            if np.max(tmp_subcounts[:, r]) == 1:
                self.rr[r] = -1

            njk = len(subind)
            if njk > 1:
                tmp_dst_sub[r, subind] = self.dst_sub[r, subind] - other.dst_sub[r, subind]
                tmp_dsx_sub[r, subind] = self.dsx_sub[r, subind] - other.dsx_sub[r, subind]

        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            subind_a = self.sub_labels[np.where(self.subcounts[:, r] > 0)[0]]
            subind_b = other.sub_labels[np.where(other.subcounts[:, r] > 0)[0]]
            subind = list(set(subind_a).intersection(set(subind_b)))
            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum((self.dst_sub[r, subind] - other.dst_sub[r, subind])) / njk
                self.dsx[r] = np.sum((self.dsx_sub[r, subind] - other.dsx_sub[r, subind])) / njk
            else:
                self.rr[r] = -1.

        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                subind1a = self.sub_labels[np.where(self.subcounts[:, r1] > 0)[0]]
                subind1b = other.sub_labels[np.where(other.subcounts[:, r1] > 0)[0]]
                subind1 = list(set(subind1a).intersection(set(subind1b)))

                subind2a = self.sub_labels[np.where(self.subcounts[:, r2] > 0)[0]]
                subind2b = other.sub_labels[np.where(other.subcounts[:, r2] > 0)[0]]
                subind2 = list(set(subind2a).intersection(set(subind2b)))

                subind =  list(set(subind1).intersection(set(subind2)))
                njk = len(subind)
                # print(njk)
                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum(((self.dst_sub[r1, subind] - other.dst_sub[r1, subind]) - self.dst[r1]) * ((self.dst_sub[r2, subind] - other.dst_sub[r2, subind]) - self.dst[r2])) * (njk - 1.0) / njk
                    self.dsx_cov[r1, r2] = np.sum(((self.dsx_sub[r1, subind] - other.dsx_sub[r1, subind]) - self.dsx[r1]) * ((self.dsx_sub[r2, subind] - other.dsx_sub[r2, subind]) - self.dsx[r2])) * (njk - 1.0) / njk
                elif r1 == r2:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

        self.sub_labels = tmp_sub_labels
        self.subcounts = tmp_subcounts
        self.dst_sub = tmp_dst_sub
        self.dsx_sub = tmp_dsx_sub








def ppcov(pc_list):

    dtvec = np.array([pc.dst for pc in pc_list]).flatten()
    dxvec = np.array([pc.dsx for pc in pc_list]).flatten()

    dlen = len(dtvec)
    rlen = pc_list[0].nbin
    supercov_t= np.zeros(shape=(dlen, dlen))
    supercov_x= np.zeros(shape=(dlen, dlen))

    for i1 in range(dlen):
        p1 = i1 // rlen
        pc1 = pc_list[p1]
        r1 = i1 % rlen
        for i2 in range(dlen):
            p2 = i2 // rlen
            pc2 = pc_list[p2]
            r2 = i2 % rlen

            subind1 = pc1.sub_labels[np.nonzero(pc1.subcounts[:, r1])[0]]
            subind2 = pc2.sub_labels[np.nonzero(pc2.subcounts[:, r2])[0]]
            subind = list(set(subind1).intersection(set(subind2)))
            njk = len(subind)
            if njk > 1:

                part1_t = (pc1.dst_sub[r1, subind] - pc1.dst[r1])
                part2_t = (pc2.dst_sub[r2, subind] - pc2.dst[r2])

                part1_x = (pc1.dsx_sub[r1, subind] - pc1.dsx[r1])
                part2_x = (pc2.dsx_sub[r2, subind] - pc2.dsx[r2])

                supercov_t[i1, i2] = np.sum(part1_t * part2_t) * (njk - 1.) / njk
                supercov_x[i1, i2] = np.sum(part1_x * part2_x) * (njk - 1.) / njk

    return supercov_t, supercov_x



class PosProfileWeights(object):
    def __init__(self, data, raw_pos, weights, centers, cut_spte=True):
        self.rwdata = data
        self.rwpos = raw_pos
        self.rww = weights
        self.cent = centers
        self.ncen = len(centers)

        if cut_spte:
        # constraining to SPT-E field
            spind = self._get_spte()
            self.data = self.rwdata[spind]
            self.pos = self.rwpos[spind]
            self.w = self.rww[spind]
            self.num = len(self.pos)
        else:
            self.data = self.rwdata
            self.pos = self.rwpos
            self.w = self.rww
            self.num = len(self.pos)

        print('number of objects: ', len(self.pos))

        # Handling xshear output
        self.nbin, self.xinfo, self.xdata = self.xhandler(self.data)

        # using km centers to obtain jackknife subpatches
        self.km = krd.KMeans(self.cent)
        self.labels = self.km.find_nearest(self.pos[:, 1:3]).astype(int)
        self.sub_labels = np.unique(self.labels)

        self.subcounts = None

        # indexes of clusters for subsample i
        self.indexes = [np.where(self.labels != ind)[0]
                        for ind in self.sub_labels]

        # indexes of clusters not in subsample i
        self.non_indexes = [np.where(self.labels == ind)[0]
                            for ind in self.sub_labels]

        # radial bin centers are initialized to -1, to signify missing data
        self.rr = np.ones(self.nbin) * -1.0
        self.dst0 = np.zeros(self.nbin)
        self.dst = np.zeros(self.nbin)

        self.dsx0 = np.zeros(self.nbin)
        self.dsx = np.zeros(self.nbin)

        self.dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)

        self.dsx_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

    def _get_spte(self):
        spte = np.where(self.rwpos[:, 1] > 60.0)[0]
        return spte

    def prof_maker(self):
        self._prof_prep()
        self._prof_calc()

    def _prof_prep(self):

        # checking bins with zero counts
        self.nzind = np.where(np.sum(self.xdata[0, :, :], axis=0) > 0)[0]

        # calculating radial values for data points
        self.rr[self.nzind] = np.sum(self.xdata[1, :, self.nzind], axis=1) /\
                  np.sum(self.xdata[2, :, self.nzind], axis=1)

        # calculating combined profiles
        dsum_jack = np.sum(self.xdata[3, :, self.nzind] * self.w, axis=1) / np.sum(self.w)
        dsensum_jack = np.sum(self.xdata[5, :, self.nzind] * self.w, axis=1) / np.sum(self.w)
        self.dst0[self.nzind] = dsum_jack / dsensum_jack

        osum_jack = np.sum(self.xdata[4, :, self.nzind] * self.w, axis=1) / np.sum(self.w)
        osensum_jack = np.sum(self.xdata[6, :, self.nzind] * self.w, axis=1) / np.sum(self.w)
        self.dsx0[self.nzind] = osum_jack / osensum_jack

        # does the p-th subpatch has counts for radial bin r?
        self.subcounts = np.array([np.sum(self.xdata[0, ind, :], axis=0)
                                   for ind in self.indexes])
        hasval = [np.nonzero(arr.astype(bool))[0] for arr in self.subcounts]

        # calculating jackknife subprofiles
        for i, lab in enumerate(self.sub_labels):
            ind = self.indexes[i]

            cind = hasval[i]

            dsum_jack = np.sum(self.xdata[3, ind][:, cind] * self.w[ind, np.newaxis], axis=0) / np.sum(self.w[ind])
            dsensum_jack = np.sum(self.xdata[5, ind][:, cind] * self.w[ind, np.newaxis], axis=0) / np.sum(self.w[ind])
            self.dst_sub[cind, lab] = dsum_jack / dsensum_jack

            osum_jack = np.sum(self.xdata[4, ind][:, cind] * self.w[ind, np.newaxis], axis=0) / np.sum(self.w[ind])
            osensum_jack = np.sum(self.xdata[6, ind][:, cind] * self.w[ind, np.newaxis], axis=0) / np.sum(self.w[ind])
            self.dsx_sub[cind, lab] = osum_jack / osensum_jack

    def _prof_calc(self):
        # calculating the JK estimate on the mean profile
        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            #subind = self.sub_labels[np.nonzero(self.subcounts[:, r])[0]]
            subind = self.sub_labels[np.where(self.subcounts[:, r] > 0)[0]]

            if np.max(self.subcounts[:, r]) == 1:
                self.rr[r] = -1

            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum(self.dst_sub[r, subind]) / njk
                self.dsx[r] = np.sum(self.dsx_sub[r, subind]) / njk
            else:
                self.rr[r] = -1.


        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                # subind1 = self.sub_labels[np.nonzero(self.subcounts[:, r1])[0]]
                subind1 = self.sub_labels[np.where(self.subcounts[:, r1] > 0)[0]]
                # njk1 = len(subind1)
                subind2 = self.sub_labels[np.where(self.subcounts[:, r2] > 0)[0]]
                # njk2 = len(subind2)
                subind = list(set(subind1).intersection(set(subind2)))
                njk = len(subind)

                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum((self.dst_sub[r1, subind] -
                                                   self.dst[r1]) *
                                                  (self.dst_sub[r2, subind] -
                                                   self.dst[r2])) *\
                                           (njk - 1.0) / njk

                    self.dsx_cov[r1, r2] = np.sum((self.dsx_sub[r1, subind] -
                                                   self.dsx[r1]) *
                                                  (self.dsx_sub[r2, subind] -
                                                   self.dsx[r2])) *\
                                           (njk - 1.0) / njk
                elif r1 == r2:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

    def test_err(self):
             # testing the error
        err = np.zeros(self.nbin)

        # print(self.subcounts)

        for r in range(self.nbin):
            print("r==", r)
            subind = self.sub_labels[np.where(self.subcounts[:, r].astype(int) > 1)[0]]

            # if r == :
            # print(self.dst_sub[r, subind])

            njk = len(subind)
            print(njk)
            if njk > 1:
                # print(np.sum((self.dst_sub[r, subind] - self.dst[r])**2.))
                # print((njk - 1.0) / njk)
                # print(np.sum((self.dst_sub[r, subind] - self.dst[r])**2.) * (njk - 1.0) / njk)
                err[r] = np.sum((self.dst_sub[r, subind] - self.dst[r])**2.) * (njk - 1.0) / njk


        return np.sqrt(err)

    @staticmethod
    def xhandler(data):
        """
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i", "dsensum_i", "osensum_i"),
        """
        bins = (data.shape[1] - 3) // 7

        # position indexes
        sid = 3
        pos_npair = 0
        pos_rsum = 1
        pos_wsum = 2
        pos_dsum = 3
        pos_osum = 4
        pos_dsensum = 5
        pos_osensum = 6

        gid = data[:, 0]
        weight_tot = data[:, 1]
        tot_pairs = data[:, 2]
        npair = data[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
        rsum = data[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
        wsum = data[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
        dsum = data[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
        osum = data[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
        dsensum = data[:,
                  sid + pos_dsensum * bins: sid + (pos_dsensum + 1) * bins]
        osensum = data[:,
                  sid + pos_osensum * bins: sid + (pos_osensum + 1) * bins]

        info = np.vstack((gid, weight_tot, tot_pairs)).T
        data = np.dstack((npair, rsum, wsum, dsum, osum, dsensum, osensum))
        data = np.transpose(data, axes=(2, 0, 1))

        # checking if loading made sense
        assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

        return bins, info, data

    def subtract2(self, other):
        self.dst = np.zeros(self.nbin)

        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

        tmp_dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        tmp_dsx_sub = np.zeros(shape=(self.nbin, self.ncen))

        tmp_sub_labels = np.array(list(set(self.sub_labels).intersection(set(other.sub_labels))))
        tmp_subcounts = np.zeros((len(tmp_sub_labels), self.nbin))
        # print(self.subcounts)
        for i in range(len(tmp_sub_labels)):
            ind = tmp_sub_labels[i]
            ind1 = np.where(self.sub_labels == ind)[0][0]
            ind2 = np.where(other.sub_labels == ind)[0][0]
            for j in range(self.nbin):
                tmp_subcounts[i, j] = np.min((self.subcounts[ind1, j], other.subcounts[ind2, j]))

        for r in range(self.nbin):
            subind = tmp_sub_labels[np.where(tmp_subcounts[:, r] > 0)[0]]
            if np.max(tmp_subcounts[:, r]) == 1:
                self.rr[r] = -1

            njk = len(subind)
            if njk > 1:
                tmp_dst_sub[r, subind] = self.dst_sub[r, subind] - other.dst_sub[r, subind]
                tmp_dsx_sub[r, subind] = self.dsx_sub[r, subind] - other.dsx_sub[r, subind]

        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            subind_a = self.sub_labels[np.where(self.subcounts[:, r] > 0)[0]]
            subind_b = other.sub_labels[np.where(other.subcounts[:, r] > 0)[0]]
            subind = list(set(subind_a).intersection(set(subind_b)))
            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum((self.dst_sub[r, subind] - other.dst_sub[r, subind])) / njk
                self.dsx[r] = np.sum((self.dsx_sub[r, subind] - other.dsx_sub[r, subind])) / njk
            else:
                self.rr[r] = -1.

        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                subind1a = self.sub_labels[np.where(self.subcounts[:, r1] > 0)[0]]
                subind1b = other.sub_labels[np.where(other.subcounts[:, r1] > 0)[0]]
                subind1 = list(set(subind1a).intersection(set(subind1b)))

                subind2a = self.sub_labels[np.where(self.subcounts[:, r2] > 0)[0]]
                subind2b = other.sub_labels[np.where(other.subcounts[:, r2] > 0)[0]]
                subind2 = list(set(subind2a).intersection(set(subind2b)))

                subind =  list(set(subind1).intersection(set(subind2)))
                njk = len(subind)
                # print(njk)
                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum(((self.dst_sub[r1, subind] - other.dst_sub[r1, subind]) - self.dst[r1]) * ((self.dst_sub[r2, subind] - other.dst_sub[r2, subind]) - self.dst[r2])) * (njk - 1.0) / njk
                    self.dsx_cov[r1, r2] = np.sum(((self.dsx_sub[r1, subind] - other.dsx_sub[r1, subind]) - self.dsx[r1]) * ((self.dsx_sub[r2, subind] - other.dsx_sub[r2, subind]) - self.dsx[r2])) * (njk - 1.0) / njk
                elif r1 == r2:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

        self.sub_labels = tmp_sub_labels
        self.subcounts = tmp_subcounts
        self.dst_sub = tmp_dst_sub
        self.dsx_sub = tmp_dsx_sub




