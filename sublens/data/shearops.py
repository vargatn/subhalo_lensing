"""
Shear operations and containers
"""

import math
import numpy as np
import kmeans_radec as krd
import pickle
import os

from ..io import xread, fread


def _get_nbin(data):
    """obtains number of radial bins"""
    return len(data[0, 0, :])


def redges(rmin, rmax, nbin):
    """
    Calculates nominal edges and centers for logarithmic bins
    (base10 logarithm is used)

    Edges and areas are exact, "center" values are estimated assuming a
    uniform source surface density. That is it gives the CIRCUMFERENCE weighted
    radius...

    :param rmin: inner edge

    :param rmax: outer edge

    :param nbin: number of bins

    :returns: centers, edges, areas

    """
    edges = np.logspace(math.log10(rmin), math.log10(rmax), nbin + 1,
                        endpoint=True)
    cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                     (edges[i + 1] ** 2. - edges[i] ** 2.)
                     for i, edge in enumerate(edges[:-1])])

    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, val in enumerate(edges[:-1])])

    return cens, edges, areas


class RawProfileContainer(object):
    def __init__(self, data, info, ra, dec, cat, rmin, rmax, lcname,
                 ind=None, **kwargs):
        """
        Shear data file handling for raw profiles

        :param data: data table from xshear output

        :param info: info table from xshear output

        :param ra: RA coordinate of lens cat (np.array)

        :param dec: DEC coordinate of lens cat (np.array)

        :param cat: Catalog containing lens parameters (recordarray like)

        :param nbin: number of logarithmic bins

        :param rmin: innermost bin edge

        :param rmax: outermost bin edge

        :param lcname: lens catalog string description
        """

        self.data = data
        self.info = info
        self.ra = ra
        self.dec = dec
        self.cat = np.array(cat)
        self.nbin = _get_nbin(data)
        self.rmin = rmin
        self.rmax = rmax
        self.lcname = lcname
        self.ind = ind

        # saving additional parameters (e.g. cosmology)
        self.settings = kwargs
        # calculating nominal bin centers, edges, areas
        self.cens, self.edges, self.areas = redges(rmin, rmax, self.nbin)

    @classmethod
    def from_xfile(cls, fname, cat):
        """
        Loads raw profile data based on xshear log pickle

        :param fname: log pickle file

        :param cat: Catalog
        """

        head, tail = os.path.split(fname)
        head += '/'
        log = pickle.load(open(fname, "rb"))
        info, data, valnames = xread(head + log['res_name'])
        # lens_file = np.loadtxt(head + log['lens_name'])
        lens_file = fread(head + log['lens_name'])
        ra = lens_file[:, 1]
        dec = lens_file[:, 2]
        sio1 = cls(data, info, ra, dec, cat, **log)
        return sio1

    def shrink(self, ind=None):
        """
        Shrinks dataset to index selection, (default is copy!)

        :param ind: array of indices to use

        :returns: instance of RawProfileContainer
        """
        if ind is None:
            ind = np.arange(len(self.ra))

        return RawProfileContainer(self.data[:, ind, :], self.info[ind, :],
                                   self.ra[ind], self.dec[ind],
                                   self.cat[:][ind], rmin=self.rmin,
                                   rmax=self.rmax, lcname=self.lcname, ind=ind,
                                   **self.settings)


class StackedProfileContainer(object):
    def __init__(self, info, data, pos, rmin, rmax, lcname, **kwargs):
        """
        Container for Stacked profile

        NOTE: the calculation right now is quite slow, the bottleneck seems to
        be in the repeated indexing -- slicing of arrays in the _subprofiles
        function!!!

        :param info: first part of xshear outfile

        :param data: second part of xshear outfile

        :param pos: (RA, DEC) in degree

        :param nbin: number of logarithmic bins

        :param rmin: innermost bin edge

        :param rmax: outermost bin edge

        :param lcname: lens catalog string description
        """

        # input params saved
        self.info = info
        self.data = data
        self.pos = pos
        self.nbin = _get_nbin(data)
        self.rmin = rmin
        self.rmax = rmax
        self.lcname = lcname
        self.num = len(pos)

        # calculating nominal bin centers, edges, areas
        self.cens, self.edges, self.areas = redges(rmin, rmax, self.nbin)

        # containers for stacking parameters
        self.weights = None  # stacking weights

        # containers for the Jackknife sampling
        self.centers = None  # centers to be used for Jackknife (RA, DEC)
        self.ncen = 1  # number of centers
        self.km = None
        self.labels = None
        self.sub_labels = None
        self.subcounts = None
        self.indexes = None
        self.non_indexes = None
        self.hasval = None
        self.wdata = None

        self.dsx_sub = None
        self.dst_sub = None

        # containers for the resulting profile
        self.w = np.ones(self.num)
        self.rr = np.ones(self.nbin) * -1.0
        self.dst0 = np.zeros(self.nbin)
        self.dsx0 = np.zeros(self.nbin)
        self.dst = np.zeros(self.nbin)
        self.dsx = np.zeros(self.nbin)
        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

        self.neff = 0  # number of entries with sources in any bin
        self.hasprofile = False

    @classmethod
    def from_raw(cls, raw):
        """
        Extracts stacked profile from RawProfileContainer
        """
        return cls(info=raw.info, data=raw.data,
                   pos=np.vstack((raw.ra, raw.dec)).T, rmin=raw.rmin,
                   rmax=raw.rmax, lcname=raw.lcname, **raw.settings)

    def _reset_profile(self):
        """Resets the profile container"""
        self.w = np.ones(self.num)
        self.rr = np.ones(self.nbin) * -1.0
        self.dst0 = np.zeros(self.nbin)
        self.dsx0 = np.zeros(self.nbin)
        self.dst = np.zeros(self.nbin)
        self.dsx = np.zeros(self.nbin)
        self.dst_cov = np.zeros((self.nbin, self.nbin))
        self.dst_err = np.zeros(self.nbin)
        self.dsx_cov = np.zeros((self.nbin, self.nbin))
        self.dsx_err = np.zeros(self.nbin)

        self.hasprofile = False

    def get_patches(self, centers, verbose=False):
        """
        Obtains JK subpatches using a spherical k-means algorithm (from Erin)

        :param centers: JK center coordinates (RA, DEC) or numbers

        :param verbose: passed to kmeans radec
        """

        if not np.iterable(centers):  # if centers is a number
            self.ncen = centers
            nsample = self.pos.shape[0] // 2
            self.km = krd.kmeans_sample(self.pos, ncen=self.ncen,
                                        nsample=nsample, verbose=verbose)
            if not self.km.converged:
                self.km.run(self.pos, maxiter=100)
            self.centers = self.km.centers
        else:  # if centers is an array of RA, DEC pairs
            assert len(centers.shape) == 2  # shape should be (:, 2)
            self.km = krd.KMeans(centers)
            self.centers = centers
            self.ncen = len(centers)

        self.labels = self.km.find_nearest(self.pos).astype(int)
        self.sub_labels = np.unique(self.labels)

        # indexes of clusters for subsample i
        self.indexes = [np.where(self.labels != ind)[0]
                        for ind in self.sub_labels]

        # indexes of clusters not in subsample i
        self.non_indexes = [np.where(self.labels == ind)[0]
                            for ind in self.sub_labels]

        self.dsx_sub = np.zeros(shape=(self.nbin, self.ncen))
        self.dst_sub = np.zeros(shape=(self.nbin, self.ncen))

    def _get_rr(self):
        """calculating radial values for data points"""
        nzind = np.where(np.sum(self.data[0, :, :], axis=0) > 0)[0]
        self.rr[nzind] = np.sum(self.data[1, :, nzind] * self.w, axis=1) /\
                         np.sum(self.data[2, :, nzind] * self.w, axis=1)

    def _get_neff(self):
        """calculates effective number of entries (lenses)"""
        return len(np.nonzero(self.info[:, 2])[0])

    def _nullprofiles(self):
        """Calculates reference profile from all entries"""
        # checking for radial bins with zero source count (in total)
        nzind = np.where(np.sum(self.data[0, :, :], axis=0) > 0)[0]

        # calculating radial values for data points
        self.rr[nzind] = np.sum(self.data[1, :, nzind], axis=1) /\
                  np.sum(self.data[2, :, nzind], axis=1)

        dsum_jack = np.average(self.data[3, :, nzind], axis=1, weights=self.w)
        dsum_w_jack = np.average(self.data[5, :, nzind], axis=1,
                                 weights=self.w)
        self.dst0[nzind] = dsum_jack / dsum_w_jack

        osum_jack = np.average(self.data[4, :, nzind], axis=1, weights=self.w)
        osum_w_jack = np.average(self.data[6, :, nzind], axis=1,
                                 weights=self.w)
        self.dsx0[nzind] = osum_jack / osum_w_jack

    # THIS takes a lot of resources
    def _subprofiles(self):
        """Calculates subprofiles for each patch"""

        # does the p-th subpatch has counts for radial bin r?
        self.subcounts = np.array([np.sum(self.data[0, ind, :], axis=0)
                                   for ind in self.indexes])
        hasval = [np.nonzero(arr.astype(bool))[0] for arr in self.subcounts]

        # calculating jackknife subprofiles
        for i, lab in enumerate(self.sub_labels):
            ind = self.indexes[i]
            cind = hasval[i]

            wsum = np.sum(self.w[ind])

            dsum_jack = np.sum(self.data[3, ind][:, cind] *
                               self.w[ind, np.newaxis], axis=0) / wsum
            dsum_w_jack = np.sum(self.data[5, ind][:, cind] *
                                  self.w[ind, np.newaxis], axis=0) / wsum
            self.dst_sub[cind, lab] = dsum_jack / dsum_w_jack

            osum_jack = np.sum(self.data[4, ind][:, cind] *
                               self.w[ind, np.newaxis], axis=0) / wsum
            osum_w_jack = np.sum(self.data[6, ind][:, cind] *
                                  self.w[ind, np.newaxis], axis=0) / wsum
            self.dsx_sub[cind, lab] = osum_jack / osum_w_jack

    def _profcalc(self):
        """JK estimate on the mean profile"""
        for r in range(self.nbin):
            # checking for radial bins with 0 pair count (to avoid NaNs)
            subind = self.sub_labels[np.where(self.subcounts[:, r] > 0)[0]]
            if np.max(self.subcounts[:, r]) == 1:
                self.rr[r] = -1
            njk = len(subind)
            if njk > 1:
                self.dst[r] = np.sum(self.dst_sub[r, subind]) / njk
                self.dsx[r] = np.sum(self.dsx_sub[r, subind]) / njk
            else:
                self.rr[r] = -1.

    def _covcalc(self):
        """JK estimate on the covariance matrix"""
        # calculating the covariance
        for r1 in range(self.nbin):
            for r2 in range(self.nbin):
                # getting patches where there are elements in both indices
                subind1 = self.sub_labels[np.where(
                    self.subcounts[:, r1] > 0)[0]]
                subind2 = self.sub_labels[np.where(
                    self.subcounts[:, r2] > 0)[0]]
                # overlapping indices
                subind = list(set(subind1).intersection(set(subind2)))
                njk = len(subind)
                if njk > 1:
                    self.dst_cov[r1, r2] = np.sum((self.dst_sub[r1, subind] -
                                                   self.dst[r1]) *
                                                  (self.dst_sub[r2, subind] -
                                                   self.dst[r2])) * \
                                           (njk - 1.0) / njk

                    self.dsx_cov[r1, r2] = np.sum((self.dsx_sub[r1, subind] -
                                                   self.dsx[r1]) *
                                                  (self.dsx_sub[r2, subind] -
                                                   self.dsx[r2])) * \
                                           (njk - 1.0) / njk
                elif r1 == r2:
                    self.rr[r1] = -1
        self.dst_err = np.sqrt(np.diag(self.dst_cov))
        self.dsx_err = np.sqrt(np.diag(self.dsx_cov))

    def prof_maker(self, centers, weights=None):
        """
        Calculates the Jackknife estimate of the stacked profile

        :param centers: JK centers (number or list)

        :param weights: weight for each entry in the datafile
        """

        # adding weights
        if weights is None:
            weights = np.ones(self.num)
        self.w = weights

        # preparing the JK patches
        self.get_patches(centers=centers)

        # getting radius values
        self._get_rr()

        # calculating the profiles
        self._subprofiles()
        self._profcalc()
        self._covcalc()

        self.neff = self._get_neff()
        self.hasprofile = True

    def _composite_subprofiles(self, other, operation="-"):
        """
        Applies the binary operation to the profile objects


        :param other: other instance of the StackedProfileContainer class

        :param operation: "+" or "-"
        """

        tmp_dst_sub = np.zeros(shape=(self.nbin, self.ncen))
        tmp_dsx_sub = np.zeros(shape=(self.nbin, self.ncen))

        tmp_sub_labels = np.array(
            list(set(self.sub_labels).intersection(set(other.sub_labels))))
        tmp_subcounts = np.zeros((len(tmp_sub_labels), self.nbin))
        # print(self.subcounts)
        for i in range(len(tmp_sub_labels)):
            ind = tmp_sub_labels[i]
            ind1 = np.where(self.sub_labels == ind)[0][0]
            ind2 = np.where(other.sub_labels == ind)[0][0]
            for j in range(self.nbin):
                tmp_subcounts[i, j] = np.min(
                    (self.subcounts[ind1, j], other.subcounts[ind2, j]))

        for r in range(self.nbin):
            subind = tmp_sub_labels[np.where(tmp_subcounts[:, r] > 0)[0]]
            if np.max(tmp_subcounts[:, r]) == 1:
                self.rr[r] = -1

            njk = len(subind)
            if njk > 1:
                if operation == "-":
                    tmp_dst_sub[r, subind] = self.dst_sub[r, subind] -\
                                             other.dst_sub[r, subind]
                    tmp_dsx_sub[r, subind] = self.dsx_sub[r, subind] -\
                                             other.dsx_sub[r, subind]
                elif operation == "+":
                    tmp_dst_sub[r, subind] = self.dst_sub[r, subind] +\
                                             other.dst_sub[r, subind]
                    tmp_dsx_sub[r, subind] = self.dsx_sub[r, subind] +\
                                             other.dsx_sub[r, subind]
                else:
                    raise ValueError("Operation not supported, use '+' or '-'")

        # assigning updated containers
        self.sub_labels = tmp_sub_labels
        self.subcounts = tmp_subcounts
        self.dst_sub = tmp_dst_sub
        self.dsx_sub = tmp_dsx_sub

    def composite(self, other, operation="-"):
        """
        Calculate the JK estimate on the operation applied to the two profiles

        Possible Operations:
        --------------------
        "-": self - other
        "+": self + other

        The results is updated to self. Use deepcopy to obtain
        copies of the object for storing the previous state.

        :param other: StackedProfileContainer instance

        :param operation: string specifying what to do...
        """

        # making sure that there is a profile in both containers
        assert self.hasprofile and other.hasprofile

        # making sure that the two profiles use the same centers
        err_msg = 'JK centers do not agree within 1e-5'
        np.testing.assert_allclose(self.centers, other.centers,
                                   rtol=1e-5, err_msg=err_msg)
        assert self.dst_sub.shape == other.dst_sub.shape

        # clears the profile container
        self._reset_profile()

        # getting radius values
        self._get_rr()

        # updates subprofiles
        self._composite_subprofiles(other=other, operation=operation)
        self._profcalc()
        self._covcalc()

        self.hasprofile = True


def stacked_pcov(plist):
    """
    Calculates the Covariance between a list of profiles

    :param plist: list of StackedProfileContainer objects

    :return: supercov_t, supercov_x matrices
    """
    # checking that input is of correct format
    assert np.iterable(plist)
    assert isinstance(plist[0], StackedProfileContainer)

    # data vectors for covariance
    dtvec = np.array([pc.dst for pc in plist]).flatten()
    dxvec = np.array([pc.dsx for pc in plist]).flatten()

    # lengths of bins
    dlen = len(dtvec)
    rlen = plist[0].nbin

    # container for the results
    supercov_t = np.zeros(shape=(dlen, dlen))
    supercov_x = np.zeros(shape=(dlen, dlen))

    # building up the covariance matrix
    for i1 in range(dlen):
        p1 = i1 // rlen  # the p-th profile
        pc1 = plist[p1]
        r1 = i1 % rlen  # the r-th radial bin within the p-th profile
        for i2 in range(dlen):
            p2 = i2 // rlen
            pc2 = plist[p2]
            r2 = i2 % rlen

            # calculating subpatches with data
            subind1 = pc1.sub_labels[np.nonzero(pc1.subcounts[:, r1])[0]]
            subind2 = pc2.sub_labels[np.nonzero(pc2.subcounts[:, r2])[0]]
            subind = list(set(subind1).intersection(set(subind2)))
            njk = len(subind)  # number of subpatches used
            if njk > 1:

                part1_t = (pc1.dst_sub[r1, subind] - pc1.dst[r1])
                part2_t = (pc2.dst_sub[r2, subind] - pc2.dst[r2])

                part1_x = (pc1.dsx_sub[r1, subind] - pc1.dsx[r1])
                part2_x = (pc2.dsx_sub[r2, subind] - pc2.dsx[r2])

                supercov_t[i1, i2] = np.sum(part1_t * part2_t) *\
                                     (njk - 1.) / njk
                supercov_x[i1, i2] = np.sum(part1_x * part2_x) *\
                                     (njk - 1.) / njk

    return supercov_t, supercov_x
