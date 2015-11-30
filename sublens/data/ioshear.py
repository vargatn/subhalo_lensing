"""
Class to document xshear calculations
"""

import math
import numpy as np
from astropy.io import fits
import pickle
import pandas as pd
import kmeans_radec as krd


class ShearIO:

    def __init__(self, info, data, cat, ra, dec, nbin=15, rmin=0.02, rmax=30.,
                 doc=""):
        self.info = info
        self.data = data
        self.cat = cat
        self.ra = ra
        self.dec = dec
        self.nbin = 15
        self.rmin = 0.02
        self.rmax = rmax
        self.doc = doc
        self.cens, self.edges, self.areas = self.redges(rmin, rmax, nbin)

    @classmethod
    def from_file(cls, fname):
        """loads data file from WrapX log"""
        log = pickle.load(open(fname, "rb"))

        names, info, data = cls.xout(log['res_name'], mode='dat')
        cat = fits.open(log["lens_path"])[1].data
        lens_file = np.loadtxt(log['lens_name'])
        ra = lens_file[:, 1]
        dec = lens_file[:, 2]

        sio1 = cls(info, data, cat, ra, dec, nbin=log['nbin'],
                   rmin=log['rmin'], rmax=log['rmax'], doc=log['doc'])
        return sio1

    @staticmethod
    def redges(rmin, rmax, nbin):
        """calculates true edges and center for logarithmic bins"""
        edges = np.logspace(math.log10(rmin), math.log10(rmax), nbin + 1,
                            endpoint=True)
        cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                         (edges[i + 1] ** 2. - edges[i] ** 2.)
                         for i, edge in enumerate(edges[:-1])])

        areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                          for i, val in enumerate(edges[:-1])])

        return cens, edges, areas

    @staticmethod
    def xout(fname, mode="dat", bins="auto"):
        """
        loads lensfit style output cfile from xshear

        This is the cfile where the output is piped, e.g.: lens_res.dat

        columns:
        ---------
        index, weight_tot, totpairs, npair_i, rsum_i, wsum_i, dsum_i, osum_i,
        dsensum_i, osensum_i

        description:
        -------------
        index:      index from lens catalog
        weight_tot: sum of all weights for all source pairs in all radial bins
        totpairs:   total pairs used
        npair_i:    number of pairs in radial bin i.  N columns.
        rsum_i:     sum of radius in radial bin i
        wsum_i:     sum of weights in radial bin i
        dsum_i:     sum of \Delta\Sigma_+ * weights in radial bin i.
        osum_i:     sum of \Delta\Sigma_x * weights in  radial bin i.
        dsensum_i:  sum of weights times sensitivities
        osensum_i:  sum of weights times sensitivities


        :param fname: cfile name
        :param mode: switches between ascii and npy table
        :return: numpy array
        """

        names = {
            "info": ("index", "weight_tot", "totpairs"),
            "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                     "dsensum_i", "osensum_i"),
        }

        #  reading output data
        if mode == "npy":
            raw_data = np.load(fname)
        elif mode == "dat":
            raw_data = pd.read_csv(fname, header=None, sep=" ").values
        else:
            raise NameError("Invalid format\n please use: npy or dat")

        # calculates number of radial bins used
        if bins == "auto":
            bins = (raw_data.shape[1] - 3) // 7

        # position indexes
        sid = 3
        pos_npair = 0
        pos_rsum = 1
        pos_wsum = 2
        pos_dsum = 3
        pos_osum = 4
        pos_dsensum = 5
        pos_osensum = 6

        gid = raw_data[:, 0]
        weight_tot = raw_data[:, 1]
        tot_pairs = raw_data[:, 2]
        npair = raw_data[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
        rsum = raw_data[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
        wsum = raw_data[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
        dsum = raw_data[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
        osum = raw_data[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
        dsensum = raw_data[:,
                  sid + pos_dsensum * bins: sid + (pos_dsensum + 1) * bins]
        osensum = raw_data[:,
                  sid + pos_osensum * bins: sid + (pos_osensum + 1) * bins]

        info = np.vstack((gid, weight_tot, tot_pairs)).T
        data = np.dstack((npair, rsum, wsum, dsum, osum, dsensum, osensum))
        data = np.transpose(data, axes=(2, 0, 1))

        # checking if loading made sense
        assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

        return names, info, data

    def prof_raw(self, ids=None, pw=None):
        """calculates raw profile based on dataset"""

        if ids is None:
            ids = np.arange(len(self.data[0, :, 0]))
        idata = self.data[:, ids, :]

        if pw is None:
            pw = np.expand_dims(np.ones(len(idata[0, :, 0])), axis=1)

        psum = np.sum(pw)

        rr = np.sum(np.multiply(idata[1, :, :], pw), axis=0) /\
             np.sum(np.multiply(idata[2, :, :], pw), axis=0)

        dsum = np.sum(np.multiply(idata[3, :, :], pw), axis=0) / psum

        dsensum = np.sum(np.multiply(idata[5, :, :], pw), axis=0) / psum

        osum = np.sum(np.multiply(idata[4, :, :], pw), axis=0) / psum

        osensum = np.sum(np.multiply(idata[6, :, :], pw), axis=0) / psum

        dst = dsum / dsensum
        dsx = osum / osensum

        return rr, dst, dsx

    def prof_err(self, ids=None, pw=None, ncen=100, verbose=False):
        """shorthand wrapper"""
        return self.jack_err(self.ra, self.dec, ids=ids, pw=pw, ncen=ncen,
                             verbose=verbose)

    def jack_err(self, ra, dec, ids=None, pw=None, ncen=100, verbose=False):
        """calculates raw profile based on dataset"""

        if ids is None:
            ids = np.array(range(self.data.shape[1]))
        idata = self.data[:, ids, :]

        if pw is None:
            pw = np.expand_dims(np.ones(shape=ids.shape), axis=1)
        psum = np.sum(pw)

        rr = np.sum(np.multiply(idata[1, :, :], pw), axis=0) /\
             np.sum(np.multiply(idata[2, :, :], pw), axis=0)

        # creating kmeans patches
        X = np.vstack((ra[ids], dec[ids])).T
        nsample = X.shape[0] // 2
        km = krd.kmeans_sample(X, ncen=ncen, nsample=nsample, verbose=verbose)

        if not km.converged:
            km.run(X, maxiter=100)

        labels = np.unique(km.labels)

        dst_est = np.zeros((ncen, idata.shape[2]))
        dsx_est = np.zeros((ncen, idata.shape[2]))

        # calculating subprofiles
        for i, lab in enumerate(labels):

            ind = np.where(km.labels != lab)[0]

            dsum_jack = np.sum(np.multiply(idata[3, ind, :], pw[ind]), axis=0) / psum
            dsensum_jack = np.sum(np.multiply(idata[5, ind, :],
                                              pw[ind]), axis=0) / psum
            osum_jack = np.sum(np.multiply(idata[4, ind, :],
                                           pw[ind]), axis=0) / psum
            osensum_jack = np.sum(np.multiply(idata[6, ind, :],
                                              pw[ind]), axis=0) / psum

            dst_est[i, :] = dsum_jack / dsensum_jack
            dsx_est[i, :] = osum_jack / osensum_jack

        # estimating value
        dst = np.mean(dst_est, axis=0)
        dsx = np.mean(dsx_est, axis=0)
        dst_var = (ncen - 1.) / ncen * np.sum((dst_est - dst) ** 2., axis=0)
        dsx_var = (ncen - 1.) / ncen * np.sum((dsx_est - dsx) ** 2., axis=0)

        dst_e = np.sqrt(dst_var)
        dsx_e = np.sqrt(dsx_var)

        return rr, dst, dst_e, dsx, dsx_e

    def prof_cov(self, ids=None, pw=None, ncen=100, verbose=False):
        """shorthand wrapper"""
        return self.jack_cov(self.ra, self.dec, ids=ids, pw=pw, ncen=ncen,
                             verbose=verbose)

    def jack_cov(self, ra, dec, ids=None, pw=None, ncen=100, verbose=False):
        """calculates raw profile based on dataset"""

        if ids is None:
            ids = np.array(range(self.data.shape[1]))
        idata = self.data[:, ids, :]

        if pw is None:
            pw = np.expand_dims(np.ones(shape=ids.shape), axis=1)
        psum = np.sum(pw)

        rr = np.sum(np.multiply(idata[1, :, :], pw), axis=0) /\
             np.sum(np.multiply(idata[2, :, :], pw), axis=0)

        # creating kmeans patches
        X = np.vstack((ra[ids], dec[ids])).T
        nsample = X.shape[0] // 2
        km = krd.kmeans_sample(X, ncen=ncen, nsample=nsample, verbose=verbose)

        if not km.converged:
            km.run(X, maxiter=100)

        labels = np.unique(km.labels)

        dst_est = np.zeros((ncen, idata.shape[2]))
        dsx_est = np.zeros((ncen, idata.shape[2]))

        # calculating subprofiles
        for i, lab in enumerate(labels):

            ind = np.where(km.labels != lab)[0]

            dsum_jack = np.sum(np.multiply(idata[3, ind, :], pw[ind]), axis=0) / psum
            dsensum_jack = np.sum(np.multiply(idata[5, ind, :],
                                              pw[ind]), axis=0) / psum
            osum_jack = np.sum(np.multiply(idata[4, ind, :],
                                           pw[ind]), axis=0) / psum
            osensum_jack = np.sum(np.multiply(idata[6, ind, :],
                                              pw[ind]), axis=0) / psum

            dst_est[i, :] = dsum_jack / dsensum_jack
            dsx_est[i, :] = osum_jack / osensum_jack

        # estimating value
        dst = np.mean(dst_est, axis=0)
        dsx = np.mean(dsx_est, axis=0)

        # dst_var = (ncen - 1.) / ncen * np.sum((dst_est - dst) ** 2., axis=0)
        # dsx_var = (ncen - 1.) / ncen * np.sum((dsx_est - dsx) ** 2., axis=0)


        nbin = len(rr)

        dst_cov = np.array([[np.sum((dst_est[:, i] - dst[i]) *
                                    (dst_est[:, j] - dst[j]))
                             for j in range(nbin)] for i in range(nbin)])
        dst_cov *= (ncen - 1.) / ncen

        dsx_cov = np.array([[np.sum((dsx_est[:, i] - dsx[i]) *
                                    (dsx_est[:, j] - dsx[j]))
                             for j in range(nbin)] for i in range(nbin)])
        dsx_cov *= (ncen - 1.) / ncen

        return rr, dst, dst_cov, dsx, dsx_cov


class WrapX:

    def __init__(self, name, lens_path, source_path, xshear_path, h0=100.,
                 omega_m=0.3, healpix_nside=64, nbin=15, rmin=0.02, rmax=30,
                 zlvals='default', doc=""):

        self.name = name
        self.config_name = self.name + ".cfg"
        self.lens_name = self.name + "_lens.dat"
        self.log_name = self.name + "_log.p"
        self.res_name = self.name + "_res.dat"
        self.lens_path = lens_path
        self.source_path = source_path
        self.xshear_path = xshear_path
        self.h0 = h0
        self.omega_m = omega_m
        self.healpix_nside = healpix_nside
        self.nbin = nbin
        self.rmin = rmin
        self.rmax = rmax
        self.doc = doc
        self.lpars = None

        _default_zlvals = [0., 0.01532258, 0.03064516, 0.04596774, 0.06129032,
                 0.0766129, 0.09193548, 0.10725806, 0.12258065, 0.13790323,
                 0.15322581, 0.16854839, 0.18387097, 0.19919355, 0.21451613,
                 0.22983871, 0.24516129, 0.26048387, 0.27580645, 0.29112903,
                 0.30645161, 0.32177419, 0.33709677, 0.35241935, 0.36774194,
                 0.38306452, 0.3983871, 0.41370968, 0.42903226, 0.44435484,
                 0.45967742, 0.475, 0.49032258, 0.50564516, 0.52096774,
                 0.53629032, 0.5516129, 0.56693548, 0.58225806, 0.59758065,
                 0.61290323, 0.62822581, 0.64354839, 0.65887097, 0.67419355,
                 0.68951613, 0.70483871, 0.72016129, 0.73548387, 0.75080645,
                 0.76612903, 0.78145161, 0.79677419, 0.81209677, 0.82741935,
                 0.84274194, 0.85806452, 0.8733871, 0.88870968, 0.90403226,
                 0.91935484, 0.93467742, 0.95]

        if zlvals == "default":
            self.zlvals = _default_zlvals
        else:
            self.zlvals = zlvals

    def config(self):
        """creates config file"""
        conf = "H0                = {:.2f}\n".format(self.h0) + \
               "omega_m           = {:.2f}\n".format(self.omega_m) + \
               "healpix_nside     = {:d}\n".format(self.healpix_nside) + \
               'mask_style        = "none"\n' + \
               'shear_style       = "lensfit"\n' + \
               "nbin              = {:d}\n".format(self.nbin) + \
               "rmin              = {:.2f}\n".format(self.rmin) + \
               "rmax              = {:.2f}\n".format(self.rmax) + \
               'sigmacrit_style   = "interp"\n' +\
               "zlvals = " + str(self.zlvals)

        cfile = open(self.config_name, 'w')
        cfile.write(conf)
        cfile.close()

    def write_lens(self,ids="cat_matched", ra="RA", dec="DEC", z="Z",
                   mode="fits"):
        """creates lens data file"""

        self.lpars = {
            "ids": ids,
            "ra": ra,
            "dec": dec,
            "z": z,
        }

        if mode == "fits":
            ldata = fits.open(self.lens_path)[1].data
        else:
            raise TypeError("currently only fits files are supported")

        if ids == "cat_matched":
            ids = np.arange(len(ldata[ra]))

        field_mask_dummy = np.zeros(shape=ids.shape)
        lens = np.vstack((ids, ldata[ra], ldata[dec], ldata[z],
                          field_mask_dummy)).T

        fmt = ["%d", "%.18f", "%.18f", "%.18f", "%d"]

        np.savetxt(self.lens_name, lens, fmt=fmt)

    def write_log(self):

        log_dict =  {
            "name": self.name,
            "config_name": self.config_name,
            "lens_name": self.lens_name,
            "log_name": self.log_name,
            "res_name": self.res_name,
            "lens_path": self.lens_path,
            "source_path": self.source_path,
            "xshear_path": self.xshear_path,
            "h0": self.h0,
            "omega_m": self.omega_m,
            "healpix_nside": self.healpix_nside,
            "nbin": self.nbin,
            "rmin": self.rmin,
            "rmax": self.rmax,
            "doc": self.doc,
            "lpars": self.lpars,
            "zlvals": self.zlvals,

        }

        pickle.dump(log_dict, open(self.log_name, "wb"))

    def write_script(self, check=False):
        """executable xshear script"""

        script = "#!bin/bash \n"
        if check:
            script += "head -1 " + self.source_path + " | "
        else:
            script += "cat " + self.source_path + " | "

        script += self.xshear_path + " " + self.config_name + " " +\
                  self.lens_name + " > " + self.res_name

        sfile = open(self.name + ".sh", "w")
        sfile.write(script)
        sfile.close()

    def prepall(self):
        self.config()
        self.write_lens()
        self.write_script()
        self.write_log()






