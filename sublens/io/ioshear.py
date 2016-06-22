"""
Wrapper for xshear and xshear output
"""


import numpy as np
import pandas as pd
import pickle
import os


def tonpy(fname, force=False, **kwargs):
    """
    Converts text file to npy format

    :param fname: dat file to convert

    :param force: force overwriting existing npy file

    :return: True if writing has happened, False if didn't
    """
    # getting extension
    name, ext = fname.rsplit(sep='.', maxsplit=1)

    # checking if file exists
    newname = name + ".npy"
    fexist = os.path.isfile(newname)

    # writing
    if (ext == 'dat') and (not fexist or force):
        raw_data = pd.read_csv(fname, header=None, sep=" ", engine='c').values
        np.save(name, raw_data)
        retstate = True
    else:
        retstate = False
    return retstate


def xconv(dname, force=False):
    """
    Converts xshear text datafiles to npy binary

    :param dname: path of dictionary log

    :param force: force overwrite existing result
    """

    head, tail = os.path.split(dname)
    head += '/'
    log = pickle.load(open(dname, "rb"))

    nlist = [log['res_name'], log['lens_name']]
    for name in nlist:
        tonpy(head + name, force=force)


def fread(fname, mode='auto', **kwargs):
    """
    Loads file output

    modes:
    -----------
    dat: text file
    npy: numpy binary file
    auto: tries dat and then npy

    :param fname: file name string

    :param mode: switch between modes

    :return: data table as numpy array
    """

    #  reading output data
    if mode == "npy":
        raw_data = np.load(fname)
    elif mode == "dat":
        raw_data = pd.read_csv(fname, header=None, sep=" ", engine='c').values
    elif mode == "auto":
        name, ext = fname.rsplit(sep='.', maxsplit=1)
        try:
            raw_data = np.load(name + '.npy')
        except FileNotFoundError as e:
            raw_data = pd.read_csv(name + '.dat', header=None, sep=" ",
                                   engine='c').values
    else:
        raise NotImplementedError("Invalid format\n" +
                                  " please use: npy or dat")
    return raw_data


def xread(fname, fmode="dat", xmode='reduced', **kwargs):
    """
    Loads lensfit style output from xshear

    (This is the file where the output is piped, e.g.: lens_res.dat)

    :param fname: file name

    :param fmode: switches between ascii and npy table

    :param xmode: switches different output file formats:
                  point, reduced, sample

    :return: info, data, valnames
    """

    raw_data = fread(fname, fmode)
    return xhandler(raw_data, xmode, **kwargs)


def xhandler(xdata, xmode='reduced', **kwargs):
    """
    Converts xshear output table into a readable format

    columns:
    ---------
    index, weight_tot, totpairs, npair_i, rsum_i, wsum_i, dsum_i, osum_i,
    dsensum_i, osensum_i

    (for lensfit mode, in reduced mode dsensum_i and osensum_i is replaced
     with wsum_i)

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

    :param xdata: xshear data file

    :param xmode: switches different output file formats:
                  point, reduced, sample

    :returns: info, data, valnames
    """

    if xmode == 'reduced':
        info, data, valnames = xreduced(xdata)
    elif xmode == 'sample':
        info, data, valnames = xsample(xdata)
    elif xmode == 'lensfit':
        info, data, valnames = xlensfit(xdata)
    else:
        raise ValueError('invalid type specified')

    return info, data, valnames


def xsample(xdata, **kwargs):
    """Loader for reduced-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 'wscinvsum_i', 'wscinvsum_i'),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 6
    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4
    pos_scinv = 5

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    scinv = xdata[:, sid + pos_scinv * bins: sid + (pos_scinv + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T

    data = np.dstack((npair, rsum, wsum, dsum, osum, scinv, scinv))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


def xreduced(xdata, **kwargs):
    """Loader for reduced-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 "wsum_i", "wsum_i"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 5

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, dsum, osum, wsum, wsum))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


def xlensfit(xdata, **kwargs):
    """Loader for lensfit-style xshear output"""
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "dsum_i", "osum_i",
                 "dsensum_i", "osensum_i"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 7

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_dsum = 3
    pos_osum = 4
    pos_dsensum = 5
    pos_osensum = 6

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]
    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
    dsensum = xdata[:,
              sid + pos_dsensum * bins: sid + (pos_dsensum + 1) * bins]
    osensum = xdata[:,
              sid + pos_osensum * bins: sid + (pos_osensum + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, dsum, osum, dsensum, osensum))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


class WrapX(object):
    def __init__(self, doc, name, source_path, xshear_path, h0=70.,
                 omega_m=0.3, healpix_nside=64, nbin=15, rmin=0.02, rmax=30,
                 zlvals='default'):
        """
        Script wrapper for Erin's xshear (only lensfit mode!)

        Usage:
        -----------------
        1) specify text description of the operation
        2) give root filename for products
        3) specify source catalog path
        4) specify path for xshear executable

        5) Make sure the Cosmology (H0, omega_m) is correct
        6) Set appropriate radial scale

        the rest of the parameters can be left to default

        -----------------------------------------------------------------------
        The doc string should ideally contain a unique description of the
        used selection, so that the parameters can be grabbed for each entry
        -----------------------------------------------------------------------

        :param doc: documentation string (detailed)

        :param name: name (short) of the current setup

        :param source_path: path of the source catalog

        :param xshear_path: path of the xshear executable

        :param h0: Hubble constant base at z=0 (defines h or h_70 etc...)

        :param omega_m: matter density parameter at z=0

        :param healpix_nside: number of healpix cells to use along a side

        :param nbin: number of logarithmic radial bins

        :param rmin: innermost radial bin edge

        :param rmax: outermost radial bin edge

        :param zlvals: used redshift values (array/list)
        """

        self.name = name
        self.config_name = self.name + ".cfg"
        self.lens_name = self.name + "_lens.dat"
        self.log_name = self.name + "_log.p"
        self.res_name = self.name + "_res.dat"
        self.source_path = source_path
        self.xshear_path = xshear_path
        self.h0 = h0
        self.omega_m = omega_m
        self.healpix_nside = healpix_nside
        self.nbin = nbin
        self.rmin = rmin
        self.rmax = rmax
        self.doc = doc

        self.lcname = None
        self.lens_data = None

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

    def add_lens(self, ids, ra, dec, z, lcname):
        """creates lens data file"""

        field_mask_dummy = np.zeros(shape=ids.shape)
        lens = np.vstack((ids, ra, dec, z,
                          field_mask_dummy)).T
        self.lens_data = lens
        self.lcname = lcname

    def write_config(self, base_path):
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
        cfile = open(base_path + self.config_name, 'w')
        cfile.write(conf)
        cfile.close()

    def write_lens(self, base_path):
        """creates lens data file"""
        assert self.lens_data is not None
        fmt = ["%d", "%.18f", "%.18f", "%.18f", "%d"]
        np.savetxt(base_path + self.lens_name, self.lens_data, fmt=fmt)

    def write_log(self, base_path):
        """Creates log dictionary"""
        log_dict =  {
            "name": self.name,
            "config_name": self.config_name,
            "lens_name": self.lens_name,
            "log_name": self.log_name,
            "res_name": self.res_name,
            "source_path": self.source_path,
            "xshear_path": self.xshear_path,
            "h0": self.h0,
            "omega_m": self.omega_m,
            "healpix_nside": self.healpix_nside,
            "nbin": self.nbin,
            "rmin": self.rmin,
            "rmax": self.rmax,
            "doc": self.doc,
            "lcname": self.lcname,
            "zlvals": self.zlvals,
        }
        pickle.dump(log_dict, open(base_path + self.log_name, "wb"))

    def write_script(self, base_path, check=False):
        """executable xshear script"""

        script = "#!bin/bash \n"
        if check:
            script += "head -1 " + self.source_path + " | "
        else:
            script += "cat " + self.source_path + " | "

        script += self.xshear_path + " " + self.config_name + " " +\
                  self.lens_name + " > " + self.res_name

        sfile = open(base_path + self.name + ".sh", "w")
        sfile.write(script)
        sfile.close()

    def write_all(self, base_path):
        self.write_config(base_path)
        self.write_lens(base_path)
        self.write_script(base_path)
        self.write_log(base_path)

