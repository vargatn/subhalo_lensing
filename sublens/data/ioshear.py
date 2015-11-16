"""
Class to document xshear calculations
"""


import numpy as np
from astropy.io import fits
import pickle


class ShearIO:

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
        assert self.xshear_path[:2] == "./", "invalid call for xshear!"

        script = "#!bin/bash \n"
        if check:
            script += "head -1 " + self.source_path + " | "
        else:
            script += "cat " + self.source_path + " | "

        script += self.xshear_path + " " + self.config_name + " " +\
                  self.lens_name + " > " + self.res_name

        # script = "#!bin/bash \n" + "cat " + self.source_path + " |" + \
        #     self.xshear_path + " " + self.config_name + " " + self.lens_name +\
        #          " >" + self.res_name

        sfile = open(self.name + ".sh", "w")
        sfile.write(script)
        sfile.close()






