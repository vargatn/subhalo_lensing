import time
import math
import numpy as np
from ..model.misc import nfw_pars
import jdnfw.nfw as jd_nfw
from ..model.misc import default_cosmo
from ..model.autoscale import cscale_duffy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from scipy import integrate as integr


from ..model.halo1 import CentHaloNFW
from ..model.halo2 import SecHalo
from ..model.halo2 import SecHaloOld


class Halo(object):
    def __init__(self, pscont, cscale=cscale_duffy, cosmo=None):

        # primary halo object
        self.ph = CentHaloNFW(cosmo)

        # secondary halo object
        self.sh = SecHalo(pscont, cosmo=cosmo)

    def ds(self, m, z, rr):
        assert z > 0.05, 'Two halo term only works for z > 0'

        rarr, ds2 = self.sh.ds(m, z, rr)
        rarr, ds1 = self.ph.ds(m, z, rr)

        return rr, ds1 + ds2


class HaloOld(object):
    def __init__(self, pscont, w2log, cscale=cscale_duffy, cosmo=None):
        """creates centered halo object"""

        # primary halo object
        self.ph = CentHaloNFW(cosmo)

        # secondary halo object
        self.sh = SecHaloOld(pscont, w2log, cosmo=cosmo)


    def ds(self, m, z):
        assert z > 0.05, 'Two halo term only works for z > 0'

        rr, ds2 = self.sh.ds(m, z)
        rr, ds1 = self.ph.ds(m, z, rr)

        return rr, ds1 + ds2
