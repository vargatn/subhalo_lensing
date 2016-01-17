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

from ..model.haloc import Halo

# class
