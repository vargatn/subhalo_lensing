import time
import math
import numpy as np
from ..model.misc import nfw_pars
import jdnfw.nfw as jd_nfw

from scipy import integrate as integr


def nfw_prof_noint(c200, m200, z, edges):
    rs, rho_s, r200 = nfw_pars(m200, c200, z)
    # print(rs, rho_s)
    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, edge in enumerate(edges[:-1])])
    cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                     (edges[i + 1] ** 2. - edges[i] ** 2.)
                     for i, edge in enumerate(edges[:-1])])

    # print(cens)
    # print(rs)
    ds = np.array([nfw_shear_t(cen, rs, rho_s) / cen
                   for i, cen in enumerate(cens)])

    ds = ds / 1e12
    # ds = ds / 1e12 * np.pi * 2.

    return cens, ds


def nfw_shear_t(r, rs, rho_s):
    """
    RAW centered NFW halo

    Equation from:
    Gravitational Lensing by NFW Halos
    Wright, Candace Oaxaca; Brainerd, Tereasa G.
    http://adsabs.harvard.edu/abs/2000ApJ...534...34W

    :return: value of tangential shear_old at distance x
    """

    x = r / rs

    shear = 0

    if 0. < x < 1.:
        shear = rs * rho_s * ((8. * math.atanh(math.sqrt((1. - x) / (1. + x))) / (
            x ** 2. * math.sqrt(1. - x ** 2.)) +
                        4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1.) +
                        (4. * math.atanh(math.sqrt((1. - x) / (1. + x)))) / (
                            (x ** 2. - 1.) * math.sqrt(1. - x ** 2.))))

    elif x == 1.:
        shear = rs * rho_s *(10. / 3. + 4. * math.log(1. / 2.))

    elif x > 1.:
        shear = rs * rho_s *((8. * math.atan(math.sqrt((x - 1.) / (1. + x))) / (
            x ** 2. * math.sqrt(x ** 2. - 1.)) +
                        4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1) +
                        (4. * math.atan(math.sqrt((x - 1.) / (1. + x)))) / (
                            (x ** 2. - 1.) ** (3. / 2.))))

    return shear * r
