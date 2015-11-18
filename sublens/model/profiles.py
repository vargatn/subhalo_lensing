"""
Density and \Delta\Sigma profile functions


The core functionality is focused on different versions of NFW profiles.
Also includes conversion for (m_200, c_200, z) ->  (r_s, rho_s)

Contents:
----------
*   3D NFW density in h^2 M_\odot / Mpc^3
*   2D NFW density in h M_\odot / Mpc^2
*   \Delta\Sigma of NFW in h M_\odot / Mpc^2

*   conversion tool for NFW parameters
"""

import math
import numpy as np
from ..model.misc import nfw_pars


def poolable_example_profile(params):
    return example_profile(**params)


def example_profile(m200, c200, z, lims=(0.02, 2.0), num=30):
    """
    Creates an example profile based on a 3D NFW density profile
    :param m200: Virial mass
    :param c200: concentration
    :param z: redshift
    :param lims: radiaé range of the profile
    :param num: number of points in the profile
    :return: radial density profile
    """
    # evaluation points
    rarr = np.logspace(np.log10(lims[0]), np.log10(lims[1]), num=num)

    # density at each point
    darr = np.array([nfw_3d(r, m200, c200, z) for r in rarr])

    return rarr, darr


def nfw_3d(r, m200, c200, z):
    """
    3D Navarro-Frenk-White function

    :param m200: Virial mass
    :param c200: Concentration relative to \rho_{crit}
    :param z: redshift
    :return: density
    """
    # obtaining profile parameters
    rs, rho_s, r200 = nfw_pars(m200, c200, z)

    x = r / rs
    dens =  rho_s / (x * (1 + x) ** 2.)
    return dens / 1e12