"""
Parameter scaling relationships

THESE FUNCTIONS SHOULD BE PICKLEABLE
"""

import numpy as np


def lm200_rykoff_orig(l, **kwargs):
    """
    Matches redmapper richness to cluster m200

    Rykoff assumes h70, this is converted to h100

    :param l: richness
    :return: M200 [M_sun / h100]
    """
    mpivot = 1e14 * 0.7  # msun / h100
    m200 = np.exp(1.48 + 1.06 * np.log(l / 60.)) * mpivot

    return m200


def cscale_duffy(m200=0.0, z=0.0, **kwargs):
    """
    Calculates NFW concentration based on M200 and redshift

    I use the halo definition with r200 from Duffy et al. 2008

    h is propagated through

    :param m200: nfw mass [M_sun / h100]
    :param z: halo redshift
    :return: concentration
    """
    mpivot = 2. * 1e12  # Msun / h100

    a200 = 6.71
    b200 = -0.091
    c200 = -0.44

    c = a200 * (m200 / mpivot) ** b200 * (1. + z) ** c200
    return c

