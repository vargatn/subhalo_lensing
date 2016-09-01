"""
Calculates DeltaSigma profile for a point mass
"""

import math
from scipy.integrate import quad



def pointmass(r, mpoint, *args, **kwargs):
    """
    Delta-Sigma profile of a point mass

    :param r: radius in physical units

    :return: value of tangential shear at distance r
    """

    shear = mpoint /r /r
    return shear


def _point_ring(r, mpoint, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    return pointmass(r, mpoint, *args, **kwargs) * 2. * math.pi * r


def point_ring(r0, r1, mpoint, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    dsum = quad(_point_ring, r0, r1, args=(mpoint,))[0]
    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring