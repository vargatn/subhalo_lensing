"""
Python-based calculations
"""

import math


def oc_transform(phi, r, dist):
    """
    Argument expression for Integral

    Off-centers the shear profile usig the fact that it is a spin-2 field

    :param phi: polar angle

    :param r: radius in physical units

    :param dist: offset distance

    :return: (rr, term1, term2) terms for the transformation
    """
    assert r > 0.
    # creating transformation variables
    dist2 = dist * dist
    rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
    rr = math.sqrt(rr2)
    term1 = (dist2 +
             r * (2. * r * math.cos(phi) ** 2. -
                  2. * dist * math.cos(phi) - r)) / rr2
    term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2

    return rr, term1, term2
