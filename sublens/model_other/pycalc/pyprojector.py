"""
Python based integrals
"""

import math
import numpy as np
from scipy.integrate import quad, dblquad


def dens2d(x, y, func, params, *args, **kwargs):
    """
    Projects the 3D density to a 2D plane (x, y)

    :param x: x-coord
    :param y: y-coord
    :param func: 3D density function which expects the *params as argument
    :param params: tuple of paramaters
    :return: 2D projected density at (x, y)
    """
    intres, interr = quad(func, 0.0, np.inf, args=params)
    return intres * 2.


def polarf(phi, r, func, params):
    """Calls the 2D density in polar form"""
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return dens2d(x, y, func, params) * r


def mean_r(r, func, params, tuple_check=True):
    """integrating in a counter-clockwise direction"""
    if tuple_check:
        params = tuple(params)
    intres = quad(polarf, 0.0, 2 * math.pi, args=((r,) + params))
    return np.array(intres[0]) / (2. * math.pi * r)


def mean_rdisk(r, func, params):
    """Integrating over a disk of radius r"""
    def minval(phi):
        return 0.0

    def maxval(phi):
        return r

    intres = dblquad(polarf, 0.0, 2. * math.pi, minval, maxval, args=params)
    return intres[0] / (math.pi * r * r)


def deltasigma(r, func, params, *args, **kwargs):
    return mean_rdisk(r, func, params) - mean_r(r, func, params)
