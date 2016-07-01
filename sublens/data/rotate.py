"""
Spherical transformations and 3D rotations
"""

import healpy as hp
import numpy as np


def caparea(theta):
    """Area of a spherical cap (radians)"""
    val = 2. * np.pi * (1. - np.cos(theta))
    return val


def rotate(vv, kk, angles):
    """
    Rotation on a sphere unit with vectors using the exact formula

    parameters:
    -------------
    :param vv: position to be rotated

    :param kk: center for rotation

    :param angles: angle for rotation in *radians*

    :returns: rotated vectors
    """
    term1 = vv * np.cos(angles[:, np.newaxis])
    term2 = np.cross(kk, vv, axis=1) * np.sin(angles[:, np.newaxis])
    term3 = kk * np.array([np.inner(k, v) for k, v in zip(kk, vv)]
                          )[:, np.newaxis] *\
            (1. - np.cos(angles[:, np.newaxis]))
    return term1 + term2 + term3


def gcdist(pa, pb, mode='precise', deg=True):
    """
    Great circle distance

    NOTE: there is no reason to use anything other than "precise"...

    :param pa: coordinates N x (RA, DEC)

    :param pb: coordinates N x (RA, DEC)

    :param mode: calculatinon formula (may affect precision)

    :param deg: True: degrees, false: radians

    :return: array of distances
    """

    if mode == 'simple':
        val = _sphere_dist(pa, pb, deg=deg)
    elif mode == 'precise':
        val = hp.rotator.angdist(pa, pb, lonlat=True)
    else:
        raise NotImplementedError

    return val


def _sphere_dist(pa, pb, deg=True):
    """Simple great circle dist in RADIANS"""
    if deg:
        pa = np.array(pa, copy=True, dtype='float64') * np.pi / 180.
        pb = np.array(pb, copy=True, dtype='float64') * np.pi / 180.

    dl = np.abs(pa[:, 0] - pb[:, 0])
    dsigm = np.arccos(np.sin(pa[:, 1]) * np.sin(pb[:, 1]) +
                      np.cos(pa[:, 1]) * np.cos(pb[:, 1]) * np.cos(dl))
    if deg:
        dsigm *= 180. / np.pi

    return dsigm


def spher2cart(spher, deg=True):
    """Spherical to cartesian"""
    spher = np.array(spher, copy=True, dtype='float64')
    if deg:
        spher *= np.pi / 180.
    x = np.cos(spher[0]) * np.cos(spher[1])
    y = np.sin(spher[0]) * np.cos(spher[1])
    z = np.sin(spher[1])

    cart = np.array([x, y, z])
    return cart


def cart2spher(cart, deg=True):
    """Cartesian to Spherical"""
    cart = np.array(cart, copy=True, dtype='float64')
    r = np.sqrt(np.sum(cart ** 2.))
    az = np.arctan2(cart[1], cart[0])
    lat = np.arcsin(cart[2] / r)

    if az < 0.0:
        az += np.pi

    spher = np.array([az, lat])
    if deg:
        spher  *= 180. / np.pi
    return spher
