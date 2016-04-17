"""
Lensing related functions
"""

import math


def nfw_3d_zxy(z, x, y, rs, rho_s, x0, y0, *args, **kwargs):
    """
    NFW 3D density profile

    :param z: z-coord
    :param x: x-coord
    :param y: y-coord
    :param rs: scale radius
    :param rho_s: amplitude
    :param x0: xcenter of the halo
    :param y0: ycenter of the halo
    :return: 3D density
    """
    x = math.sqrt((x - x0) * (x - x0) + (y - y0) * (x - y0) + z * z) / rs
    return rho_s / (x * (1. + x))


def nfw_3d_rdens(r, rs, rho_s, *args, **kwargs):
    """
    NFW 3D density profile

    :param r: radius from halo center
    :param rs:
    :param rho_s:
    :return: 3D density
    """
    x = r / rs
    return rho_s / (x * (1. + x))


def nfw_shear_t(r, rs, rho_s, *args, **kwargs):
    """
    RAW centered NFW halo

    preferred units: Mpc and MSun

    No explicit unit checking done!! make sure that the units of r, rs and
    rho_s are consistent!

    Equation from:
    Gravitational Lensing by NFW Halos
    Wright, Candace Oaxaca; Brainerd, Tereasa G.
    http://adsabs.harvard.edu/abs/2000ApJ...534...34W

    :param r: radius from center of halo
    :param rs: scale radius
    :param rho_s: amplitude
    :return: value of tangential shear_old at distance x
    """

    x = r / rs
    shear = 0

    if 0. < x < 1.:
        shear = rs * rho_s * (
        (8. * math.atanh(math.sqrt((1. - x) / (1. + x))) / (
            x ** 2. * math.sqrt(1. - x ** 2.)) +
         4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1.) +
         (4. * math.atanh(math.sqrt((1. - x) / (1. + x)))) / (
             (x ** 2. - 1.) * math.sqrt(1. - x ** 2.))))

    elif x == 1.:
        shear = rs * rho_s * (10. / 3. + 4. * math.log(1. / 2.))

    elif x > 1.:
        shear = rs * rho_s * (
        (8. * math.atan(math.sqrt((x - 1.) / (1. + x))) / (
            x ** 2. * math.sqrt(x ** 2. - 1.)) +
         4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1) +
         (4. * math.atan(math.sqrt((x - 1.) / (1. + x)))) / (
             (x ** 2. - 1.) ** (3. / 2.))))
    return shear
