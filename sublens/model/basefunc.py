"""
Basic density functions
"""

import math


def tnfw_dens3d(z, x, y, rs, rho_s, rt, *args, **kwargs):
    """Truncated density profile for NFW, hard cut at rt physical radius"""
    dens = 0.0
    r = math.sqrt(x * x + y * y + z * z)
    if r <= rt:
        x = r / rs
        dens = rho_s / (x * (1. + x) * (1. + x))
    return dens


def nfw_dens3d(z, x, y, rs, rho_s, *args, **kwargs):
    """3D density for NFW profile"""
    x = math.sqrt(x * x + y * y + z * z) / rs
    return rho_s / (x * (1. + x) * (1. + x))


def nfw_rdens3d(X, rs, rho_s, *args, **kwargs):
    """3D radial density for NFW profile"""
    return rho_s / (X * (1. + X) * (1. + X))


def nfw_deltasigma(r, rs, rho_s, *args, **kwargs):
    """Delta-Sigma profile of the NFW profile (exact formula)

        Equation from:
        Gravitational Lensing by NFW Halos
        Wright, Candace Oaxaca; Brainerd, Tereasa G.
        http://adsabs.harvard.edu/abs/2000ApJ...534...34W

    :return: value of tangential shear_old at distance r
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



