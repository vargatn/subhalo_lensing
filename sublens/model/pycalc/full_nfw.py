"""
Calculates DeltaSigma profile for a off-centered NFW halo
"""

import math
from scipy.integrate import quad


def nfw_deltasigma(r, rs, rho_s, *args, **kwargs):
    """
    Delta-Sigma profile of the NFW profile (exact formula)

        Equation from:
        Gravitational Lensing by NFW Halos
        Wright, Candace Oaxaca; Brainerd, Tereasa G.
        http://adsabs.harvard.edu/abs/2000ApJ...534...34W


    :param r: radius in physical units

    :param rs: scale radius in physical units

    :param rho_s: \rho_c * \delta_c

    :return: value of tangential shear at distance r
    """

    x = r / rs
    shear = 0
    if 0. < x < 1.:
        atanh_arg = math.atanh(math.sqrt((1. - x) / (1. + x)))
        shear = rs * rho_s * (
        (8. * atanh_arg / (
            x ** 2. * math.sqrt(1. - x ** 2.)) +
         4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1.) +
         (4. * atanh_arg) / (
             (x ** 2. - 1.) * math.sqrt(1. - x ** 2.))))
    elif x == 1.:
        shear = rs * rho_s * (10. / 3. + 4. * math.log(1. / 2.))
    elif x > 1.:
        atan_arg = math.atan(math.sqrt((x - 1.) / (1. + x)))
        shear = rs * rho_s * (
        (8. * atan_arg/ (
            x ** 2. * math.sqrt(x ** 2. - 1.)) +
         4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1) +
         (4. * atan_arg) / (
             (x ** 2. - 1.) ** (3. / 2.))))
    return shear


def _nfw_ring(r, rs, rho_s, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    return nfw_deltasigma(r, rs, rho_s, *args, **kwargs) * 2. * math.pi * r


def nfw_ring(r0, r1, rs, rho_s, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    dsum = quad(_nfw_ring, r0, r1, args=(rs, rho_s))[0]
    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring
