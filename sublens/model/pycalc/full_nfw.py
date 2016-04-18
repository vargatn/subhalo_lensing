"""
Calculates DeltaSigma profile for a off-centered NFW halo
"""

import math
from scipy.integrate import quad


def nfw_deltasigma(r, rs, rho_s, *args, **kwargs):
    """Delta-Sigma profile of the NFW profile (exact formula)

        Equation from:
        Gravitational Lensing by NFW Halos
        Wright, Candace Oaxaca; Brainerd, Tereasa G.
        http://adsabs.harvard.edu/abs/2000ApJ...534...34W

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


def oc_intarg(phi, r, rs, rho_s, dist):
    """
    Argument expression for Integral

    Offcenters the shear profile usig the fact that it is a spin-2 field

    :param phi: polar angle
    :param r: polar radius
    :param rs:
    :param rho_s:
    :param dist: offset distance
    :return:
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

    # the * r is there for the polar Jacobian
    dst_cen = nfw_deltasigma(rr, rs, rho_s) * r

    dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
    return dst


def oc_nfw(r, rs, rho_s, dist, *args, **kwargs):
    """Calculates the average DeltaSigma at polar radius r"""
    circ = 2. * math.pi * r
    ds = quad(oc_intarg, -math.pi, math.pi, args=(r, rs, rho_s, dist),
               points=(0.0,))
    return ds[0] / circ


def oc_nfw_ring(r0, r1, rs, rho_s, dist, split=True, *args, **kwargs):
    dsum = 0.0
    if split * (r0 < dist <=r1):
        dsum0 = quad(oc_nfw, r0, dist, args=(rs, rho_s, dist))[0]
        dsum1 = quad(oc_nfw, dist, r1, args=(rs, rho_s, dist))[0]
        dsum = dsum0 + dsum1
    else:
        dsum = quad(oc_nfw, r0, r1, args=(rs, rho_s, dist))[0]

    return dsum / (r1 - r0)


def nfw_ring(r0, r1, rs, rho_s, *args, **kwargs):
    dsum = quad(nfw_deltasigma, r0, r1, args=(rs, rho_s))[0]
    return dsum / (r1 - r0)

