"""
Calculates DeltaSigma profile for a off-centered NFW halo
"""

import math
from scipy.integrate import quad, dblquad


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


def oc_intarg(phi, r, rs, rho_s, dist):
    """
    Argument expression for Integral

    Offcenters the shear profile usig the fact that it is a spin-2 field

    :param phi: polar angle
    :param r: radius in physical units
    :param rs: scale radius in physical units
    :param rho_s: \rho_c * \delta_c
    :param dist: offset distance
    :return: density
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

    # the * r is there for the polar Jacobian which is needed
    #  for the path integral
    dst_cen = nfw_deltasigma(rr, rs, rho_s)

    dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
    return dst


def oc_nfw(r, rs, rho_s, dist, *args, **kwargs):
    """Calculates the angle-averaged DeltaSigma at polar radius r"""
    dsum = quad(oc_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist), points=(0.0,))[0]
    return dsum / (2. * math.pi)


def _oc_nfw_ring(r, rs, rho_s, dist, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    dsum = quad(oc_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist), points=(0.0,))[0]
    return dsum * r


def oc_nfw_ring(r0, r1, rs, rho_s, dist, split=True, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    dsum = 0.0
    if split * (r0 < dist <=r1):
        dsum0 = quad(_oc_nfw_ring, r0, dist, args=(rs, rho_s, dist))[0]
        dsum1 = quad(_oc_nfw_ring, dist, r1, args=(rs, rho_s, dist))[0]
        dsum = dsum0 + dsum1
    else:
        dsum = quad(_oc_nfw_ring, r0, r1, args=(rs, rho_s, dist))[0]

    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring


def _nfw_ring(r, rs, rho_s, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    return nfw_deltasigma(r, rs, rho_s, *args, **kwargs) * 2. * math.pi * r


def nfw_ring(r0, r1, rs, rho_s, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    dsum = quad(_nfw_ring, r0, r1, args=(rs, rho_s))[0]
    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring










