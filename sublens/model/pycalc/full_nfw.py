"""
Calculates DeltaSigma profile for a off-centered NFW halo
"""

import math
from scipy.integrate import quad

from ..pycalc import oc_transform


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


def _oc_nfw_intarg(phi, r, rs, rho_s, dist):
    """argument for the integral of Off-centered shear"""
    rr, term1, term2 = oc_transform(phi=phi, r=r, dist=dist)
    dst_cen = nfw_deltasigma(rr, rs, rho_s)
    dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
    return dst


def oc_nfw(r, rs, rho_s, dist, *args, **kwargs):
    """Calculates the angle-averaged DeltaSigma at polar radius r"""
    dsum = quad(_oc_nfw_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist), points=(0.0,), epsabs=0,
                epsrel=1e-5)[0]
    return dsum / (2. * math.pi)


def _oc_nfw_ring(r, rs, rho_s, dist, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    dsum = quad(_oc_nfw_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist), points=(0.0,),
                epsabs=0, epsrel=1e-5)[0]
    return dsum * r


def oc_nfw_ring(r0, r1, rs, rho_s, dist, split=True, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    if split * (r0 < dist <=r1):
        dsum0 = quad(_oc_nfw_ring, r0, dist, args=(rs, rho_s, dist),
                     epsabs=0, epsrel=1e-4)[0]
        dsum1 = quad(_oc_nfw_ring, dist, r1, args=(rs, rho_s, dist),
                     epsabs=0, epsrel=1e-4)[0]
        dsum = dsum0 + dsum1
    else:
        dsum = quad(_oc_nfw_ring, r0, r1, args=(rs, rho_s, dist),
                    epsabs=0, epsrel=1e-4)[0]

    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring

