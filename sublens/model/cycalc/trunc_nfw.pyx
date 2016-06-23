"""
Calculates DeltaSigma profile for a truncated NFW halo
"""

import math
import numpy as np
from scipy.integrate import quad


# NFW function
cdef double nfw3d(double X, double rs, double rho_s, double rt):
    """
    3D-density of Navarro-Frenk-White profile
    """
    cdef double r = X * rs
    cdef double val = 0.0
    if r <= rt:
        val = rho_s / (X * (1. + X) * (1. + X))
    return val


cpdef double project(double Z, double x, double rs, double rho_s,
                    double rt):
    """Re-parametrize 3D density with Z and R_2D"""
    cdef double X = math.sqrt(x**2. + Z**2.)
    return nfw3d(X, rs, rho_s, rt)


def _f2dint(x, rs, rho_s, rt=np.inf):
    """2D density from 3D density"""
    intres = quad(project, 0.0, np.inf, args=(x, rs, rho_s, rt))[0]
    return 2. * intres * rs


def areafunc(x, rs, rho_s, rt=np.inf):
    """Multiplies by polar Jacobian"""
    return _f2dint(x, rs, rho_s, rt) * x  # notice the * x  !!!


def _f2dint_mean(x, rs, rho_s, rt=np.inf):
    """Mean surface density within a ring of radius x"""
    xt = rt / rs
    intres0 = quad(areafunc, 0.0, xt, args=(rs, rho_s, rt), limit=50,
                   epsrel=1e-3, epsabs=0)[0]
    intres1 = quad(areafunc, xt, x, args=(rs, rho_s, rt), limit=50,
                   epsrel=1e-3, epsabs=0)[0]
    intres = intres0 + intres1
    return 2. / (x*x) * np.array(intres)


def tnfw(r, rs, rho_s, rt=np.inf):
    """
    Delta Sigma of truncated NFW profile at a single value
    """
    x = r / rs
    return _f2dint_mean(x, rs, rho_s, rt) - _f2dint(x, rs, rho_s, rt)


def _tnfw_ring(r, rs, rho_s, rt=np.inf, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    return tnfw(r, rs, rho_s, rt) * 2. * math.pi * r


def tnfw_ring(r0, r1, rs, rho_s, rt=np.inf, split=True):
    """
    Ring averaged truncated NFW profile
    """
    dsum = 0.0
    if split * (r0 < rt <=r1):
        dsum0 = quad(_tnfw_ring, r0, rt, args=(rs, rho_s, rt), epsrel=1e-3)[0]
        dsum1 = quad(_tnfw_ring, rt, r1, args=(rs, rho_s, rt), epsrel=1e-3)[0]
        dsum = dsum0 + dsum1
    else:
        dsum = quad(_tnfw_ring, r0, r1, args=(rs, rho_s, rt), epsrel=1e-3)[0]
    aring = math.pi * (r1**2. - r0 **2.)
    return dsum / aring

