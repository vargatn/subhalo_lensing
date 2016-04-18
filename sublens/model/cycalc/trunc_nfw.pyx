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


def _f2dint(x, rs=1.0, rho_s=1.0, rt=np.inf):
    """2D density from 3D density"""
    intres = quad(project, 0.0, np.inf, args=(x, rs, rho_s, rt))
    return 2. * np.array(intres) * rs


def f2dint(xx, rs=1.0, rho_s=1.0, rt=np.inf):
    """2D density from 3D density"""
    return np.array([_f2dint(x, rs, rho_s, rt) for x in xx])


def areafunc(x, rs=1.0, rho_s=1.0, rt=np.inf):
    """Multiplies by polar Jacobian"""
    return _f2dint(x, rs, rho_s, rt)[0] * x  # notice the * x  !!!


def _f2dint_mean(x, rs=1.0, rho_s=1.0, rt=np.inf):
    """Mean surface density within a ring of radius x"""
    intres = quad(areafunc, 0.0, x, args=(rs, rho_s, rt), limit=50, epsrel=1e-3)
    return 2. / (x*x) * np.array(intres)


def f2dint_mean(xx, rs=1.0, rho_s=1.0, rt=np.inf):
    """Mean surface density within a ring of radius x"""
    return np.array([_f2dint_mean(x, rs, rho_s, rt) for x in xx])


def tnfw(x, rs=1.0, rho_s=1.0, rt=np.inf):
    """
    Delta Sigma of truncated NFW profile at a single value
    """
    return _f2dint_mean(x, rs, rho_s, rt)[0] - _f2dint(x, rs, rho_s, rt)[0]


def tnfw_ring(r0, r1, rs, rho_s, rt=np.inf, split=True):
    """
    Ring averaged truncated NFW profile
    """
    x0 = r0 / rs
    x1 = r1 / rs
    xt = rt / rs
    dsum = 0.0
    if split * (r0 < rt <=r1):
        dsum0 = quad(tnfw, x0, xt, args=(rs, rho_s, rt), epsrel=1e-3)[0]
        dsum1 = quad(tnfw, xt, x1, args=(rs, rho_s, rt), epsrel=1e-3)[0]
        dsum = dsum0 + dsum1
    else:
        dsum = quad(tnfw, x0, x1, args=(rs, rho_s, rt), epsrel=1e-3)[0]
    return dsum / (x1 - x0)

