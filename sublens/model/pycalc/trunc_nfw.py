"""
Calculates DeltaSigma profile for a truncated NFW halo
"""

import math
import numpy as np
from scipy.integrate import quad


def nfw3d(X, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """
    3D-density of Navarro-Frenk-White profile
    """
    r = X * rs
    val = 0.0
    if r <= rt:
        val = rhos / (X * (1 + X)**2.)
    return val


def project(Z, x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Re-parametrize 3D density with Z and R_2D"""
    X = math.sqrt(x**2. + Z**2.)
    return nfw3d(X, rs, rhos, rt, **kwargs)


def _f2dint(x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """2D density from 3D density"""
    intres = quad(project, 0.0, np.inf, args=(x, rs, rhos, rt))
    return 2. * np.array(intres) * rs


def areafunc(x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Multiplies by polar Jacobian"""
    return _f2dint(x, rs, rhos, rt, **kwargs)[0] * x


def _f2dint_mean(x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Mean surface density within a ring of radius x"""
    intres = quad(areafunc, 0.0, x, args=(rs, rhos, rt), limit=50, epsrel=1e-3)
    return 2. / (x*x) * np.array(intres)


def tnfw(r, rs, rho_s, rt=np.inf):
    """
    Delta Sigma of truncated NFW profile at a single value
    """
    x = r / rs
    return _f2dint_mean(x, rs, rho_s, rt)[0] - _f2dint(x, rs, rho_s, rt)[0]


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