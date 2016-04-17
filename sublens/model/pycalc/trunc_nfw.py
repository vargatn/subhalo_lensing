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


def f2dint(xx, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """2D density from 3D density"""
    return np.array([_f2dint(x, rs, rhos, rt, **kwargs) for x in xx])


def areafunc(x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Multiplies by polar Jacobian"""
    return _f2dint(x, rs, rhos, rt, **kwargs)[0] * x


def _f2dint_mean(x, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Mean surface density within a ring of radius x"""
    intres = quad(areafunc, 0.0, x, args=(rs, rhos, rt), limit=50, epsrel=1e-3)
    return 2. / (x*x) * np.array(intres)


def f2dint_mean(xx, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """Mean surface density within a ring of radius x"""
    return np.array([_f2dint_mean(x, rs, rhos, rt) for x in xx])


def ds_tnfw(xx, rs=1.0, rhos=1.0, rt=np.inf, **kwargs):
    """
    Delta Sigma of truncated NFW profile at a single value
    """
    return _f2dint_mean(xx, rs, rhos, rt) - _f2dint(xx, rs, rhos, rt)
