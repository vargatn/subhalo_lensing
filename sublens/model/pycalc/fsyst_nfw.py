"""
Calculates DeltaSigma profile for a off-centered NFW halo
"""

import math
from scipy.integrate import quad

from ..pycalc import oc_transform
from .full_nfw import nfw_deltasigma


def funit(r):
    return 1.0


def _oc_fsyst_intarg(phi, r, dist, fsyst=funit):
    #     print(fsyst)
    rr, term1, term2 = oc_transform(phi=phi, r=r, dist=dist)
    dst = fsyst(rr)
    return dst


def _oc_nfw_intarg(phi, r, rs, rho_s, dist, fsyst=funit):
    """argument for the integral of Off-centered shear"""
    rr, term1, term2 = oc_transform(phi=phi, r=r, dist=dist)
    dst_cen = nfw_deltasigma(rr, rs, rho_s) * fsyst(rr)
    dst = dst_cen * (term1 * math.cos(2. * phi) + term2 * math.sin(2. * phi))
    return dst


def soc_nfw(r, rs, rho_s, dist, fsyst, *args, **kwargs):
    """Calculates the angle-averaged DeltaSigma at polar radius r"""
    dsum = quad(_oc_nfw_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist, fsyst), points=(0.0,), epsabs=0,
                epsrel=1e-5)[0]
    area = quad(_oc_fsyst_intarg, -math.pi, math.pi,
                args=(r, dist, fsyst), points=(0.0,), epsabs=0,
                epsrel=1e-5)[0]
    return dsum / area


def _oc_nfw_ring(r, rs, rho_s, dist, fsyst, *args, **kwargs):
    """Calculates the angle-integrated DeltaSigma at polar radius r"""
    dsum = quad(_oc_nfw_intarg, -math.pi, math.pi,
                args=(r, rs, rho_s, dist, fsyst), points=(0.0,),
                epsabs=0, epsrel=1e-5)[0]
    return dsum * r


def _oc_fsyst_ring(r, dist, fsyst, *args, **kwargs):
    dsum = quad(_oc_fsyst_intarg, -math.pi, math.pi,
                args=(r, dist, fsyst), points=(0.0,),
                epsabs=0, epsrel=1e-5)[0]
    return dsum * r


def soc_nfw_ring(r0, r1, rs, rho_s, dist, fsyst, split=True, *args, **kwargs):
    """Calculates the ring-averaged DeltaSigma between r0 and r1"""
    if split * (r0 < dist <= r1):
        dsum0 = quad(_oc_nfw_ring, r0, dist, args=(rs, rho_s, dist, fsyst),
                     epsabs=0, epsrel=1e-4)[0]
        dsum1 = quad(_oc_nfw_ring, dist, r1, args=(rs, rho_s, dist, fsyst),
                     epsabs=0, epsrel=1e-4)[0]
        dsum = dsum0 + dsum1

        area0 = quad(_oc_fsyst_ring, r0, dist, args=(dist, fsyst),
                     epsabs=0, epsrel=1e-4)[0]
        area1 = quad(_oc_fsyst_ring, dist, r1, args=(dist, fsyst),
                     epsabs=0, epsrel=1e-4)[0]
        area = area0 + area1

    else:
        dsum = quad(_oc_nfw_ring, r0, r1, args=(rs, rho_s, dist, fsyst),
                    epsabs=0, epsrel=1e-4)[0]
        area = quad(_oc_fsyst_ring, r0, r1, args=(dist, fsyst),
                    epsabs=0, epsrel=1e-4)[0]
    # aring = math.pi * (r1**2. - r0 **2.)
    return dsum / area

