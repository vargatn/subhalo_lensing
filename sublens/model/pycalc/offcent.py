"""
Offcentered profiles
"""

import numpy as np
import math
from .full_nfw import nfw_deltasigma
from scipy.integrate import quad
from ..pycalc import oc_transform
from .halo2 import H2calc

# This does not really work...
class MainHalo(object):
    """Group halo NFW with 2-halo term"""
    def __init__(self, z, spectra, cosmo, scalar_ind=0.96):
        self.z = z
        self.h2 = H2calc(z, spectra=spectra, cosmo=cosmo,
                         scalar_ind=scalar_ind)

    def point_ds(self, r, rs, rho_s, b):
        """Single circle averaged shear at radius r"""
        shear = nfw_deltasigma(r, rs, rho_s) +\
                self.h2.wint(r)[0] * self.h2.prefac() * b
        return shear

    def _ring(self, r, rs, rho_s, b):
        intres = self.point_ds(r, rs, rho_s, b) * r * 2. * np.pi
        return intres

    def ring_ds(self, r0, r1, rs, rho_s, b):
        """ordinary ring averaged deltasigma"""
        dsum = np.array(quad(self._ring, r0, r1, args=(rs, rho_s, b), epsabs=0,
                        epsrel=1e-3))
        aring = np.pi * (r1 ** 2. - r0 ** 2.)
        return dsum[0] / aring

    def _oc_intarg(self, phi, r, dist, rs, rho_s, b):
        """argument for the integral of Off-centered shear"""
        rr, term1, term2 = oc_transform(phi=phi, r=r, dist=dist)
        dst_cen = self.point_ds(rr, rs, rho_s, b)
        dst = dst_cen * (term1 * math.cos(2. * phi) +
                         term2 * math.sin(2. * phi))
        return dst

    def oc_ds(self, r, dist, rs, rho_s, b):
        """Calculates the angle-averaged DeltaSigma at polar radius r"""
        dsum = quad(self._oc_intarg, -math.pi, math.pi,
                    args=(r, dist, rs, rho_s, b), points=(0.0,), epsabs=0,
                    epsrel=1e-5)[0]
        return dsum / (2. * math.pi)

    def _oc_ring(self, r, dist, rs, rho_s, b, *args, **kwargs):
        """Calculates the angle-integrated DeltaSigma at polar radius r"""
        dsum = quad(self._oc_intarg, -math.pi, math.pi,
                    args=(r, dist, rs, rho_s, b), points=(0.0,),
                    epsabs=0, epsrel=1e-5)[0]
        return dsum * r

    def oc_ring(self, r0, r1, dist, rs, rho_s, b, split=True,
                     *args, **kwargs):
        """Calculates the ring-averaged DeltaSigma between r0 and r1"""
        args = (dist, rs, rho_s, b)
        if split * (r0 < dist <= r1):
            dsum0 = quad(self._oc_ring, r0, dist, args=args,
                         epsabs=0, epsrel=1e-4)[0]
            dsum1 = quad(self._oc_ring, dist, r1, args=args,
                         epsabs=0, epsrel=1e-4)[0]
            dsum = dsum0 + dsum1
        else:
            dsum = quad(self._oc_ring, r0, r1, args=args,
                        epsabs=0, epsrel=1e-4)[0]

        aring = math.pi * (r1 ** 2. - r0 ** 2.)
        return dsum / aring


