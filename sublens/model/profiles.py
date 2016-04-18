"""
Interface for DeltaSigma profiles
"""

# import pyximport; pyximport.install()
import numpy as np

from ..model import default_cosmo
from ..model.astroconvert import nfw_params

from ..model.cycalc import ds_tnfw
from ..model.cycalc import ds_tnfw_ring
# from ..model.pycalc import fds_tnfw

from ..model.pycalc import nfw_deltasigma
from ..model.pycalc import ds_nfw_ring

from ..model.pycalc import ds_oc_nfw
from ..model.pycalc import ds_oc_nfw_ring

from ..model.pycalc.full_nfw import direct_ring_nfw
from ..model.pycalc.full_nfw import direct_circ_nfw
from ..model.pycalc.full_nfw import direct_ring_oc_nfw
from ..model.pycalc.full_nfw import indirect_ring_oc_nfw

# TODO write documentation

class DeltaSigmaProfile(object):
    def __init__(self, cosmo=None):
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.

        self.profpars = []
        self.parnames = []
        self.pardict = {}

        self.rr = None
        self.redges = None
        self.ds = None

        self._prepared = False

    def prepare(self, **kwargs):
        raise NotImplementedError

    def point_ds(self, r, *args, **kwargs):
        raise NotImplementedError

    def deltasigma(self, rr, *args, **kwargs):
        assert self._prepared

        if np.iterable(rr):
            ds = np.array([self.point_ds(r) for r in rr])
        else:
            ds = np.array([self.point_ds(rr)])
            rr = np.array([rr])

        self.rr = rr
        self.ds = ds

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        raise NotImplementedError

    def rbin_deltasigma(self, redges, *args, **kwargs):
        assert np.iterable(redges)
        res = np.array([self.single_rbin_ds(redges[i], redges[i + 1])
                        for i, val in enumerate(redges[:-1])])

        # this is assuming a constant source surface density
        self.rr = np.array([(redges[i + 1] ** 3. - redges[i] ** 3.) /
                            (redges[i + 1] ** 2. - redges[i] ** 2.)
                            for i, val in enumerate(redges[:-1])])  * 2. / 3.
        self.redges = redges
        self.ds = res


class SimpleNFWProfile(DeltaSigmaProfile):
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z'])

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return nfw_deltasigma(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return ds_nfw_ring(r0, r1, **self.pardict)


class DirectRingCeck(SimpleNFWProfile):
    def point_ds(self, r, *args, **kwargs):
        return direct_circ_nfw(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return direct_ring_nfw(r0, r1, **self.pardict)


class TruncatedNFWProfile(DeltaSigmaProfile):
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z', 'rt'])

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.profpars += (kwargs["rt"],)
        self.parnames += ("rt",)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        x = r / self.profpars[0]  # r / rs
        return ds_tnfw(x, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return ds_tnfw_ring(r0, r1, **self.pardict)


class OffsetNFWProfile(DeltaSigmaProfile):
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z', 'dist'])

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.profpars += (kwargs["dist"],)
        self.parnames += ("dist",)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return ds_oc_nfw(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return ds_oc_nfw_ring(r0, r1, **self.pardict)


class DirectOffsetCheck(OffsetNFWProfile):
    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return indirect_ring_oc_nfw(r0, r1, **self.pardict)








