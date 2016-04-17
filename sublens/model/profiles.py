"""
Interface for DeltaSigma profiles
"""

# import pyximport; pyximport.install()
import numpy as np

from ..model import default_cosmo
from ..model.astroconvert import nfw_params
from ..model.basefunc import nfw_deltasigma

from ..model.cycalc import ds_tnfw


class DeltaSigmaProfile(object):
    def __init__(self, cosmo=None):
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.

        self.rr = None
        self.ds = None

        self._prepared = False

    def prepare(self, **kwargs):
        raise NotImplementedError

    def point_deltasigma(self, r, *args, **kwargs):
        raise NotImplementedError

    def deltasigma(self, rr, *args, **kwargs):
        assert self._prepared

        if np.iterable(rr):
            ds = np.array([self.point_deltasigma(r) for r in rr])
        else:
            ds = np.array([self.point_deltasigma(rr)])
            rr = np.array([rr])

        self.rr = rr
        self.ds = ds


class SimpleNFWProfile(DeltaSigmaProfile):
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)

        self.requires = sorted(['c200c', 'm200c', 'z'])
        self.profpars = []
        self.parnames = []

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self._prepared = True

    def point_deltasigma(self, r, *args, **kwargs):
        return nfw_deltasigma(r, *self.profpars)


class truncatedNFWProfile(DeltaSigmaProfile):
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)

        self.requires = sorted(['c200c', 'm200c', 'z', 'rt'])
        self.profpars = []
        self.parnames = []

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.profpars += (kwargs["rt"],)
        self.parnames += ("rt",)
        self._prepared = True

    def point_deltasigma(self, r, *args, **kwargs):
        x = r / self.profpars[0]  # r / rs
        return ds_tnfw(x, *self.profpars)[0]

#


