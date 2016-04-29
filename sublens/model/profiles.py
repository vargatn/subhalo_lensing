"""
Interface for DeltaSigma profiles

The idea is that here I just import the actual implementations, so that there
is a layer between what happens numerically and how it is interfaced with.
"""

import numpy as np

from sublens.io import default_cosmo
from ..model.astroconvert import nfw_params

from ..model.cycalc import tnfw
from ..model.cycalc import tnfw_ring

from ..model.pycalc import nfw_deltasigma
from ..model.pycalc.full_nfw import nfw_ring

from ..model.pycalc.full_nfw import oc_nfw
from ..model.pycalc.full_nfw import oc_nfw_ring


class DeltaSigmaProfile(object):
    """Base class for lensing shear profiles"""
    def __init__(self, cosmo=None):
        """
        This abstract class should be used to subclass particular scenarios
        :param cosmo: cosmology object
        """
        if cosmo is None:
            cosmo = default_cosmo()
        self.cosmo = cosmo
        self.h = self.cosmo.H0.value / 100.

        self.requires = []
        self.profpars = []
        self.parnames = []
        self.pardict = {}

        self.rr = None
        self.redges = None
        self.ds = None

        self._prepared = False

    def __str__(self):
        return "DeltaSigmaProfile"

    def reset(self):
        self.rr = None
        self.redges = None
        self.ds = None

        self.profpars = []
        self.parnames = []
        self.pardict = {}
        self._prepared = False

    def prepare(self, **kwargs):
        """Prepares the profile calculations"""
        raise NotImplementedError

    def point_ds(self, r, *args, **kwargs):
        """Evaluates the angle averaged DeltaSigma at a single radial value"""
        raise NotImplementedError

    def deltasigma(self, rr, *args, **kwargs):
        """Creates a DeltaSigma profile by calling self.point_ds"""
        assert self._prepared, "The profile parameters are not prepared!!!"

        if np.iterable(rr):
            ds = np.array([self.point_ds(r) for r in rr])
        else:
            ds = np.array([self.point_ds(rr)])
            rr = np.array([rr])

        self.rr = rr
        self.ds = ds / 1e12

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        """Evaluates the ring averaged DeltaSigma for a single ring"""
        raise NotImplementedError

    def rbin_deltasigma(self, redges, *args, **kwargs):
        """Evaluates the ring averaged DeltaSigma based on a set of edges"""
        assert np.iterable(redges)
        res = np.array([self.single_rbin_ds(redges[i], redges[i + 1])
                        for i, val in enumerate(redges[:-1])])

        # this is assuming a constant source surface density
        self.rr = np.array([(redges[i + 1] ** 3. - redges[i] ** 3.) /
                            (redges[i + 1] ** 2. - redges[i] ** 2.)
                            for i, val in enumerate(redges[:-1])]) * 2. / 3.
        self.redges = redges
        self.ds = res / 1e12

    def calc(self, rvals, mode, *args, **kwargs):
        """tunable calculation"""
        if mode == "rr":
            self.deltasigma(rvals, *args, **kwargs)
        elif mode == "edges":
            self.rbin_deltasigma(rvals, *args, **kwargs)
        else:
            raise ValueError('mode must be "rr" or "edges"' )


class SimpleNFWProfile(DeltaSigmaProfile):
    """The conventional spherical NFW profile"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z'])

    def __str__(self):
        return "SimpleNFWProfile"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return nfw_deltasigma(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return nfw_ring(r0, r1, **self.pardict)


class TruncatedNFWProfile(DeltaSigmaProfile):
    """The NFW profile with a hard cutoff at rt"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z', 'rt'])

    def __str__(self):
        return "TruncatedNFWProfile"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.profpars += (kwargs["rt"],)
        self.parnames += ("rt",)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return tnfw(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return tnfw_ring(r0, r1, **self.pardict)


class OffsetNFWProfile(DeltaSigmaProfile):
    """The NFW profile offset by dist"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z', 'dist'])

    def __str__(self):
        return "OffsetNFWProfile"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **kwargs)
        self.profpars += (kwargs["dist"],)
        self.parnames += ("dist",)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return oc_nfw(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return oc_nfw_ring(r0, r1, **self.pardict)


# FIXME this is a priority 1 task
class SubhaloProfile(DeltaSigmaProfile):
    def __init__(self):
        super().__init__()


# FIXME this is a priority 1 task
class ParenthaloPofile(DeltaSigmaProfile):
    def __init__(self):
        super().__init__()


# -----------------------------------------------------------------------------


# TODO Add 2halo term with single bias parameter
class CorrelatedMatterProfile(DeltaSigmaProfile):
    def __init__(self):
        super().__init__()


# TODO This is similar to the above, but with off-centering added
class OffsetCorrelatedMatterProfile(DeltaSigmaProfile):
    def __init__(self):
        super().__init__()


# TODO this is for the cluster BCG...
class SingularProfile(DeltaSigmaProfile):
    def __init__(self):
        super().__init__()


