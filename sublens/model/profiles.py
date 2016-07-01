"""
Interface for DeltaSigma profiles

The idea is that here I just import the actual implementations, so that there
is a layer between what happens numerically and how it is interfaced with.
"""

import numpy as np

from sublens.io import default_cosmo
from ..model.astroconvert import nfw_params

# from ..model.cycalc import tnfw
# from ..model.cycalc import tnfw_ring

from ..model.pycalc.full_nfw import nfw_deltasigma
from ..model.pycalc.full_nfw import nfw_ring

from ..model.pycalc.full_nfw import oc_nfw
from ..model.pycalc.full_nfw import oc_nfw_ring

from ..model.pycalc.halo2 import H2calc


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
            raise ValueError('mode must be "rr" or "edges"')


class SimpleNFWProfile(DeltaSigmaProfile):
    """The conventional spherical NFW profile"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z'])
        self._provides = ['clike', 'mlike']
        self._requires = ['c200c', 'm200c']

    def __str__(self):
        return "SimpleNFWProfile"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        prov = dict(zip(self._provides, [kwargs.pop(key)
                                         for key in self._requires]))
        prov.update(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, **prov)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return nfw_deltasigma(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return nfw_ring(r0, r1, **self.pardict)


class SimpleNFWProfile500(SimpleNFWProfile):
    """The conventional spherical NFW profile"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c500c', 'm500c', 'z'])
        self._provides = ['clike', 'mlike']
        self._requires = ['c500c', 'm500c']

    def __str__(self):
        return "SimpleNFWProfile500"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        prov = dict(zip(self._provides, [kwargs.pop(key)
                                         for key in self._requires]))
        prov.update(kwargs)
        self.profpars, self.parnames = nfw_params(self.cosmo, delta=500,
                                                  **prov)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True


# class TruncatedNFWProfile(DeltaSigmaProfile):
#     """The NFW profile with a hard cutoff at rt"""
#     def __init__(self, cosmo=None):
#         super().__init__(cosmo=cosmo)
#         self.requires = sorted(['c200c', 'm200c', 'z', 'rt'])
#         self._provides = ['clike', 'mlike']
#         self._requires = ['c200c', 'm200c']
#
#     def __str__(self):
#         return "TruncatedNFWProfile"
#
#     def prepare(self, **kwargs):
#         assert set(self.requires) <= set(kwargs)
#         prov = dict(zip(self._provides, [kwargs.pop(key)
#                                          for key in self._requires]))
#         prov.update(kwargs)
#         self.profpars, self.parnames = nfw_params(self.cosmo, delta=200,
#                                                   **prov)
#         self.profpars += (kwargs["rt"],)
#         self.parnames += ("rt",)
#         self.pardict = dict(zip(self.parnames, self.profpars))
#         self._prepared = True
#
#     def point_ds(self, r, *args, **kwargs):
#         return tnfw(r, **self.pardict)
#
#     def single_rbin_ds(self, r0, r1, *args, **kwargs):
#         return tnfw_ring(r0, r1, **self.pardict)


class OffsetNFWProfile(DeltaSigmaProfile):
    """The NFW profile offset by dist"""
    def __init__(self, cosmo=None):
        super().__init__(cosmo=cosmo)
        self.requires = sorted(['c200c', 'm200c', 'z', 'dist'])
        self._provides = ['clike', 'mlike']
        self._requires = ['c200c', 'm200c']

    def __str__(self):
        return "OffsetNFWProfile"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        prov = dict(zip(self._provides, [kwargs.pop(key)
                                         for key in self._requires]))
        prov.update(kwargs)
        # print(prov)
        self.profpars, self.parnames = nfw_params(self.cosmo, delta=200,
                                                  **prov)
        # print(self.profpars)
        self.profpars += (kwargs["dist"],)
        self.parnames += ("dist",)
        self.pardict = dict(zip(self.parnames, self.profpars))
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return oc_nfw(r, **self.pardict)

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return oc_nfw_ring(r0, r1, **self.pardict)

# -----------------------------------------------------------------------------


class CorrelatedMatterProfile(DeltaSigmaProfile):
    """Simple 2-halo term for lensing analysis"""
    def __init__(self, z, powspec, cosmo, scalar_ind=0.96):
        super().__init__()
        self.requires = sorted(['b'])
        self.powspec = z
        self.powspec = powspec
        self.cosmo = cosmo

        self.w2calc = H2calc(z, powspec, cosmo, scalar_ind)

    def __str__(self):
        return "Simple2HaloTerm"

    def prepare(self, **kwargs):
        assert set(self.requires) <= set(kwargs)
        self.pardict = {"b": kwargs["b"]}
        self._prepared = True

    def point_ds(self, r, *args, **kwargs):
        return self.w2calc.wval(r) * self.pardict["b"]

    def single_rbin_ds(self, r0, r1, *args, **kwargs):
        return self.w2calc.wring(r0, r1) * self.pardict["b"]

