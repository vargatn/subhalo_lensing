"""
Front-end interface for lensing radial shear profiles
"""

import numpy as np
from astropy import cosmology
from ..model_other import default_cosmo


class RadialLensingProfile(object):
    def __init__(self, cosmo=None, *args, **kwargs):
        """General parent to all lensing related density and shear profiles"""
        if cosmo is None:
            cosmo = default_cosmo()
        err_msg = 'currently only FlatLambdaCDM cosmology is supported'
        assert isinstance(cosmo, cosmology.FlatLambdaCDM), err_msg
        self.cosmo = cosmo

    def give_profile(self, *args, **kwargs):
        """Should return the requested lensing profile"""
        raise NotImplementedError


class ExplicitLensingProfile(RadialLensingProfile):
    def __init__(self, formula, cosmo=None, *args, **kwargs):
        """
        Profile for the case when the result is known in a closed form

        In this case the profile is evaluated and returned

        :param formula: Actual calculation routine to be executed
        :param cosmo: cosmology
        """

        super().__init__(cosmo=cosmo)
        err_msg = 'Calculation function must be properly specified'
        assert hasattr(formula, '__call__'), err_msg
        self._formula = formula

    def give_profile(self, rval, *args, **kwargs):

        if np.iterable(rval):
            res = np.array([self._formula(rv, *args, **kwargs)
                            for rv in rval])
        else:
            res = self._formula(rval, *args, **kwargs)
        return res


class NumericalLensingProfile(RadialLensingProfile):
    def __init__(self, cosmo=None, *args, **kwargs):
        """
        Profile for the case when it must be calculated from a 3D density

        In this case the profile is evaluated and returned

        :param cosmo: cosmology
        """
        super().__init__(cosmo=cosmo)
        self.hasprofile = False

    def give_profile(self, *args, **kwargs):
        assert self.hasprofile
        raise NotImplementedError

    def calc_profile(self, *args, **kwargs):
        """Calculates profile"""
        self.hasprofile = True
        raise NotImplementedError

    
class LookupLensingProfile(RadialLensingProfile):
    def __init__(self, cosmo=None, *args, **kwargs):
        super().__init__(cosmo=cosmo)

    def give_profile(self, *args, **kwargs):
        raise NotImplementedError




