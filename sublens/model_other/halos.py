"""
Mass distributions
"""

from astropy import cosmology
from ..model_other import default_cosmo


class MassDistribution(object):
    def __init__(self, *args, **kwargs):
        pass


class MassComponent(object):
    def __init__(self, cosmo=None, *args, **kwargs):
        if cosmo is None:
            cosmo = default_cosmo()
        err_msg = 'currently only FlatLambdaCDM cosmology is supported'
        assert isinstance(cosmo, cosmology.FlatLambdaCDM), err_msg
        self.cosmo = cosmo


class NFW(MassComponent):
    def __init__(self, cosmo=None, *args, **kwargs):
        super().__init__(cosmo=cosmo)


class TruncatedNFW(MassComponent):
    def __init__(self, cosmo=None, *args, **kwargs):
        super().__init__(cosmo=cosmo)


class TwoHalo(MassComponent):
    def __init__(self, cosmo=None, *args, **kwargs):
        super().__init__(cosmo=cosmo)


