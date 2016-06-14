"""
Python-based calculations

Not as fast as the cython based alternative
"""

from ..pycalc.trunc_nfw import tnfw as ds_tnfw

from ..pycalc.full_nfw import nfw_deltasigma
from ..pycalc.full_nfw import nfw_ring as ds_nfw_ring

from ..pycalc.full_nfw import oc_nfw as ds_oc_nfw
from ..pycalc.full_nfw import oc_nfw_ring as ds_oc_nfw_ring

from ..pycalc.halo2 import W2calc