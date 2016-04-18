"""
Cython-based calculations
"""

import pyximport; pyximport.install()
from ..cycalc.trunc_nfw import tnfw as ds_tnfw
from ..cycalc.trunc_nfw import tnfw_ring as ds_tnfw_ring

