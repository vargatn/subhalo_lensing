"""
I/O package


Contents:
----------

* iocosmo:
    - get default cosmology for the analysis package
    - extract cosmologyu parameters to dictionary

* ioshear:
    - file conversion
    - read xshear output from various file formats and data styles:
        point, sample, reduced
    - wrap xshear trough python (rudimentary)

* visual:
    - Kernel density smoothed contour levels fro 1D and 2D
    - 95 and 68 percentile levels for distributions in 1D and 2D
    - overlayable triangle (corner) plot


Class for easy interface with xshear and the output file
"""

from .ioshear import *
from .iocosmo import *
from .visual import *

