"""
I/O and profile maker for measured \Delta\Sigma profiles


Class for easy interface with xshear output and config files, also for
easy selection of appropriate lenses.


Functionality:
---------------

* WrapX: creates config and description file for shear calc.
                        (containes name and descriptin of the original data)

* ShearData output reader
"""

from .ioshear import *