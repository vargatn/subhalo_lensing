__author__ = 'vtn'

from astropy.cosmology import FlatLambdaCDM

# class ModelSuite:
#
#     def __init__(self, cosmology="default"):
#
#         cosmo0 = {
#             'H0': 100,
#             'Om0': 0.3
#         }
#
#         if isinstance(cosmology, dict):
#             self.cosmo = FlatLambdaCDM(**cosmology)
#         elif isinstance(cosmology, str) & cosmology is "default":
#             self.cosmo = FlatLambdaCDM(**cosmo0)
#         else:
#             raise TypeError("invalid cosmology defined," +
#                             " for default use 'default")
#
