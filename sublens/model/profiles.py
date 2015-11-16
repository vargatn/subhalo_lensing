"""
Density and \Delta\Sigma profile functions


The core functionality is focused on different versions of NFW profiles.
Also includes conversion for (m_200, c_200, z) ->  (r_s, rho_s)

Contents:
----------
*   3D NFW density in h^2 M_\odot / Mpc^3
*   2D NFW density in h M_\odot / Mpc^2
*   \Delta\Sigma of NFW in h M_\odot / Mpc^2

*   conversion tool for NFW parameters
"""

import math


def converter(m200, c200, z):


    cdens = cosmo.crit_dens(z) * (nc.Mpc ** 3. / nc.M_sun)
    r200 = (3. / 4. * m200 / (200. * cdens) / math.pi) ** (1. / 3.)
    rs = r200 / c200
    dc = 200. / 3. * (c200 ** 3.) / (math.log(1. + c200 - c200 / (1. + c200)))
    rho_s = dc * cdens

    return rs, rho_s, r200


