"""
Conversions and scaling relations related to astrophysisc/cosmology
"""

import math
import astropy.units as units


def nfw_params(cosmo, z, mlike, clike, delta=200., dens="crit", mlog10=True,
               *args, **kwargs):
    """
    Calculates the parameters of an NFW halo based on the given cosmology

    :param cosmo: FlatLambdaCDM cosmology object to be used

    :param z: redshift of object

    :param mlike: mass like parameter

    :param clike: concentraton like parameter

    :param delta: overdensity value e.g.: 200, 500 etc...

    :param dens: density type: crit or mean

    :param mlog10: if mass is given in log10 base

    :return: rs, rho_s), in units of Mpc and MSol
    """
    if mlog10:
        mlike = 10. ** mlike
    mlike *= units.solMass

    if dens == "crit":
        denslike = cosmo.critical_density(z).to(units.solMass / units.Mpc ** 3)
    elif dens == "mean":
        denslike = cosmo.critical_density(z).to(units.solMass /
                                                units.Mpc ** 3) * cosmo.Om(z)

    dc = delta / 3. * (clike ** 3.) / \
         (math.log(1. + clike) - clike / (1. + clike))
    rho_s = dc * denslike

    rlike = (3. / 4. * mlike / (delta * denslike) / math.pi) ** (1. / 3.)
    rs = rlike / clike
    rs = rs.to(units.Mpc)

    return (rs.value, rho_s.value), ('rs', 'rho_s')
