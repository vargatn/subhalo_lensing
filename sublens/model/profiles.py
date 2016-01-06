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
import time
import math
import numpy as np
from ..model.misc import nfw_pars
import jdnfw.nfw as jd_nfw

from scipy import integrate as integr
# -----------------------------------------------------------------------------
# POOLABLE DECORATORS

def poolable_example_profile(params):
    return example_profile(**params)

def poolable_nfw_prof(params):
    return nfw_prof(**params)

# -----------------------------------------------------------------------------
# DELTA SIGMA PROFILE


# def nfw_prof(m200, c200, z, edges, **kwargs):
#     rs, rho_s, r200 = nfw_pars(m200, c200, z)

def nfw_prof_noint(c200, m200, z, edges):
    rs, rho_s, r200 = nfw_pars(m200, c200, z)
    # print(rs, rho_s)
    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, edge in enumerate(edges[:-1])])
    cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                     (edges[i + 1] ** 2. - edges[i] ** 2.)
                     for i, edge in enumerate(edges[:-1])])

    # print(cens)
    # print(rs)
    ds = np.array([nfw_shear_t(cen, rs, rho_s) / cen
                   for i, cen in enumerate(cens)])

    ds = ds / 1e12
    # ds = ds / 1e12 * np.pi * 2.

    return cens, ds


def nfw_prof(c200, m200, z, edges, epsabs=1.49e-4, epsrel=1.49e-8,
             verbose=False):

    rs, rho_s, r200 = nfw_pars(m200, c200, z)
    # print(rs, rho_s, r200)
    fargs = (rs, rho_s)

    areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
                      for i, edge in enumerate(edges[:-1])])
    cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
                     (edges[i + 1] ** 2. - edges[i] ** 2.)
                     for i, edge in enumerate(edges[:-1])])

    t0 = time.time()
    sigma_rim = np.array([integr.quad(nfw_shear_t, edges[i], edges[i + 1],
                                     args=fargs)
                          for i, val in enumerate(edges[:-1])])
    sigma_rim = 2. * math.pi * sigma_rim / 1e12

    t1 = time.time()
    if verbose:
        print(t1 - t0, "s")

    ds = sigma_rim / areas[:, np.newaxis]

    return cens, ds[:, 0]


def nfw_shear_t(r, rs, rho_s):
    """
    RAW centered NFW halo

    Equation from:
    Gravitational Lensing by NFW Halos
    Wright, Candace Oaxaca; Brainerd, Tereasa G.
    http://adsabs.harvard.edu/abs/2000ApJ...534...34W

    :return: value of tangential shear_old at distance x
    """

    x = r / rs

    shear = 0

    if 0. < x < 1.:
        shear = rs * rho_s * ((8. * math.atanh(math.sqrt((1. - x) / (1. + x))) / (
            x ** 2. * math.sqrt(1. - x ** 2.)) +
                        4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1.) +
                        (4. * math.atanh(math.sqrt((1. - x) / (1. + x)))) / (
                            (x ** 2. - 1.) * math.sqrt(1. - x ** 2.))))

    elif x == 1.:
        shear = rs * rho_s *(10. / 3. + 4. * math.log(1. / 2.))

    elif x > 1.:
        shear = rs * rho_s *((8. * math.atan(math.sqrt((x - 1.) / (1. + x))) / (
            x ** 2. * math.sqrt(x ** 2. - 1.)) +
                        4. / x ** 2. * math.log(x / 2.) - 2. / (x ** 2. - 1) +
                        (4. * math.atan(math.sqrt((x - 1.) / (1. + x)))) / (
                            (x ** 2. - 1.) ** (3. / 2.))))

    return shear * r


def nfw_ds_direct(r, m200, c200, z):
    rs, rho_s, r200 = nfw_pars(m200, c200, z)

    x = r / rs

    sigma = 0.0

    if 0.0 < x < 1.0:
        sigma = 2. * rs * rho_s / (x*x - 1.) *\
                (1. - 2. / math.sqrt(1. - x*x) *
                 math.atanh(math.sqrt((1. - x) / (1. + x))))
    if x == 1.:
        sigma = 2. * rs * rho_s / 3.
    if x > 1.:
        sigma = 2. * rs * rho_s / (x*x - 1.) *\
                (1. - 2. / math.sqrt(x*x - 1.) *
                 math.atan(math.sqrt((x - 1.) / (1. + x))))

    smean = 0.0

    if 0.0 < x < 1.0:
        smean = 4. / (x*x) * rs * rho_s *\
                (2. / math.sqrt(1. - x*x) *
                 math.atanh(math.sqrt((1. - x) / (1. + x))) + math.log(x / 2.))
    if x == 1.0:
        smean = 4. * rs * rho_s * (1. + math.log(1. / 2.))
    if x > 1.0:
        smean = 4. / (x*x) * rs * rho_s *\
                (2. / math.sqrt(x*x - 1.) *
                 math.atan(math.sqrt((x - 1.) / (1. + x))) + math.log(x / 2.))

    return smean - sigma



# -----------------------------------------------------------------------------
# EXAMPLE PROFILES

def disc_dbl(phi, x, *args, **kwargs):
    return x

def example_profile(m200, c200, z, lims=(0.02, 2.0), num=30):
    """
    Creates an example profile based on a 3D NFW density profile
    :param m200: Virial mass
    :param c200: concentration
    :param z: redshift
    :param lims: radiaé range of the profile
    :param num: number of points in the profile
    :return: radial density profile
    """
    # evaluation points
    rarr = np.logspace(np.log10(lims[0]), np.log10(lims[1]), num=num)

    # density at each point
    darr = np.array([nfw_3d(r, m200, c200, z) for r in rarr])

    return rarr, darr


def nfw_3d(r, m200, c200, z):
    """
    3D Navarro-Frenk-White function

    :param m200: Virial mass
    :param c200: Concentration relative to \rho_{crit}
    :param z: redshift
    :return: density
    """
    # obtaining profile parameters
    rs, rho_s, r200 = nfw_pars(m200, c200, z)

    x = r / rs
    dens =  rho_s / (x * (1 + x) ** 2.)
    return dens / 1e12

# -----------------------------------------------------------------------------
# NFW TANGENTIAL SHEAR PROFILES

# def cshear(m200, c200, z, edges, dist=0.0, dpad=0.1, sdeg=2., verbose=True,
#            epsabs=1.49e-4, epsrel=1.49e-8):
#
#     def bmaker(lval):
#         def bound(val):
#             return lval * math.pi
#         return bound
#
#     gfun = bmaker(0.0)
#     hfun = bmaker(1.0)
#
#     rs, rho_s, r200 = nfw_pars(m200, c200, z)
#     fargs = (rs, rho_s, dist)
#
#     areas = np.array([np.pi * (edges[i + 1] ** 2. - edges[i] ** 2.)
#                       for i, edge in enumerate(edges[:-1])])
#
#     cens = np.array([(edges[i + 1] ** 3. - edges[i] ** 3.) * 2. / 3. /
#                      (edges[i + 1] ** 2. - edges[i] ** 2.)
#                      for i, edge in enumerate(edges[:-1])])
#
#     # find bin location of singularity
#     sid = np.argmin((cens - dist) ** 2)
#     t0 = time.time()
#
#     sigma = np.zeros(len(cens))
#     for i, edge in enumerate(edges[:-1]):
#         ival = 0.0
#         if i != sid:
#         # if True:
#             ival, ierr = integr.dblquad(nfw_offc_dblint, edges[i],
#                                         edges[i + 1], gfun, hfun,
#                                         args=fargs,
#                                         epsabs=epsabs, epsrel=epsrel)
#         else:
#
#             rmin = max(edges[i], dist - dpad)
#             rmax = min(edges[i + 1], dist + dpad)
#             print(rmin, rmax)
#
#             # # calculating larger patch
#             ll = sdeg  / 180.
#             # ll = 0.0
#             # print(ll)
#             # ival0, ierr0 = integr.dblquad(nfw_offc_dblint, edges[i],
#             #                               edges[i + 1], bmaker(ll), hfun,
#             #                               args=fargs,
#             #                               epsabs=epsabs, epsrel=epsrel)
#             # #
#             # calculating zoomed in patch
#             ival1, ierr1 = integr.dblquad(nfw_offc_dblint, rmin,
#                                           rmax, bmaker(-ll), bmaker(ll),
#                                           args=fargs,
#                                           epsabs=epsabs, epsrel=epsrel)
#
#             # ival = ival0 + ival1
#             # ierr = ierr0 + ierr1
#
#         sigma[i] = 2. * ival / areas[i]
#
#     t1 = time.time()
#     if verbose:
#         print(t1 - t0, "s")
#
#     return cens, sigma
#
# def nfw_t(m200, c200, z, edges, dist=0.0, verbose=True):
#     rs, rho_s, r200 = nfw_pars(m200, c200, z)
#
#     fargs = (rs, rho_s, dist)
#
#     t0 = time.time()
#     res = ishear(nfw_offc_dblint, fargs, edges)
#     t1 = time.time()
#     if verbose:
#         print(t1 - t0, "s")
#     return res

# -----------------------------------------------------------------------------
# NFW SHEAR T

# def nfw_offc_dblint(phi, r, rs, rho_s, dist=0.0, *args, **kwargs):
#     """tangential profile of Off-Centered NFW halo"""
#
#     dist2 = dist * dist
#     assert r > 0.
#     rr2 = dist2 + r * (r - 2. * dist * math.cos(phi))
#     assert rr2 > 0.
#     rr = math.sqrt(rr2)
#
#     term1 = (dist2 + r * (2. * r * math.cos(phi) ** 2. -
#                           2. * dist * math.cos(phi) - r)) / rr2
#     term2 = (2. * r * math.sin(phi) * (r * math.cos(phi) - dist)) / rr2
#
#     par = rho_s * rs / 1e12
#     x = rr / rs
#
#     shear_t_cent = nfw_shear_t(x=x, par=par) * r
#     shear_t = shear_t_cent * (term1 * math.cos(2. * phi) +
#                               term2 * math.sin(2. * phi))
#
#     return shear_t

