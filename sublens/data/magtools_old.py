"""
Absolute magnitudes and K-correction
"""

import sys
import time

import numpy as np
import scipy.integrate as integr
import scipy.interpolate as interp

# import tools.load as load
# import tools.selectors as select
# import settings.paths as paths
# import tools.cosmology as cosmo
# import settings.constants as nc


def apply_kcorr(robs, zarr):
    """
    Applies k-correction to apparent magnitudes for z0=0.0 frame

    :param robs:  observed des r-band observed magnitudes
    :param zarr: redshifts for the same entries
    :return: rabs absolute magnitudes
    """

    assert robs.shape == zarr.shape

    try:
        rcorr_ref_table = load.kcorr_table()
    except FileNotFoundError as e:
        print(e)
        print('Kcorr lookup table not found!')
        sys.exit(1)

    rcorr = select.interpolate_table(zarr, rcorr_ref_table)
    rmag_abs = robs - rcorr
    return rmag_abs


def make_kcorr_table(zrange=(0.1, 0.9), points=100, save_name=None,
                     h=nc.h, om_m=nc.Om_M):
    """
    Creates lookup table for k-correction and distance modulus

    Assumes DES obsevation bands, and flat LCDM cosmology with Om_M = 0.3 and
    h = 0.7

    table format:
    -------------
    0   1                          2                    3
    z,  distance modulus + kcorr,  distance modulus,    kcorr

    :param zrange: redshift range
    :param points: number of redshift values to use
    :param save_name: filename to use for saving
    :param h: Hubble parameter
    :param om_m: \Omega_{Matter}
    :return:
    """
    if save_name is None:
        save_name = paths.ktable

    zarr = np.linspace(zrange[0], zrange[1], num=points, endpoint=True)
    darr = np.array([dist_modulus(z, h=h, om_m=om_m) for z in zarr])
    karr = np.array([des_rband_kcorr(z) for z in zarr])

    corr_arr = darr + karr

    corr = np.vstack((zarr, corr_arr, darr, karr)).T
    np.save(save_name, corr)
    return corr


def dist_modulus(z, h=1.0, om_m=nc.Om_M):
    """
    Calculates distance modulus in a flat LCDM cosmology

    :param z: redshift
    :param h: Hubble parameter
    :param om_m: \Omega_{Matter}
    :return:
    """
    dl = cosmo.d_l(z, om_r=0., om_m=om_m, pc=True)  # this is in pc / h
    dl *= h
    dm = 5. * np.log10(dl / 10.)
    return dm


def des_rband_kcorr(z, templ, resp, low=540., high=730, tme=False):
    """
    Calculate K-correction for the DES R-band

    Uses Red-template from Blanton et al. 2007

    NOTE:
    ---------
    Due to numerical errors the wavelengths range should be reasonably close to
    the actual bandpass limits.

    :param z: redshift
    :param templ: red template
    :param resp: response curve
    :param low: lower bound of the wavelength range
    :param high: upper bound of the wavelength range
    :param tme: prints integration time to stdout if True
    :return: value of K-correction
    """

    ftempl = interp.interp1d(templ[:, 0], templ[:, 1])
    fresp = interp.interp1d(resp[:, 0], resp[:, 1])

    time0 = time.time()

    oint, oerr = des_r_obsint(z, ftempl, fresp, low=low, high=high)
    eint, eerr = des_r_emint(ftempl, fresp, low=low, high=high)

    kval = -2.5 * np.log10(oint / eint / (1. + z))

    time1 = time.time()
    if tme:
        print(time1 - time0)
    return kval


def des_r_emint(ftemplate, fresponse, low=540., high=730):
    """Integral over emitted spectra for  r band DES"""
    y, abserr = integr.quad(eexpr, low, high,  args=(ftemplate, fresponse))
    return y, abserr


def des_r_obsint(z, ftemplate, fresponse, low=540., high=730):
    """Integral over observed spectra for  r band DES"""
    y, abserr = integr.quad(oexpr, low, high,
                            args=(ftemplate, fresponse, z))
    return y, abserr


def eexpr(ll, ftemplate, fresponse):
    """Emitted frame"""

    return ll * ftemplate(ll) * fresponse(ll)

def oexpr(ll, ftemplate, fresponse, z):
    """Observed frame"""
    return ll * ftemplate(ll / (1. + z)) * fresponse(ll)