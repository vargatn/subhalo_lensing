import time
import math
import scipy
import pickle
import numpy as np
import astropy.units as u
import multiprocessing as mp
import scipy.stats as stats
import scipy.interpolate as interp
import scipy.integrate as integr

from sublens import default_cosmo


def clustint_maker(medge, mtable, **kwargs):
        ppint = scipy.interpolate.interp1d(medge, mtable, axis=0,
                                           fill_value=np.inf, bounds_error=False)
        def clustinterp(mclust, **kwargs):
            return ppint(mclust)

        return clustinterp

def contprep(data, sample=1e4, **kwargs):
    if sample is not None:
        subsample = data[np.random.choice(np.arange(len(data)), int(sample)), :]
    else:
        subsample = data

    allgrid = kde_smoother_2d(subsample, **kwargs)
    ta68, pa68 = conf2d(0.68, allgrid[0], allgrid[1], allgrid[2])
    ta95, pa95 = conf2d(0.95, allgrid[0], allgrid[1], allgrid[2])

    return allgrid, ta68, ta95

def contourprep(lname, sample=1e4, **kwargs):
    """Only for file loading"""
    simlog = pickle.load(open(lname, 'rb'))
    tracks = simlog['tracks']

    alltrack = np.vstack(tracks)
    if sample is not None:
        subsample = alltrack[np.random.choice(np.arange(len(alltrack)), int(sample)), :]
    else:
        subsample = alltrack

    allgrid = kde_smoother_2d(subsample, **kwargs)
    ta68, pa68 = conf2d(0.68, allgrid[0], allgrid[1], allgrid[2])
    ta95, pa95 = conf2d(0.95, allgrid[0], allgrid[1], allgrid[2])

    return allgrid, ta68, ta95, simlog


def kde_smoother_1d(pararr, xlim=None, num=100, pad=0):
    """
    Creates a smoothed histogram from 1D scattered data

    :param pararr: list of parameters shape (Npoint, Npar)
    :param xlim: x range of the grid
    :param num: number of gridpoints on each axis
    :return: xgrid, values for each point
    """
    # creating smoothing function
    kernel = stats.gaussian_kde(pararr)

    # getting boundaries
    if xlim is None:
        xlim = [np.min(pararr), np.max(pararr)]
        xpad = pad * np.diff(xlim)
        xlim[0] -= xpad
        xlim[1] += xpad

    # building grid
    xgrid = np.linspace(xlim[0], xlim[1], num)

    # evaluating kernel on grid
    kvals = kernel(xgrid)

    return xgrid, kvals


def kde_smoother_2d(pararr, xlim=None, ylim=None, num=100, pad=0.1):
    """
    Creates a smoothed histogram from 2D scattered data

    :param pararr: list of parameters shape (Npoint, Npar)
    :param xlim: x range of the grid
    :param ylim: y range of the grid
    :param num: number of gridpoints on each axis
    :return: xgrid, ygrid, values for each point
    """
    # creating smoothing function
    kernel = stats.gaussian_kde(pararr.T)

    # getting boundaries
    if xlim is None:
        xlim = [np.min(pararr[:, 0]), np.max(pararr[:, 0])]
        xpad = pad * np.diff(xlim)
        xlim[0] -= xpad
        xlim[1] += xpad
    if ylim is None:
        ylim = [np.min(pararr[:, 1]), np.max(pararr[:, 1])]
        ypad = pad * np.diff(ylim)
        ylim[0] -= ypad
        ylim[1] += ypad

    # building grid
    xgrid = np.linspace(xlim[0], xlim[1], num)
    ygrid = np.linspace(ylim[0], ylim[1], num)
    xx, yy = np.meshgrid(xgrid, ygrid )
    grid_coords = np.append(xx.reshape(-1,1), yy.reshape(-1,1),axis=1)

    # evaluating kernel on grid
    kvals = kernel(grid_coords.T).reshape(xx.shape)

    return xx, yy, kvals


def snoisemaker(dvec, covm, seed=13141, **kwargs):
    """Creates a noisy data based on the passed covmatrix"""
    if seed is not None:
        np.random.seed(seed)

    err = np.sqrt(np.diag(covm))
    offset = np.random.multivariate_normal(np.zeros(shape=err.shape), covm)
    return dvec + offset, err


def conf1d(pval, grid, vals, res=200, etol=1e-3, **kwargs):
    """
    Calculates cutoff values for a given percentile for 1D distribution

    Requires evenly spaced grid!

    :param pval: percentile
    :param grid: parameter
    :param vals: value of the p.d.f at given gridpoint
    :param res: resolution of the percentile search
    :return: cutoff value, actual percentile
    """

    area = np.mean(np.diff(grid))
    assert (np.sum(vals*area) - 1.) < etol, 'Incorrect normalization!!!'

    mx = np.max(vals)

    tryvals = np.linspace(mx, 0.0, res)
    pvals = np.array([np.sum(vals[np.where(vals > level)] * area)
                      for level in tryvals])

    tind = np.argmin((pvals - pval)**2.)
    tcut = tryvals[tind]
    return tcut, pvals[tind]


def conf2d(pval, xxg, yyg, vals, res=200, etol=1e-3, **kwargs):
    """
    Calculates cutoff values for a given percentile for 2D distribution

    :param pval: percentile
    :param xxg: grid for the first parameter
    :param yyg: grid for the second parameter
    :param vals: value of the p.d.f at given gridpoint
    :param res: resolution of the percentile search
    :return: cutoff value, actual percentile
    """
    edge1 = xxg[0, :]
    edge2 = yyg[:, 0]

    area = np.mean(np.diff(edge1)) * np.mean(np.diff(edge2))
    assert (np.sum(vals*area) - 1.) < etol, 'Incorrect normalization!!!'

    mx = np.max(vals)
    tryvals = np.linspace(mx, 0.0, res)
    pvals = np.array([np.sum(vals[np.where(vals > level)] * area)
                      for level in tryvals])

    tind = np.argmin((pvals - pval)**2.)
    tcut = tryvals[tind]
    return tcut, pvals[tind]


def llike(hh, rvals, dvec, dcov, **kwargs):
    """
    Evaluates log-likelihood

    Uses centered non-bin-averaged DeltaSigma profile

    :param hh: Halo object
    :param rvals: radius values to use
    :param dvec: data vector
    :param dcov: data covariance matrix
    :param kwargs: arguements passed to
    :return: chi2 value
    """
    cinv = np.linalg.inv(dcov)

    model = hh.cen_ds_curve(rvals, **kwargs)

    diff = (dvec - model)
    chisq = float(np.dot(diff.T, np.dot(cinv, diff)))

    return chisq


def llike0(dvec, dcov, **kwargs):
    """
    Evaluates log-likelihood consistency with 0
    :param dvec: data vector
    :param dcov: data covariance matrix
    :return: chi2 value
    """
    cinv = np.linalg.inv(dcov)
    model = np.zeros(shape=dvec.shape)

    diff = (dvec - model)
    chisq = float(np.dot(diff.T, np.dot(cinv, diff)))

    return chisq


def llike_joint(hh, rvals, dvec, dcov, ppint, **kwargs):
    """
    Joint fit evaluation log-likelihood

    Uses centered non-bin-averaged DeltaSigma profile and parent cluster

    :param hh: Halo object
    :param rvals: radius values to use
    :param dvec: data vector
    :param dcov: data covariance matrix
    :param kwargs: arguements passed to
    :return: chi2 value
    """
    cinv = np.linalg.inv(dcov)

    model0 = hh.cen_ds_curve(rvals, **kwargs)
    model1 = ppint(**kwargs)

    model = model0 + model1


    diff = (dvec - model)
    chisq = float(np.dot(diff.T, np.dot(cinv, diff)))

    if model1[0] == np.inf:
        print('problem here')
        print(chisq)

    return chisq, model


def mymc(like, settings, nstep=10, seed=None, verbose='True', rng=None, **kwargs):
    """
    MCMC optimizer

    All parameters should be 'linear' eg m=12 instead of 1e12

    :param like: llike function
    :param settings: arguements for the llike plus
                     * dict of par0
                     * dict of fitted parameter names
                     * dict of step sizes for fitted parameters
                     * settings for llike
    :param nstep: Number of MCMC steps to make
    :param seed: random seed for the rng
    :return: parameter array, chi2 array
    """

    if rng is None:
        rng = np.random.RandomState(seed=seed)

    # getting parameters
    parnames = np.asanyarray(sorted(settings['par0'].keys()))
    parval0 = np.array([settings['par0'][name] for name in parnames])
    fitted_ind = np.asanyarray([settings['fitted'][name] for name in parnames])
    fitpars = parnames[fitted_ind]
    fixed_ind = np.invert(fitted_ind.astype(bool))
    fixpars = parnames[fixed_ind]
    fixvals = parval0[fixed_ind]
    fparvals = parval0[fitted_ind]


    # creating container
    parvec0 = np.array(fparvals)
    parvec = np.array([parvec0])

    tmpdict = {}
    tmpdict.update(settings)
    tmpdict.update(settings['par0'])

    # calculating chi2
    chi, dsvec0 = like(**(tmpdict))
    chivec =  np.array([chi])

    dsvec = np.array([dsvec0])

    # step covariance matrix
    stepmatr = np.diag([settings['step0'][name] for name in fitpars])

    # going through the steps
    for i in np.arange(nstep):
        if i%1000 == 0:
            print(i)
        ds = dsvec[-1]
        par = parvec[-1]

        # making random step
        dpar = rng.multivariate_normal(mean=np.zeros(par.shape), cov=stepmatr)
        parnew = par + dpar


        # checking for parameter ranges
        for i, name in enumerate(fitpars):
            if (parnew[i] < settings['ranges'][name][0]) | (parnew[i] > settings['ranges'][name][1]):
                # print('----------------')
                # print(settings['ranges'][name])
                # print(parnew[i], par[i])
                parnew = par

        tmpdict = {}
        tmpdict.update(settings)
        [tmpdict.update({key: val}) for (key, val) in zip(fitpars, parnew)]
        [tmpdict.update({key: val}) for (key, val) in zip(fixpars, fixvals)]

        # calculating chi2 value for new position
        chinew, dsnew = like(**tmpdict)
        # obtaining reference random number
        ref = rng.uniform()

        # checkong for accepting new step
        if np.exp(-1. * (chinew - chi)) > ref:
            par = parnew
            chi = chinew
            ds = dsnew

        # saving data
        parvec = np.vstack((parvec, par))
        chivec = np.vstack((chivec, chi))
        dsvec = np.vstack((dsvec, ds))

    return parvec, chivec, dsvec


# def rec_chain(m, c=None, z=0.5, dist=0.0, nreal=20, seed=5, nstep=2000, save_name=None):
#     if save_name is None:
#         save_name = 'run_m{:.2f}.p'.format(m)
#
#     mids, edges, areas = genhalo.get_edges(mx=2., bins=15)
#     h1 = genhalo.NFW()
#     hh = genhalo.Halo((h1,))
