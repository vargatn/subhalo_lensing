"""
Plotting and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def contprep(data, sample=1e4, **kwargs):
    if sample is not None and sample > len(data):
        subsample = data[np.random.choice(np.arange(len(data)), int(sample)), :]
    else:
        subsample = data

    allgrid = kde_smoother_2d(subsample, **kwargs)
    ta68, pa68 = conf2d(0.68, allgrid[0], allgrid[1], allgrid[2])
    ta95, pa95 = conf2d(0.95, allgrid[0], allgrid[1], allgrid[2])

    return allgrid, (ta95, ta68)


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


def corner(pars, par_list, par_edges, figsize=(8, 8), color='black', fig=None,
           axarr=None, mode="hist", cmap="gray_r", normed=True, **kwargs):
    npars = len(pars)
    if fig is None and axarr is None:
        fig, axarr = plt.subplots(nrows=npars, ncols=npars, sharex=False,
                                  sharey=False, figsize=figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # hiding upper triangle
        [[ax.axis('off') for ax in axrow[(i+1):]]
         for i, axrow in enumerate(axarr)]

        # hiding xlabels
        [[ax.set_xticklabels([]) for ax in axrow]
         for i, axrow in enumerate(axarr[:-1, :]) ]
        [[ax.set_yticklabels([]) for ax in axrow[1:]]
         for i, axrow in enumerate(axarr[:, :]) ]

        # Adding the distribution of parameters
        [ax.set_xlabel(par) for (ax, par) in zip(axarr[-1, :], pars)]
        [ax.set_ylabel(par) for (ax, par) in zip(axarr[1:, 0], pars[1:])]

        [ax.tick_params(labelsize=8) for ax in axarr.flatten()]

    [axarr[i, i].hist(par_list[i], bins=par_edges[i], color=color,
                      histtype='step', lw=2.0, normed=normed)
     for i in range(len(par_list))]

    [axarr[i, i].set_xlim((np.min(par_edges[i]), np.max(par_edges[i])))
     for i in range(len(par_list))]

    for i, axrow in enumerate(axarr[1:]):
        for j, ax in enumerate(axrow[:i+1]):
            if mode == "hist":
                [[ax.hist2d(par_list[j], par_list[i+1],
                            bins=(par_edges[j], par_edges[i+1]), cmap=cmap,
                            normed=normed)
                  for j, ax in enumerate(axrow[:i+1])]
                 for i, axrow in enumerate(axarr[1:])]
            elif mode == "contour":

                params = np.vstack((par_list[j], par_list[i + 1])).T
                xlim = (par_edges[j][0], par_edges[j][-1])
                ylim = (par_edges[i + 1][0], par_edges[i + 1][-1])
                # print(params.shape)
                allgrid, tba = contprep(params, xlim=xlim ,ylim=ylim, **kwargs)
                print(tba)
                ax.contour(allgrid[0], allgrid[1], allgrid[2],
                           levels=tba, colors=color)
                # counts, xbins, ybins = np.histogram2d(par_list[j],
                #                                       par_list[i+1],
                #                                       bins=(par_edges[j],
                #                                             par_edges[i+1]),
                #                                       normed=normed)
                # ax.contour(counts.T, extent=[xbins.min(), xbins.max(),
                #                              ybins.min(), ybins.max()],
                #            linewidths=2, colors=color)

    return fig, axarr