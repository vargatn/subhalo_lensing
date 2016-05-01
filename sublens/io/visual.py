"""
Plotting and visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def corner(pars, par_list, par_edges, figsize=(8, 8), color='black', fig=None,
           axarr=None, mode="hist", cmap="gray_r", normed=True):
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

    for i, axrow in enumerate(axarr[1:]):
        for j, ax in enumerate(axrow[:i+1]):
            if mode == "hist":
                [[ax.hist2d(par_list[j], par_list[i+1],
                            bins=(par_edges[j], par_edges[i+1]), cmap=cmap,
                            normed=normed)
                  for j, ax in enumerate(axrow[:i+1])]
                 for i, axrow in enumerate(axarr[1:])]
            elif mode == "contour":
                counts, xbins, ybins = np.histogram2d(par_list[j],
                                                      par_list[i+1],
                                                      bins=(par_edges[j],
                                                            par_edges[i+1]),
                                                      normed=normed)
                ax.contour(counts.T, extent=[xbins.min(), xbins.max(),
                                             ybins.min(), ybins.max()],
                           linewidths=2, colors=color)

    return fig, axarr