__author__ = 'vtn'

import multiprocessing as multi
import numpy as np
import math


class LookupTable:

    # TODO add saving capability

    def __init__(self, func, grid_axes, extra_params, auto_params=None):
        """
        Simplified grid evaluation

        :param func: Function to call at each point
        :param grid_axes: [[par_name, (min, max), numpoints, scale]]
        :param auto_params: [[par_name, func, arg_names]]
        :param extra_params: dict of extra params
        """

        assert hasattr(func, '__call__')
        self.func = func

        self.anames = [ax[0] for ax in grid_axes]

        self.grid_axes = grid_axes
        self.auto_params = auto_params
        self.extra_params = extra_params

        # creating N-D grid
        self.grid, self.flatgrid = self._mkgrid()

        # creating 1-D list of parameter dicts
        self.fargs = self._mkparlist()

        # at this point the lookup table is not yet created
        self.x_flat = None
        self.y_flat = None
        self.x_grid = None
        self.y_grid = None

    def _mkgrid(self):
        pars = []
        for i, ax in enumerate(self.grid_axes):
            scale = ax[3]  # this should be 'lin' or 'log'
            if scale is 'lin':
                tmp_ax = np.linspace(ax[1][0], ax[1][1], num=ax[2])
            elif scale is 'log':
                tmp_ax = np.logspace(math.log10(ax[1][0]),
                                     math.log10(ax[1][1]), num=ax[2])
            else:
                raise NameError('scale should be "lin" or "log"')

            pars.append(tmp_ax)

        par_axes = np.array(pars)
        grid = np.meshgrid(*par_axes, indexing='ij')
        flattened_grid = np.array([arr.flatten() for arr in grid])

        return grid, flattened_grid

    def _mkparlist(self):
        """creates list of dicts for func call parameters"""
        npoints = self.flatgrid.shape[1]
        fargs = []
        for i in range(npoints):
            farg = {}

            # adding gridded params
            [farg.update({name: self.flatgrid[j, i]})
             for j, name in enumerate(self.anames)]

            # adding extra params
            farg.update(self.extra_params)

            # adding autoscaled params
            if self.auto_params is not None:
                [farg.update({auto[0]: auto[1](**farg)})
                for auto in self.auto_params]

            fargs.append(farg)

        return fargs

    def _reftable(self, n_multi=1):
        """
        Creates a lookup reference table

        Evaluates a passed function at each point of an N-dimensional
         parameter grid.
        """

        # setting up multiprocessing pool
        # pool = multi.Pool(processes=n_multi)
        pool = multi.Pool(processes=1)

        # evaluating func for the reference_list
        ref_list = np.array(pool.map(self.func, self.fargs))

        # assuming: 0: x values, 1: y values for profile
        self.x_flat = ref_list[:, 0]
        self.y_flat = ref_list[:, 1]

        gshape = self.grid[0].shape

        # regaining original shape
        newshape = gshape + self.y_flat.shape[1:]
        self.x_grid = self.x_flat.reshape(newshape)
        self.y_grid = self.y_flat.reshape(newshape)

    # TODO add query API
    # TODO add query (interpolate)
    # TODO add query (grid)