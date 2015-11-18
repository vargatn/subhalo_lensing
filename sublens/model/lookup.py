__author__ = 'vtn'

import multiprocessing as multi
import numpy as np
import pickle
import math


class LookupTable:

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
        self.grid, self.flatgrid, self.par_axes, self.edges = self._mkgrid()

        # creating 1-D list of parameter dicts
        self.fargs = self._mkparlist()

        # at this point the lookup table is not yet created
        self.x_flat = None
        self.y_flat = None
        self.x_grid = None
        self.y_grid = None

    @classmethod
    def from_file(cls, file_name):
        params = pickle.load(open(file_name, "rb"))
        ltab = cls(params['func'], params['grid_axes'], params['extra_params'],
                   params['auto_params'])

        # ltab.anames = params['anames']
        # ltab.grid = params['grid']
        # ltab.flatgrid = params['flatgrid']
        # ltab.par_axes = params['par_axes']
        # ltab.edges = params['edges']
        # ltab.fargs = params['fargs']
        ltab.x_flat = params['x_flat']
        ltab.y_flat = params['y_flat']
        ltab.x_grid = params['x_grid']
        ltab.y_grid = params['y_grid']

        return ltab

    def save(self, save_path):
        save_dict = {
            "func": self.func,
            "anames": self.anames,
            "grid_axes": self.grid_axes,
            "auto_params": self.auto_params,
            "extra_params": self.extra_params,
            "grid": self.grid,
            "flatgrid": self.flatgrid,
            "par_axes": self.par_axes,
            "edges": self.edges,
            "fargs": self.fargs,
            "x_flat": self.x_flat,
            "y_flat": self.y_flat,
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
        }

        pickle.dump(save_dict, open(save_path, "wb"))

    def _mkgrid(self):
        edges = []
        pars = []
        for i, ax in enumerate(self.grid_axes):
            scale = str(ax[3])  # this should be 'lin' or 'log'
            if scale == str('lin'):
                tmp_ax = np.linspace(ax[1][0], ax[1][1], num=ax[2])
                diff = np.mean(np.diff(tmp_ax)) / 2.
                edge = np.array([tmp_ax[0]] + list(tmp_ax[:-1] + diff) +
                                [tmp_ax[-1]])
            elif scale == str('log'):
                tmp_ax = np.logspace(math.log10(ax[1][0]),
                                     math.log10(ax[1][1]), num=ax[2])
                diff = np.mean(np.diff(np.log10(tmp_ax))) / 2.
                edge = np.array([np.log10(tmp_ax[0])] +
                                list(np.log10(tmp_ax[:-1]) + diff) +
                                [np.log10(tmp_ax[-1])])
                edge = 10. ** edge
            else:
                print(scale)
                print(scale is 'log')
                raise NameError('scale should be "lin" or "log"')

            pars.append(tmp_ax)
            edges.append(edge)
        par_axes = np.array(pars)
        bin_edges = np.array(edges)
        grid = np.array(np.meshgrid(*par_axes, indexing='ij'))
        flattened_grid = np.array([arr.flatten() for arr in grid])

        return grid, flattened_grid, par_axes, bin_edges

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
        pool = multi.Pool(processes=n_multi)

        # evaluating func for the reference_list
        ref_list = np.array(pool.map(self.func, self.fargs))

        # assuming: 0: x values, 1: y values for profile
        self.x_flat = np.array(ref_list[:, 0])
        self.y_flat = np.array(ref_list[:, 1])

        gshape = self.grid[0].shape

        # regaining original shape
        newshape = gshape + self.y_flat.shape[1:]
        self.x_grid = self.x_flat.reshape(newshape)
        self.y_grid = self.y_flat.reshape(newshape)

    def q_eval(self, query_point, n_multi=1):
        """evaluates points exactly"""
        point = np.array(query_point)
        if len(point.shape) == 1:
            point = np.array([point])
        assert point.shape[1] == len(self.flatgrid)

        fargs = []
        for p in point:
            farg = {}

            # adding gridded params
            [farg.update({name: p[j]})
             for j, name in enumerate(self.anames)]

            # adding extra params
            farg.update(self.extra_params)

            # adding autoscaled params
            if self.auto_params is not None:
                [farg.update({auto[0]: auto[1](**farg)})
                for auto in self.auto_params]

            fargs.append(farg)

        pool = multi.Pool(processes=n_multi)
        ref_list = np.array(pool.map(self.func, fargs))
        x_flat = ref_list[:, 0]
        y_flat = ref_list[:, 1]

        x_prof = np.mean(x_flat, axis=0)
        y_prof = np.mean(y_flat, axis=0)

        return x_prof, y_prof

    def query(self, query_point):
        """
        performs a grid query

        projects the query points to the parameter grid and returns the
        weighted average of the grid profiles
        """
        point = np.array(query_point)

        if len(point.shape) == 1:
            point = np.array([point])

        assert point.shape[1] == len(self.flatgrid)

        counts, edges = np.histogramdd(point, bins=self.edges)
        weights = counts.flatten()

        x_mix = np.average(self.x_flat, weights=weights, axis=0)
        y_mix = np.average(self.y_flat, weights=weights, axis=0)

        return x_mix, y_mix

