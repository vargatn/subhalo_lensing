"""
Parameter table and distribution matching
"""
import datetime
import pickle
import warnings

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from sublens.model import ConvertorBase
from ..io.iocosmo import get_cosmopars
from ..model.profiles import DeltaSigmaProfile


class TableMaker(object):
    def __init__(self, profobj, convertor=None, **kwargs):
        """
        Creates a parameter table which can be used to evaluate the profiles on

        In this approach I store the bin centers, and the edges are calculated
        dynamically around them. This hopefully enables to later merge
        different tables, as the center lists are concatenated

        :param profobj: subclass DeltaSigmaProfile
        :param convertor:
        :param etype:
        :param kwargs:
        """
        # print(convertor, "hello")

        if not isinstance(profobj, DeltaSigmaProfile):
            raise TypeError('profobj must be a subclass of DeltaSigmaProfile')

        self.profobj = profobj
        self.convertor = convertor
        self.kwargs = kwargs

        # keeping parameters in alphabetical order
        self.haspars = sorted(kwargs.keys())
        if not len(self.haspars) > 0:
            raise KeyError('no kwargs parameters are specified!')
        self.has_mids = [kwargs[key] for key in self.haspars]

        # checking that the parameter list is sorted in ascending order
        for i, par in enumerate(self.haspars):
            sortind = np.argsort(self.has_mids[i])
            if not list(sortind) == list(range(len(self.has_mids[i]))):
                raise ValueError('The parameter lists are NOT sorted!!')

        # checking convertor
        self.conversion_required = False
        if not (self.convertor is None or isinstance(self.convertor, ConvertorBase)):
            raise TypeError('convertor is of invalid type')

        # check if the table has all parameters that the profobj requires
        if self.haspars == self.profobj.requires:
            if convertor is not None:
                warnings.warn('convertor specified but not used!',
                              SyntaxWarning)

        # Checking if the convertor is able to fill in the missing paramters
        elif self.convertor is not None and\
                        set(self.convertor.requires) <= set(self.haspars):
            if (sorted(self.convertor.provides +
                           self.haspars) != self.profobj.requires):
                raise SyntaxError(('The convertor does not provide'
                                   ' the appropiate parameters, or overwrites'
                                   ' the specified ones'))
            self.conversion_required = True
        # otherwise raise error, becouse no calcultion can be performed
        else:
            raise SyntaxError('Not enough parameters defined'
                              ' to calculate profiles!')

        # creating grid
        self.grid = np.array(np.meshgrid(*(self.has_mids), indexing='ij'))
        self.fgrid = np.array([arr.flatten() for arr in self.grid]).T

        # flat grid containing all parameters
        self.fallgrid = None
        if not  self.conversion_required:
            self.fallgrid = self.fgrid

        # some placeholders for the upcoming table
        self.rr = None
        self.redges = None
        self.dstable = None
        self.cosmopars = get_cosmopars(self.profobj.cosmo)

        self.time_stamp = None
        self.hasprofile = False

    def convert(self):
        """Adds the converted parameters to the flat grid"""
        if not self.conversion_required:
            raise RuntimeError('Conversion is NOT required/possible!!!')

        convlist = self.convertor.convert(self.fgrid)
        if not convlist.shape[1] == len(self.convertor.provides):
            raise TypeError('Convertor returned array with invalid shape')

        # find appropriate place to put the new parameters
        sind = list(np.argsort(self.haspars + self.convertor.provides))
        self.fallgrid = np.hstack((self.fgrid, convlist))[:, sind]

        # resets the flag
        self.conversion_required = False

    def calc_ds(self, rr, mode="rr"):
        """Calculates the DeltaSigma profile for each parameter entry"""
        if not np.iterable(rr):
            rr = [rr]
        rr = np.array(rr)

        pnames = self.profobj.requires
        self.dstable = []
        for i, par in enumerate(self.fallgrid):
            if i%50 ==0:
                print(i)
            self.profobj.prepare(**dict(zip(pnames, par)))
            self.profobj.calc(rr, mode=mode)
            self.dstable.append(self.profobj.ds)

        self.dstable = np.array(self.dstable)
        self.rr = self.profobj.rr
        self.redges = self.profobj.redges
        self.profobj.reset()
        self.time_stamp = str(datetime.datetime.utcnow())
        self.hasprofile = True

    def make_table(self, fname=None):
        """Creates a dictionary containing all calculation results"""
        assert self.hasprofile, 'First need to calculate!!!'
        table = {
            'par_mids': self.kwargs,
            'par_grid': self.grid,
            'haspars': self.haspars,
            'has_mids': self.has_mids,
            'flat_grid': self.fgrid,
            'convertor': str(self.convertor),
            'flat_all_names': self.profobj.requires,
            'flat_all_grid': self.fallgrid,
            'dstable': self.dstable,
            'rr': self.rr,
            'redges': self.redges,
            'profile': str(self.profobj),
            'cosmopars': self.cosmopars,
            'time_stamp': self.time_stamp,
        }
        if isinstance(fname, str):
            pickle.dump(table, open(fname, 'wb'))
        return table


class LookupTable(object):
    """Lookup Table"""
    def __init__(self, table):
        """
        Enables re-averaging profiles based on the pre-calculated table
        """
        # table data
        self.table = table
        self.time_stamp = table['time_stamp']
        self.cosmopars = table['cosmopars']
        self.cosmo = FlatLambdaCDM(**self.cosmopars)
        self.profile = table['profile']
        self.redges = table['redges']
        self.rr = table['rr']
        self.dstable = table['dstable']
        self.flat_all_grid = table['flat_all_grid']
        self.flat_all_names = table['flat_all_names']
        self.convertor = table['convertor']
        self.flat_grid = table['flat_grid']
        self.has_mids = table['has_mids']
        self.haspars = table['haspars']
        self.par_grid = table['par_grid']
        self.par_mids = table['par_mids']

        # weights for each entry
        self.edges = None
        self.ww = np.ones(len(self.dstable))
        self.ds_comb = np.zeros(shape=self.rr.shape)

    def __str__(self):
        return "LookupTable"

    def get_edges(self, default_d_edge = 1.):
        """Creates DD histogram edges based on grid parameters"""
        edges = []
        for i, mid in enumerate(self.has_mids):
            if len(mid) > 1:
                diff = np.diff(self.has_mids[i])
                edge = np.array([mid[0] - diff[0]/2.] +
                                list(np.array(mid[:-1] + diff/2.)) +
                                [mid[-1] + diff[-1] / 2.])
            else:
                # edge = [mid[0] - default_d_edge, mid[0] + default_d_edge]
                edge = [-np.inf, np.inf]
            edges.append(edge)
        return edges

    def get_weights(self, sample):
        """Calculates weights based on the histogram counts within the edges"""
        gshape = self.par_grid.shape[1:]
        counts, _edges = np.histogramdd(sample, bins=self.edges)
        weights = counts.flatten() / np.sum(counts)
        return weights

    def combine_profile(self, sample):
        """Make mine like yours!... """
        self.edges = self.get_edges()
        self.ww = self.get_weights(sample)
        self.ds_comb = np.average(self.dstable, axis=0, weights=self.ww)
        return self.rr, self.ds_comb
