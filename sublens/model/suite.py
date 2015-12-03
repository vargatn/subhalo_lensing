__author__ = 'vtn'

import numpy as np
from .profiles import nfw_pars
from astropy.cosmology import FlatLambdaCDM


class ModelProf:

    def __init__(self, subarr, priscale, secscale, mofunc):
        """Sets up required objects"""
        # getting list of tracers
        tracers = []
        for sub in subarr:
            tracers += sub.tracers
        self.tracers = sorted(set(tracers))
        for sub in subarr:
            assert (self.tracers == sub.tracers,
                    "All subpatches hs to be the same")

        self.subarr = subarr

        # checking if the primary scaling accepts the available tracers
        assert self.tracers == priscale.req
        self.priscale = priscale

        # checking if the secondary scaling accepts the primary output
        assert self.priscale.prov == secscale.req
        self.secscale = secscale

        # checking if the model function accepts the secondary output
        assert self.secscale.prov == mofunc.req
        self.mofunc = mofunc

        self.mpars = mofunc.req


    def t_frame(self, dens=200, logpars=()):
        """Creates grid histogram for the space of tracers"""

        # defining edges for the parameter space grid
        t_edges = np.zeros(shape=(len(self.tracers), 2))
        for i, tra in enumerate(self.tracers):
            edges = np.array([sub.par_ranges[tra] for sub in self.subarr])
            t_edges[i, :] = np.array([np.min(edges[:, 0]),
                                      np.max(edges[:, 1])])

        # define linearly and logarithmically scaled tracer axes
        t_scaling = np.asanyarray(['lin' for t in self.tracers])
        for t in logpars:
            t_scaling[np.where(np.asanyarray(self.tracers) == t)] = 'log'

        self.t_scaling = t_scaling

        tf_edges = []
        for i, tsc in enumerate(self.t_scaling):
            if tsc == 'log':
                tf_edges.append(np.logspace(np.log10(t_edges[i, 0]),
                                               np.log10(t_edges[i, 1]),
                                               num=dens))
            elif tsc == 'lin':
                tf_edges.append(np.linspace(t_edges[i, 0], t_edges[i, 1],
                                               num=dens))

        self.tf_edges = np.array(tf_edges)

        # calculating grid centers
        frame_cens = []
        for e in tf_edges:
            frame_cens.append(e[:-1] + np.diff(e) / 2.)
        self.ctgrid = np.array(np.meshgrid(*frame_cens, indexing='ij'))

        # creating the tracer histogram for all sub
        subgrid = []
        for i, sub in enumerate(self.subarr):
            hist = np.histogramdd(sub.data, bins=self.tf_edges)
            subgrid.append(hist[0])

        self.subgrid  = np.array(subgrid)

    def m_frame(self, range, dens=100, logpars=()):
        """
        creates a sufficiently dense model frame for model func evaluation
        """
        #obtain edges for model frame
        m_edges = np.array([range[key] for key in self.priscale.prov])

        # define linearly and logscaled edges
        m_scaling = np.asanyarray(['lin' for m in self.priscale.prov])
        for m in logpars:
            m_scaling[np.where(np.asanyarray(self.priscale.prov) == m)] = 'log'
        self.m_scaling = m_scaling

        mf_edges = []
        for i, msc in enumerate(self.m_scaling):
            if msc == 'log':
                mf_edges.append(np.logspace(np.log10(m_edges[i, 0]),
                                               np.log10(m_edges[i, 1]),
                                               num=dens))
            elif msc == 'lin':
                mf_edges.append(np.linspace(m_edges[i, 0], m_edges[i, 1],
                                               num=dens))

        self.tf_edges = np.array(mf_edges)

        # calculating grid centers
        frame_cens = []
        for e in mf_edges:
            frame_cens.append(e[:-1] + np.diff(e) / 2.)
        self.cmgrid = np.array(np.meshgrid(*frame_cens, indexing='ij'))













class ModelFunc:
    """
    Class wrapping actual function evaluations
    This is now an example for the nfw pars
    """
    def __init__(self, func, req, edges, **kwargs):
        self.func = func
        self.req = req
        self.edges = edges
        self.fargs = kwargs

    def point_eval(self, point):
        return self.func(*point, edges=self.edges)




