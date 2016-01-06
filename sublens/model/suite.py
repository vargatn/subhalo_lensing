__author__ = 'vtn'

import numpy as np
from .profiles import nfw_pars
from astropy.cosmology import FlatLambdaCDM
import multiprocessing as multi
import collections




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

        # predeclaring params
        self.t_scaling = None
        self.tf_edges = None
        self.ctgrid = None
        self.subgrid = None

        self.m_scaling = None
        self.mf_edges = None
        self.cmgrid = None

        self.ref_list = None


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
            # print(hist[0].shape)

        self.subgrid  = np.array(subgrid)

    def m_frame(self, range, dens=100, logpars=()):
        """
        creates a sufficiently dense model frame for model func evaluation
        """

        if not isinstance(dens, collections.Sequence):
            adens = [dens for i in enumerate(self.priscale.prov)]
        else:
            adens = dens

        # print(adens)
        # if dens is int or dens is float:
        # dens = [dens]* len(self.m_scaling)

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
                                               num=adens[i]))
            elif msc == 'lin':
                mf_edges.append(np.linspace(m_edges[i, 0], m_edges[i, 1],
                                               num=adens[i]))

        self.mf_edges = mf_edges
        # print(self.mf_edges.shape)
        # calculating grid centers
        frame_cens = []
        for e in mf_edges:
            frame_cens.append(e[:-1] + np.diff(e) / 2.)
        self.cmgrid = np.array(np.meshgrid(*frame_cens, indexing='ij'))

    def build_lookup(self, n_multi=1):
        """calculates reference profiles"""
        assert self.mf_edges is not None
        assert self.cmgrid is not None

        # creating complete model table for grid evaluation
        # pgrid = np.rollaxis(self.cmgrid, 0, start=len(self.cmgrid.shape))
        mgrid_shape = self.cmgrid.shape[1:]
        pflat = np.array([arr.flatten() for arr in self.cmgrid]).T

        self.model_flat = np.array([self.secscale.scale(par) for par in pflat])
        # print(self.model_flat)
        # print(self.model_flat.shape)
        # TODO add multiprocessing call here

        ref_list = np.array([self.mofunc.point_eval(pars)
                             for pars in self.model_flat])

        self.ref_list = ref_list

    def scale_frame(self, pars):
        """transforms the t_frame to m_frame"""
        assert self.ref_list is not None

        # first calculate which cell points to which other,
        #  by scaling each center point
        # print(self.ctgrid.shape)
        cflat = np.array([arr.flatten() for arr in self.ctgrid]).T
        mdist = np.array([self.priscale.fitscale(val, pars) for val in cflat])

        self.mdist = mdist

        # for each sub calculate the model space distribution
        self.subdist = []
        for i, sub in enumerate(self.subarr):
            counts = self.subgrid[i].flatten()
            # print(counts.shape)
            self.subdist.append(np.histogramdd(mdist, bins=self.mf_edges,
                                               weights=counts)[0])
            # print(self.subdist[i].shape)

        self.subdist = np.array(self.subdist)

    def collapse_profiles(self):
        """Creates prediction profiles for each subatch"""

        rr_values = self.ref_list[0, 0, :]
        ds_profiles = self.ref_list[:, 1, :]
        # print(ds_profiles.shape)
        # print(rr_values.shape)
        prof_y = []
        for i, sub in enumerate(self.subarr):
            counts = self.subdist[i].flatten()
            prof_y.append(np.average(ds_profiles, weights=counts, axis=0))

        self.prof_y = np.array(prof_y)
        self.prof_x = np.array([rr_values for val in prof_y])

        # creating data vector
        self.model = self.prof_y.flatten()
        self.rr = self.prof_x.flatten()


class ExactProf(ModelProf):

    # def par_join(self):
    #     self.ctarr = [sub.data for sub in self.subarr]


    def scale_frame(self, pars):
        """transforms  to exact parameters"""
        carr = [sub.data for sub in self.subarr]
        # cflat = self.subarr[0].data

        mdarr = [np.array([self.priscale.fitscale(val, pars)
                           for val in cflat])
                 for cflat in carr]
        # mdist = np.array([self.priscale.fitscale(val, pars) for val in cflat])

        # pdist = np.array([self.secscale.scale(par) for par in mdist])
        pdist = [np.array([self.secscale.scale(par) for par in mdist]) for mdist in mdarr]
        self.pdist = pdist

    def collapse_profiles(self):

        self.prof_x = []
        self.prof_y = []
        for pd in self.pdist:
            ref_list = np.array([self.mofunc.point_eval(pars) for pars in pd])
            # print(ref_list.shape)
            self.prof_x.append(ref_list[0, 0, :])
            # print(np.mean(ref_list[:, 1, :], axis=0).shape)
            self.prof_y.append(np.average(ref_list[: 1, :], axis=0))

        self.prof_x = np.asarray(self.prof_x)
        self.prof_y = np.asarray(self.prof_y)

        self.model = self.prof_y[:, 1, :].flatten()
        self.rr = self.prof_x.flatten()

        return self.prof_x, self.prof_y[:, 1, :]










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




