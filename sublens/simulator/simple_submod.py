
import numpy as np
import multiprocessing as multi


def parmaker(**kwargs):
    # gets names of parameters
    par_names = sorted(kwargs.keys())
    # gets edges for parameters
    edges = [kwargs[key] for key in par_names]

    grid_list = np.meshgrid(*edges, indexing='ij')
    flatgrid = np.array([grid.flatten() for grid in grid_list]).T

    log_dict = {
        'names': par_names,
        'pars': flatgrid,
    }
    return log_dict


class pooler(object):
    def __init__(self):
        # self.halo = halo
        # self.pardict = pardict
        # self.par_names = self.pardict['names']
        # self.pars = self.pardict['pars']
        pass

    def calc_ring(self, halo, pardict, edges, n_multi=1):

        par_names = pardict['names']
        pars = pardict['pars']

        # print(pars[0, :])

        # par

        # halo.ocen_ds_ring(edges, 1e)
        # pool = multi.Pool(processes=n_multi)

        setting = (halo, edges, 0.0, pars[0, 0], pars[0, 1])

        self._pooler(setting)


    def _pooler(self, setting):
        print(setting[0].ocen_ds_ring(*setting[1:]))


class mocksub(object):

    def __init__(self):
        pass