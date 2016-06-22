"""
Shear calculation with troughs
"""

import numpy as np
import healpy as hp


class TroughHandler(object):
    """I/O with trough healpix maps"""

    def __init__(self, trough_names, zrange, tradius):
        """
        Loads troughs based on a list of file names

        parameters:
        ------------------
        :param trough_names: list of the paths to trough map fits files

        :param zrange: tuple of the redshift range spanned by the troughs

        :param tradius: the selection radius of the troughs in *radians*
        """
        self.names = trough_names
        self.zrange = zrange
        self.tradius = tradius
        self.tmaps = []
        self.smaps = []
        self.tpixes = []
        self.tvecs = []
        self.nsides = []
        self.nside = None

        self.oenvs = []
        self.ovlens = []
        self.ovics = []

        # loading datasets
        for i, name in enumerate(self.names):
            tmap = hp.read_map(name, verbose=False)
            self.nsides.append(hp.npix2nside(len(tmap)))
            self.tmaps.append(tmap)
            self.tpixes.append(np.nonzero(tmap)[0])
            self.smaps.append(tmap[self.tpixes[i]])
            self.tvecs.append(
                np.array(hp.pix2vec(self.nsides[i], self.tpixes[i])).T)

        if not np.all(self.nsides):
            raise ValueError("different nsides are not supported!")
        else:
            self.nside = self.nsides[0]

    def _get_environments(self, ovec, sradius):
        """Healpix environment of a vector direction"""
        self.oenvs = [hp.query_disc(self.nside, ov, radius=sradius) for ov in
                      ovec]

    def _get_tvicinity(self):
        """Selects the troughs in each localc environments"""
        self.ovics = []
        self.ovlens = []
        for i, tpix in enumerate(self.tpixes):
            ovic = []
            ovlen = []
            tset = set(tpix)
            for j, oenv in enumerate(self.oenvs):
                ov = list(set(oenv).intersection(tset))
                ovic.append(ov)
                ovlen.append(len(ov))

            self.ovics.append(ovic)
            self.ovlens.append(np.array(ovlen))

        self.ovlens = np.array(self.ovlens)

    def calc_oweights(self, ovec, sradius=None):
        """
        Calculates object weights and trough environments for the
         specified objects

        parameters:
        --------------
        :param ovec: list or array of 3D unit vectors (directions)
                     for each object

        :param sradius: search radius for the trough selection,
                        if None uses trough radius

        :returns: array of summed weights for each object
        """
        if sradius is None:
            self._get_environments(ovec, self.tradius)
        else:
            self._get_environments(ovec, sradius)

        self._get_tvicinity()

        oweights = []
        for i, ovic in enumerate(self.ovics):
            oweight = []
            for ov in ovic:
                ws = 0.0
                if len(ov) > 0:
                    ws = np.sum(self.tmaps[i][np.array(ov)])
                oweight.append(ws)
            oweights.append(oweight)
        return np.array(oweights)


class ClusterFormatter(object):
    """Creates a cluster position table with matched size"""

    def __init__(self, cvec, cra, cdec, cz):
        """
        Creates a table which a row for each cluster trough pair

        parameters:
        --------------
        :param cvec: 3D coordinate vectors of clusters on the unit sphere

        :param cra: RA list for clusters

        :param cdec: DEC list for clusters

        :param czz: redshift list for clusters
        """
        self.cvec = cvec
        self.cra = cra
        self.cdec = cdec
        self.cz = cz

        self.cvlens = []
        self.cinds = []

    def _get_cinds(self, cvlens):
        """
        Creates a list of matched indices for the cluster-trough pairs

        parameters:
        -------------
        :param cvlens: list containing the # of troughs within the
                       search radius of each cluster
        """
        self.cvlens = cvlens

        self.cinds = []
        for i, cvlen in enumerate(self.cvlens):
            cind = np.zeros(int(np.sum(cvlen)), dtype=int)
            ind = 0
            for j, ln in enumerate(cvlen):
                cind[ind:ind + ln] = j
                ind += ln
            self.cinds.append(cind)

    def calc_pos(self, cvlens):
        """
        Creates a trough matched cluster position (RA, DEC) table

        parameters:
        -------------
        :param cvlens: list containing the # of troughs within the
                       search radius of each cluster

        :returns: indices for the corresponding entries in the matched array
        """
        self._get_cinds(cvlens)
        self.craa = [self.cra[cind] for cind in self.cinds]
        self.cdee = [self.cdec[cind] for cind in self.cinds]
        self.czzz = [self.cz[cind] for cind in self.cinds]

        self.cveec = [self.cvec[cind, :] for cind in self.cinds]
        self.cpoos = [np.vstack((craa, cdee)).T for craa, cdee in
                      zip(self.craa, self.cdee)]

    def calc_linds(self, lind):
        slind = set(lind)
        linds = []
        for i, cvlen in enumerate(self.cvlens):
            cind = np.zeros(int(np.sum(cvlen)), dtype=bool)
            ind = 0
            for j, ln in enumerate(cvlen):
                val = False
                if j in slind:
                    val = True
                cind[ind:ind + ln] = val
                ind += ln
            linds.append(cind)
        return linds


class TroughFormatter(object):
    """Creates a trough position table with matched size"""

    def __init__(self, th, ovics):
        """
        Creates a trough position table with matched size

        parameters:
        -------------
        :param th: TroughHandler instance

        :param ovics: vicinities for each cluster to be considered
        """
        self.th = th
        self.ovics = ovics
        self.fovics = [np.array(np.concatenate(ovic), dtype=int) for ovic in
                       self.ovics]

        self.tveec = []

    def calc_tpos(self):
        """Calculates the actual matched sized vectors and positions"""
        self.tveec = [np.array(hp.pix2vec(self.th.nside, fovic)).T for fovic in
                      self.fovics]

        self.tww = [np.array(self.th.tmaps[i][fovic]) for i, fovic in
                    enumerate(self.fovics)]

        self.tpoos = [np.array(hp.pix2ang(self.th.nside, fovic)).T for fovic in
                      self.fovics]
        self.tpoos = [
            np.vstack((tpoos[:, 1], np.pi / 2 - tpoos[:, 0])).T * 180. / np.pi
            for tpoos in self.tpoos]

    def calc_dists(self, cf):
        """
        Calculates great circle (arc-length) distance between the two
        sets of positions
         """

        dists = [hp.rotator.angdist(cpoo.T, tpoo.T, lonlat=True) for
                 (cpoo, tpoo) in zip(cf.cpoos, self.tpoos)]
        return dists