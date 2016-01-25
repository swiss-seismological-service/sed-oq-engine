#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2014-2015, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division
import math
import logging
import operator
import collections
from functools import partial

import numpy

from openquake.hazardlib.geo.utils import get_spherical_bounding_box
from openquake.hazardlib.geo.utils import get_longitudinal_extent
from openquake.hazardlib.geo.geodetic import npoints_between
from openquake.hazardlib.calc.filters import source_site_distance_filter
from openquake.hazardlib.calc.hazard_curve import (
    hazard_curves_per_trt, zero_curves, zero_maps, agg_curves)
from openquake.risklib import scientific
from openquake.commonlib import parallel, source, datastore, sourceconverter
from openquake.baselib.general import AccumDict

from openquake.calculators import base, calc


HazardCurve = collections.namedtuple('HazardCurve', 'location poes')


# this is needed for the disaggregation
class BoundingBox(object):
    """
    A class to store the bounding box in distances, longitudes and magnitudes,
    given a source model and a site. This is used for disaggregation
    calculations. The goal is to determine the minimum and maximum
    distances of the ruptures generated from the model from the site;
    moreover the maximum and minimum longitudes and magnitudes are stored, by
    taking in account the international date line.
    """
    def __init__(self, lt_model_id, site_id):
        self.lt_model_id = lt_model_id
        self.site_id = site_id
        self.min_dist = self.max_dist = None
        self.east = self.west = self.south = self.north = None

    def update(self, dists, lons, lats):
        """
        Compare the current bounding box with the value in the arrays
        dists, lons, lats and enlarge it if needed.

        :param dists:
            a sequence of distances
        :param lons:
            a sequence of longitudes
        :param lats:
            a sequence of latitudes
        """
        if self.min_dist is not None:
            dists = [self.min_dist, self.max_dist] + dists
        if self.west is not None:
            lons = [self.west, self.east] + lons
        if self.south is not None:
            lats = [self.south, self.north] + lats
        self.min_dist, self.max_dist = min(dists), max(dists)
        self.west, self.east, self.north, self.south = \
            get_spherical_bounding_box(lons, lats)

    def update_bb(self, bb):
        """
        Compare the current bounding box with the given bounding box
        and enlarge it if needed.

        :param bb:
            an instance of :class:
            `openquake.engine.calculators.hazard.classical.core.BoundingBox`
        """
        if bb:  # the given bounding box must be non-empty
            self.update([bb.min_dist, bb.max_dist], [bb.west, bb.east],
                        [bb.south, bb.north])

    def bins_edges(self, dist_bin_width, coord_bin_width):
        """
        Define bin edges for disaggregation histograms, from the bin data
        collected from the ruptures.

        :param dists:
            array of distances from the ruptures
        :param lons:
            array of longitudes from the ruptures
        :param lats:
            array of latitudes from the ruptures
        :param dist_bin_width:
            distance_bin_width from job.ini
        :param coord_bin_width:
            coordinate_bin_width from job.ini
        """
        dist_edges = dist_bin_width * numpy.arange(
            int(self.min_dist / dist_bin_width),
            int(numpy.ceil(self.max_dist / dist_bin_width) + 1))

        west = numpy.floor(self.west / coord_bin_width) * coord_bin_width
        east = numpy.ceil(self.east / coord_bin_width) * coord_bin_width
        lon_extent = get_longitudinal_extent(west, east)

        lon_edges, _, _ = npoints_between(
            west, 0, 0, east, 0, 0,
            numpy.round(lon_extent / coord_bin_width) + 1)

        lat_edges = coord_bin_width * numpy.arange(
            int(numpy.floor(self.south / coord_bin_width)),
            int(numpy.ceil(self.north / coord_bin_width) + 1))

        return dist_edges, lon_edges, lat_edges

    def __nonzero__(self):
        """
        True if the bounding box is non empty.
        """
        return (self.min_dist is not None and self.west is not None and
                self.south is not None)


@parallel.litetask
def classical(sources, sitecol, siteidx, rlzs_assoc, monitor):
    """
    :param sources:
        a non-empty sequence of sources of homogeneous tectonic region type
    :param sitecol:
        a SiteCollection instance
    :param siteidx:
        index of the first site (0 if there is a single tile)
    :param rlzs_assoc:
        a RlzsAssoc instance
    :param monitor:
        a monitor instance
    :returns:
        an AccumDict rlz -> curves
    """
    max_dist = monitor.oqparam.maximum_distance
    truncation_level = monitor.oqparam.truncation_level
    imtls = monitor.oqparam.imtls
    trt_model_id = sources[0].trt_model_id
    # sanity check: the trt_model must be the same for all sources
    for src in sources[1:]:
        assert src.trt_model_id == trt_model_id
    gsims = rlzs_assoc.gsims_by_trt_id[trt_model_id]

    dic = AccumDict()
    dic.siteslice = slice(siteidx, siteidx + len(sitecol))
    if monitor.oqparam.poes_disagg:
        sm_id = rlzs_assoc.get_sm_id(trt_model_id)
        dic.bbs = [BoundingBox(sm_id, sid) for sid in sitecol.sids]
    else:
        dic.bbs = []
    # NB: the source_site_filter below is ESSENTIAL for performance inside
    # hazard_curves_per_trt, since it reduces the full site collection
    # to a filtered one *before* doing the rupture filtering
    curves_by_gsim = hazard_curves_per_trt(
        sources, sitecol, imtls, gsims, truncation_level,
        source_site_filter=source_site_distance_filter(max_dist),
        maximum_distance=max_dist, bbs=dic.bbs, monitor=monitor)
    dic.calc_times = monitor.calc_times  # added by hazard_curves_per_trt
    for gsim, curves in zip(gsims, curves_by_gsim):
        dic[trt_model_id, str(gsim)] = curves
    return dic


def expand(array, n, aslice):
    """
    Expand a short array to size n by adding zeros outside of the slice.
    For instance:

    >>> arr = numpy.array([1, 2, 3])
    >>> expand(arr, 5, slice(1, 4))
    array([0, 1, 2, 3, 0])
    """
    if len(array) > n:
        raise ValueError('The array is too large: %d > %d' % (len(array), n))
    elif len(array) == n:  # nothing to expand
        return array
    if aslice.stop - aslice.start != len(array):
        raise ValueError('The slice has %d places, but the array has length '
                         '%d', slice.stop - aslice.start, len(array))
    zeros = numpy.zeros((n,) + array.shape[1:], array.dtype)
    zeros[aslice] = array
    return zeros


@base.calculators.add('classical')
class ClassicalCalculator(base.HazardCalculator):
    """
    Classical PSHA calculator
    """
    core_task = classical
    source_info = datastore.persistent_attribute('source_info')

    def agg_dicts(self, acc, val):
        """
        Aggregate dictionaries of hazard curves by updating the accumulator.

        :param acc: accumulator dictionary
        :param val: a dictionary of hazard curves, keyed by (trt_id, gsim)
        """
        with self.monitor('aggregate curves', autoflush=True):
            if hasattr(val, 'calc_times'):
                acc.calc_times.extend(val.calc_times)
            for bb in getattr(val, 'bbs', []):
                acc.bb_dict[bb.lt_model_id, bb.site_id].update_bb(bb)
            self.agg_curves(acc, val)
        return acc

    def agg_curves(self, acc, val):
        """
        Aggregate the hazard curves. If tiling is enabled, expand them
        first to the full site collection.
        """
        n = len(self.sitecol)
        tiling = (self.oqparam.calculation_mode == 'classical' and
                  n > self.oqparam.sites_per_tile)
        for k, v in val.items():
            acc[k] = agg_curves(acc[k], expand(v, n, val.siteslice)
                                if tiling else v)

    @staticmethod
    def is_effective_trt_model(result_dict, trt_model):
        """
        Returns the number of tectonic region types
        with ID contained in the result_dict.

        :param result_dict: a dictionary with keys (trt_id, gsim)
        :param trt_model: a TrtModel instance
        """
        return sum(1 for key, val in result_dict.items()
                   if trt_model.id == key[0] and nonzero(val))

    def execute(self):
        """
        Run in parallel `core_task(sources, sitecol, monitor)`, by
        parallelizing on the sources according to their weight and
        tectonic region type.
        """
        monitor = self.monitor.new(self.core_task.__name__)
        monitor.oqparam = self.oqparam
        zc = zero_curves(len(self.sitecol.complete), self.oqparam.imtls)
        zerodict = AccumDict((key, zc) for key in self.rlzs_assoc)
        zerodict.calc_times = []
        zerodict.bb_dict = {
            (smodel.ordinal, site.id): BoundingBox(smodel.ordinal, site.id)
            for site in self.sitecol
            for smodel in self.csm.source_models
        } if self.oqparam.poes_disagg else {}
        curves_by_trt_gsim = self.manager.tm.reduce(self.agg_dicts, zerodict)
        with self.monitor('store source_info', autoflush=True):
            self.store_source_info(curves_by_trt_gsim)
        self.rlzs_assoc = self.csm.get_rlzs_assoc(
            partial(self.is_effective_trt_model, curves_by_trt_gsim))
        return curves_by_trt_gsim

    def store_source_info(self, curves_by_trt_gsim):
        # store the information about received data
        received = self.manager.tm.received
        if received:
            attrs = self.datastore['source_chunks'].attrs
            attrs['max_received'] = max(received)
            attrs['tot_received'] = sum(received)

        # then save the calculation times per each source
        calc_times = getattr(curves_by_trt_gsim, 'calc_times', [])
        if calc_times:
            sources = self.csm.get_sources()
            info_dict = {(rec['trt_model_id'], rec['source_id']): rec
                         for rec in self.source_info}
            for src_idx, dt in calc_times:
                src = sources[src_idx]
                info = info_dict[src.trt_model_id, src.source_id]
                info['calc_time'] += dt
            self.source_info = numpy.array(
                sorted(info_dict.values(), key=operator.itemgetter(7),
                       reverse=True))
        self.datastore.hdf5.flush()

    def post_execute(self, curves_by_trt_gsim):
        """
        Collect the hazard curves by realization and export them.

        :param curves_by_trt_gsim:
            a dictionary (trt_id, gsim) -> hazard curves
        """
        with self.monitor('save curves_by_trt_gsim', autoflush=True):
            for sm in self.rlzs_assoc.csm_info.source_models:
                group = self.datastore.hdf5.create_group(
                    'curves_by_sm/' + '_'.join(sm.path))
                group.attrs['source_model'] = sm.name
                for tm in sm.trt_models:
                    for i, gsim in enumerate(tm.gsims):
                        try:
                            curves = curves_by_trt_gsim[tm.id, gsim]
                        except KeyError:  # no data for the trt_model
                            pass
                        else:
                            ts = '%03d-%d' % (tm.id, i)
                            if nonzero(curves):
                                group[ts] = curves
                                group[ts].attrs['trt'] = tm.trt
                                group[ts].attrs['nbytes'] = curves.nbytes
                                group[ts].attrs['gsim'] = str(gsim)
                self.datastore.set_nbytes(group.name)
            self.datastore.set_nbytes('curves_by_sm')

        oq = self.oqparam
        with self.monitor('combine and save curves_by_rlz', autoflush=True):
            zc = zero_curves(len(self.sitecol.complete), oq.imtls)
            curves_by_rlz = self.rlzs_assoc.combine_curves(
                curves_by_trt_gsim, agg_curves, zc)
            rlzs = self.rlzs_assoc.realizations
            nsites = len(self.sitecol)
            if oq.individual_curves:
                for rlz, curves in curves_by_rlz.items():
                    self.store_curves('rlz-%03d' % rlz.ordinal, curves, rlz)

            if len(rlzs) == 1:  # cannot compute statistics
                [self.mean_curves] = curves_by_rlz.values()
                return

        with self.monitor('compute and save statistics', autoflush=True):
            weights = (None if oq.number_of_logic_tree_samples
                       else [rlz.weight for rlz in rlzs])
            mean = oq.mean_hazard_curves
            if mean:
                self.mean_curves = numpy.array(zc)
                for imt in oq.imtls:
                    self.mean_curves[imt] = scientific.mean_curve(
                        [curves_by_rlz[rlz][imt] for rlz in rlzs], weights)

            self.quantile = {}
            for q in oq.quantile_hazard_curves:
                self.quantile[q] = qc = numpy.array(zc)
                for imt in oq.imtls:
                    curves = [curves_by_rlz[rlz][imt] for rlz in rlzs]
                    qc[imt] = scientific.quantile_curve(
                        curves, q, weights).reshape((nsites, -1))

            if mean:
                self.store_curves('mean', self.mean_curves)
            for q in self.quantile:
                self.store_curves('quantile-%s' % q, self.quantile[q])

    def hazard_maps(self, curves):
        """
        Compute the hazard maps associated to the curves
        """
        maps = zero_maps(
            len(self.sitecol), self.oqparam.imtls, self.oqparam.poes)
        for imt in curves.dtype.fields:
            # build a matrix of size (N, P)
            data = calc.compute_hazard_maps(
                curves[imt], self.oqparam.imtls[imt], self.oqparam.poes)
            for poe, hmap in zip(self.oqparam.poes, data.T):
                maps['%s~%s' % (imt, poe)] = hmap
        return maps

    def store_curves(self, kind, curves, rlz=None):
        """
        Store all kind of curves, optionally computing maps and uhs curves.

        :param kind: the kind of curves to store
        :param curves: an array of N curves to store
        :param rlz: hazard realization, if any
        """
        oq = self.oqparam
        self._store('hcurves/' + kind, curves, rlz, nbytes=curves.nbytes)
        if oq.hazard_maps or oq.uniform_hazard_spectra:
            # hmaps is a composite array of shape (N, P)
            hmaps = self.hazard_maps(curves)
            if oq.hazard_maps:
                self._store('hmaps/' + kind, hmaps, rlz,
                            poes=oq.poes, nbytes=hmaps.nbytes)

    def _store(self, name, curves, rlz, **kw):
        self.datastore.hdf5[name] = curves
        dset = self.datastore.hdf5[name]
        if rlz is not None:
            dset.attrs['uid'] = rlz.uid
        for k, v in kw.items():
            dset.attrs[k] = v


def nonzero(val):
    """
    :returns: the sum of the composite array `val`
    """
    return sum(val[k].sum() for k in val.dtype.names)


def split_sources(sources, maxweight, splitmap):
    """
    Split the sources with weight greater than maxweight. `splitmap`
    is a cache to avoid splitting twice the same source.
    """
    ss = []
    for src in sources:
        if src.weight > maxweight:
            key = (src.trt_model_id, src.source_id)
            try:
                srcs = splitmap[key]
            except KeyError:
                logging.info('Splitting %s of weight %d > %d',
                             src, src.weight, maxweight)
                srcs = splitmap[key] = list(sourceconverter.split_source(src))
            ss.extend(srcs)
        else:
            ss.append(src)
    return ss
