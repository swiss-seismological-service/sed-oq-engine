# Copyright (c) 2010-2014, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

"""
Core functionality for the classical PSHA hazard calculator.

The difficult part of this calculator is the management of the logic tree
realizations. Let me explain how it works in a real life case.

We want to perform a SHARE calculation with a single source
model with two tectonic region types (Stable Shallow Crust and
Active Shallow Crust) and a complex GMPEs logic tree with 7
branching points with the following structure:

Active_Shallow: 4 GMPEs, weights 0.35, 0.35, 0.20, 0.10
Stable_Shallow: 5 GMPEs, weights 0.2, 0.2, 0.2, 0.2, 0.2
Shield: 2 GMPEs, weights 0.5, 0.5
Subduction_Interface: 4 GMPEs, weights 0.2, 0.2, 0.2, 0.4
Subduction_InSlab: 4 GMPEs, weights 0.2, 0.2, 0.2, 0.4
Volcanic: 1 GMPE, weight 1
Deep: 2 GMPEs, weights 0.6, 0.4

The number of realizations generated by this logic tree is
4 * 5 * 2 * 4 * 4 * 1 * 2 = 1280 and at the end we will
generate 1280 hazard curve containers. However the independent
hazard curve containers are much less, there are actually only 9 of
them, 4 coming from the Active Shallow Crust tectonic region type
and 5 from the Stable Shallow Crust tectonic region type.
The dependent hazard curves (i.e. hazard curves by realization)
can be obtained from the independent hazard curves by a composition
rule implemented in :method:`openquake.engine.models.HazardCurve.build_data`

NB: notice that 1280 / 9 = 142.22, therefore storing all the hazard
curves takes 140+ times more disk space and resources than actually needed.
The plan for the future is store only the independent curves and to
compose them on-the-fly, when they are looked up for a given realization,
since the composition is pretty fast (there are just numpy multiplications),
faster than reading from the database all the redundant data. The
impact on hazard maps has to be determined.
"""
import time
import operator
import itertools

import numpy

from openquake.hazardlib.imt import from_string
from openquake.hazardlib.geo.utils import get_spherical_bounding_box
from openquake.hazardlib.geo.utils import get_longitudinal_extent
from openquake.hazardlib.geo.geodetic import npoints_between

from openquake.engine import logs, writer
from openquake.engine.calculators.hazard import general
from openquake.engine.calculators.hazard.classical import (
    post_processing as post_proc)
from openquake.engine.db import models
from openquake.engine.utils import tasks
from openquake.engine.performance import LightMonitor


inserter = writer.CacheInserter(models.SourceInfo, 100)


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
        return (self.min_dist is not None and self.west is not None
                and self.south is not None)


def _calc_pnes(gsim, r_sites, rupture, imts, imls, truncation_level,
               make_ctxt_mon, calc_poes_mon):
    # compute the probabilities of no exceedence for each IMT
    # for the given gsim and rupture; returns a list of pairs
    # [(imt, pnes), ...]
    with make_ctxt_mon:
        sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
    with calc_poes_mon:
        for imt, levels in itertools.izip(imts, imls):
            poes = gsim.get_poes(
                sctx, rctx, dctx, imt, levels, truncation_level)
            pnes = rupture.get_probability_no_exceedance(poes)
            yield r_sites.expand(pnes, placeholder=1)


@tasks.oqtask
def compute_hazard_curves(
        job_id, sitecol, sources, trt_model_id, task_no):
    """
    This task computes R2 * I hazard curves (each one is a
    numpy array of S * L floats) from the given source_ruptures
    pairs.

    :param job_id:
        ID of the currently running job
    :param sitecol:
        a :class:`openquake.hazardlib.site.SiteCollection` instance
    :param sources:
        a block of source objects
    :param trt_model:
        a :class:`openquake.engine.db.TrtModel` instance
    :param int task_no:
        the ordinal number of the current task
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    total_sites = len(sitecol)
    sitemesh = sitecol.mesh
    sorted_imts = sorted(hc.intensity_measure_types_and_levels)
    sorted_imls = [hc.intensity_measure_types_and_levels[imt]
                   for imt in sorted_imts]
    sorted_imts = map(from_string, sorted_imts)
    trt_model = models.TrtModel.objects.get(pk=trt_model_id)
    gsims = trt_model.get_gsim_instances()
    curves = [[numpy.ones([total_sites, len(ls)]) for ls in sorted_imls]
              for gsim in gsims]
    if hc.poes_disagg:  # doing disaggregation
        lt_model_id = trt_model.lt_model.id
        bbs = [BoundingBox(lt_model_id, site_id) for site_id in sitecol.sids]
    else:
        bbs = []
    mon = LightMonitor(
        'getting ruptures', job_id, compute_hazard_curves)
    make_ctxt_mon = LightMonitor(
        'making contexts', job_id, compute_hazard_curves)
    calc_poes_mon = LightMonitor(
        'computing poes', job_id, compute_hazard_curves)

    num_sites = 0
    # NB: rows are namedtuples with fields (source, rupture, rupture_sites)
    for source, rows in itertools.groupby(
            hc.gen_ruptures(sources, mon, sitecol),
            key=operator.attrgetter('source')):
        t0 = time.time()
        num_ruptures = 0
        for _source, rupture, r_sites in rows:
            num_sites = max(num_sites, len(r_sites))
            num_ruptures += 1
            if hc.poes_disagg:  # doing disaggregation
                jb_dists = rupture.surface.get_joyner_boore_distance(sitemesh)
                closest_points = rupture.surface.get_closest_points(sitemesh)
                for bb, dist, point in itertools.izip(
                        bbs, jb_dists, closest_points):
                    if dist < hc.maximum_distance:
                        # ruptures too far away are ignored
                        bb.update([dist], [point.longitude], [point.latitude])

            # compute probabilities for all realizations
            for gsim, curv in itertools.izip(gsims, curves):
                for i, pnes in enumerate(_calc_pnes(
                        gsim, r_sites, rupture, sorted_imts, sorted_imls,
                        hc.truncation_level, make_ctxt_mon, calc_poes_mon)):
                    curv[i] *= pnes

        inserter.add(
            models.SourceInfo(trt_model_id=trt_model_id,
                              source_id=source.source_id,
                              source_class=source.__class__.__name__,
                              num_sites=num_sites,
                              num_ruptures=num_ruptures,
                              occ_ruptures=num_ruptures,
                              calc_time=time.time() - t0))

    make_ctxt_mon.flush()
    calc_poes_mon.flush()
    inserter.flush()

    # the 0 here is a shortcut for filtered sources giving no contribution;
    # this is essential for performance, we want to avoid returning
    # big arrays of zeros (MS)
    curves_by_gsim = [
        (gsim.__class__.__name__,
         [0 if general.all_equal(c, 1) else 1. - c for c in curv])
        for gsim, curv in zip(gsims, curves)]
    return curves_by_gsim, trt_model_id, bbs


class ClassicalHazardCalculator(general.BaseHazardCalculator):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = compute_hazard_curves

    def pre_execute(self):
        """
        Do pre-execution work. At the moment, this work entails:
        parsing and initializing sources, parsing and initializing the
        site model (if there is one), parsing vulnerability and
        exposure files and generating logic tree realizations. (The
        latter piece basically defines the work to be done in the
        `execute` phase.).
        """
        weights = super(ClassicalHazardCalculator, self).pre_execute()
        # a dictionary with the bounding boxes for earch source
        # model and each site, defined only for disaggregation
        # calculations:
        if self.hc.poes_disagg:
            lt_models = models.LtSourceModel.objects.filter(
                hazard_calculation=self.hc)
            self.bb_dict = dict(
                ((lt_model.id, site.id), BoundingBox(lt_model.id, site.id))
                for site in self.hc.site_collection
                for lt_model in lt_models)
        return weights

    def post_process(self):
        """
        Optionally generates aggregate curves, hazard maps and
        uniform_hazard_spectra.
        """
        logs.LOG.debug('> starting post processing')

        # means/quantiles:
        if self.hc.mean_hazard_curves or self.hc.quantile_hazard_curves:
            self.do_aggregate_post_proc()

        # hazard maps:
        # required for computing UHS
        # if `hazard_maps` is false but `uniform_hazard_spectra` is true,
        # just don't export the maps
        if self.hc.hazard_maps or self.hc.uniform_hazard_spectra:
            self.parallelize(
                post_proc.hazard_curves_to_hazard_map_task,
                post_proc.hazard_curves_to_hazard_map_task_arg_gen(self.job),
                lambda res: None)

        if self.hc.uniform_hazard_spectra:
            post_proc.do_uhs_post_proc(self.job)

        logs.LOG.debug('< done with post processing')
