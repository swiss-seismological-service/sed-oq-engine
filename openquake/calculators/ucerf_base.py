# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2018-2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import os
import copy
import math
import logging
import pickle
from datetime import datetime
import numpy
import h5py
import zlib

from openquake.baselib.general import AccumDict, cached_property
from openquake.hazardlib.calc.filters import SourceFilter
from openquake.hazardlib.source.base import BaseSeismicSource
from openquake.hazardlib.geo.geodetic import min_geodetic_distance
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.geo.surface.multi import MultiSurface
from openquake.hazardlib.geo.utils import KM_TO_DEGREES, angular_distance
from openquake.hazardlib.source.point import PointSource
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib import valid
from openquake.hazardlib.sourceconverter import SourceConverter

DEFAULT_TRT = "Active Shallow Crust"


def convert_UCERFSource(self, node):
    """
    Converts the node into an UCERFSource object
    """
    dirname = os.path.dirname(self.fname)  # where the source_model_file is
    source_file = os.path.join(dirname, node["filename"])
    if "startDate" in node.attrib and "investigationTime" in node.attrib:
        # Is a time-dependent model - even if rates were originally
        # poissonian
        # Verify that the source time span is the same as the TOM time span
        inv_time = float(node["investigationTime"])
        if inv_time != self.investigation_time:
            raise ValueError("Source investigation time (%s) is not "
                             "equal to configuration investigation time "
                             "(%s)" % (inv_time, self.investigation_time))
        start_date = datetime.strptime(node["startDate"], "%d/%m/%Y")
    else:
        start_date = None
    return UCERFSource(
        source_file,
        self.investigation_time,
        start_date,
        node["minMag"],
        npd=self.convert_npdist(node),
        hdd=self.convert_hddist(node),
        aspect=~node.ruptAspectRatio,
        upper_seismogenic_depth=~node.pointGeometry.upperSeismoDepth,
        lower_seismogenic_depth=~node.pointGeometry.lowerSeismoDepth,
        msr=valid.SCALEREL[~node.magScaleRel](),
        mesh_spacing=self.rupture_mesh_spacing,
        trt=node["tectonicRegion"])


SourceConverter.convert_UCERFSource = convert_UCERFSource


class ImperfectPlanarSurface(PlanarSurface):
    """
    The planar surface class sets a narrow tolerance for the rectangular plane
    to be distorted in cartesian space. Ruptures with aspect ratios << 1.0,
    and with a dip of less than 90 degrees, cannot be generated in a manner
    that is consistent with the definitions - and thus cannot be instantiated.
    This subclass modifies the original planar surface class such that the
    tolerance checks are over-ridden. We find that distance errors with respect
    to a simple fault surface with a mesh spacing of 0.001 km are only on the
    order of < 0.15 % for Rrup (< 2 % for Rjb, < 3.0E-5 % for Rx)
    """
    IMPERFECT_RECTANGLE_TOLERANCE = numpy.inf


class UcerfFilter(SourceFilter):
    """
    Filter for UCERF sources, both background and faults.
    """
    def filter(self, srcs):
        for src in srcs:
            if hasattr(src, 'start'):  # fault sources
                ridx = set()
                for arr in src.all_ridx:
                    ridx.update(arr)
                indices = self.get_indices(src, ridx, src.mags.max())
                if len(indices):
                    yield src, indices
            else:  # background sources
                yield from super().filter([src])

    def get_indices(self, src, ridx, mag):
        """
        :param src: an UCERF source
        :param ridx: a set of rupture indices
        :param mag: magnitude to use to compute the integration distance
        :returns: array with the IDs of the sites close to the ruptures
        """
        centroids = src.get_centroids(ridx)
        mindistance = min_geodetic_distance(
            (centroids[:, 0], centroids[:, 1]), self.sitecol.xyz)
        idist = self.integration_distance(DEFAULT_TRT, mag)
        indices, = (mindistance <= idist).nonzero()
        return indices


class UCERFSource(BaseSeismicSource):
    """
    :param source_file:
        Path to an existing HDF5 file containing the UCERF model
    :param float investigation_time:
        Investigation time of event set (years)
    :param start_date:
        Starting date of the investigation (None for time independent)
    :param float min_mag:
        Minimim magnitude for consideration of background sources
    :param npd:
        Nodal plane distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param hdd:
        Hypocentral depth distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param float aspect:
        Aspect ratio
    :param float upper_seismoge nic_depth:
        Upper seismogenic depth (km)
    :param float lower_seismogenic_depth:
        Lower seismogenic depth (km)
    :param msr:
        Magnitude scaling relation
    :param float mesh_spacing:
        Spacing (km) of fault mesh
    :param str trt:
        Tectonic region type
    :param float integration_distance:
        Maximum distance from rupture to site for consideration
    """
    code = b'U'
    MODIFICATIONS = set()
    tectonic_region_type = DEFAULT_TRT
    ruptures_per_block = None  # overridden by the source_reader
    checksum = 0
    _wkt = ''

    def __init__(
            self, source_file, investigation_time, start_date, min_mag,
            npd, hdd, aspect=1.5, upper_seismogenic_depth=0.0,
            lower_seismogenic_depth=15.0, msr=WC1994(), mesh_spacing=1.0,
            trt="Active Shallow Crust", integration_distance=1000):
        assert os.path.exists(source_file), source_file
        self.source_file = source_file
        self.source_id = None  # unset until .new is called
        self.inv_time = investigation_time
        self.start_date = start_date
        self.tom = self._get_tom()
        self.min_mag = min_mag
        self.npd = npd
        self.hdd = hdd
        self.aspect = aspect
        self.usd = upper_seismogenic_depth
        self.lsd = lower_seismogenic_depth
        self.msr = msr
        self.mesh_spacing = mesh_spacing
        self.tectonic_region_type = trt
        self.stop = None
        self.start = None

    @property
    def num_ruptures(self):
        return self.stop - self.start

    @num_ruptures.setter
    def num_ruptures(self, value):  # hack to make the sourceconverter happy
        pass

    @cached_property
    def mags(self):
        # read from FM0_0/MEANFS/MEANMSR/Magnitude
        with h5py.File(self.source_file, "r") as hdf5:
            arr = hdf5[self.idx_set["mag"]][self.start: self.stop]
        return arr

    @cached_property
    def rate(self):
        # read from FM0_0/MEANFS/MEANMSR/Rates/MeanRates
        with h5py.File(self.source_file, "r") as hdf5:
            return hdf5[self.idx_set["rate"]][self.start: self.stop]

    @cached_property
    def rake(self):
        # read from FM0_0/MEANFS/Rake
        with h5py.File(self.source_file, "r") as hdf5:
            return hdf5[self.idx_set["rake"]][self.start:self.stop]

    def wkt(self):
        return ''

    def count_ruptures(self):
        """
        The length of the rupture array if the branch_id is set, else 0
        """
        return self.num_ruptures

    def new(self, et_id, branch_id):
        """
        :param et_id: ordinal of the source group
        :param branch_name: name of the UCERF branch
        :param branch_id: string associated to the branch
        :returns: a new UCERFSource associated to the branch_id
        """
        new = copy.copy(self)
        new.et_id = et_id
        new.source_id = branch_id  # i.e. FM3_1/ABM/Shaw09Mod/
        # DsrUni_CharConst_M5Rate6.5_MMaxOff7.3_NoFix_SpatSeisU2
        new.idx_set = build_idx_set(branch_id, self.start_date)
        with h5py.File(self.source_file, "r") as hdf5:
            new.start = 0
            new.stop = len(hdf5[new.idx_set["mag"]])
        return new

    def get_min_max_mag(self):
        """
        Called when updating the SourceGroup
        """
        return self.min_mag, 10

    def _get_tom(self):
        """
        Returns the temporal occurence model as a Poisson TOM
        """
        return PoissonTOM(self.inv_time)

    def get_ridx(self, hdf5, iloc=None):
        """List of rupture indices for the given iloc"""
        if iloc is None:
            iloc = slice(self.start, self.stop)
        return hdf5[self.idx_set["geol"] + "/RuptureIndex"][iloc]

    def get_centroids(self, ridx):
        """
        :returns: array of centroids for the given rupture index
        """
        centroids = []
        with h5py.File(self.source_file, "r") as hdf5:
            for idx in ridx:
                trace = "{:s}/{:s}".format(self.idx_set["sec"], str(idx))
                centroids.append(hdf5[trace + "/Centroids"][()])
        return numpy.concatenate(centroids)

    def gen_trace_planes(self, ridx, hdf5):
        """
        :yields: trace and rupture planes for the given rupture index
        """
        for idx in ridx:
            trace = "{:s}/{:s}".format(self.idx_set["sec"], str(idx))
            plane = hdf5[trace + "/RupturePlanes"][:].astype("float64")
            yield trace, plane

    def get_bounding_box(self, maxdist):
        """
        :returns: min_lon, min_lat, max_lon, max_lat
        """
        with h5py.File(self.source_file, 'r') as hdf5:
            locations = hdf5["Grid/Locations"][()]
        lons, lats = locations[:, 0], locations[:, 1]
        bbox = lons.min(), lats.min(), lons.max(), lats.max()
        a1 = min(maxdist * KM_TO_DEGREES, 90)
        a2 = angular_distance(maxdist, bbox[1], bbox[3])
        return bbox[0] - a2, bbox[1] - a1, bbox[2] + a2, bbox[3] + a1

    def get_background_sids(self):
        """
        We can apply the filtering of the background sites as a pre-processing
        step - this is done here rather than in the sampling of the ruptures
        themselves
        """
        branch_key = self.idx_set["grid_key"]
        with h5py.File(self.source_file, 'r') as hdf5:
            bg_locations = hdf5["Grid/Locations"][()]
            if hasattr(self, 'src_filter'):
                # in event based
                idist = self.src_filter.integration_distance(DEFAULT_TRT)
            else:
                # in classical
                return range(len(bg_locations))
            distances = min_geodetic_distance(
                self.src_filter.sitecol.xyz,
                (bg_locations[:, 0], bg_locations[:, 1]))
            # Add buffer equal to half of length of median area from Mmax
            mmax_areas = self.msr.get_median_area(
                hdf5["/".join(["Grid", branch_key, "MMax"])][()], 0.0)
            # for instance hdf5['Grid/FM0_0_MEANFS_MEANMSR/MMax']
            mmax_lengths = numpy.sqrt(mmax_areas / self.aspect)
            ok = distances <= (0.5 * mmax_lengths + idist)
            # get list of indices from array of booleans
            return numpy.where(ok)[0].tolist()

    def get_ucerf_rupture(self, iloc, h5):
        """
        :param iloc:
            Location of the rupture plane in the hdf5 file
        """
        trt = self.tectonic_region_type
        if hasattr(self, 'all_ridx'):  # already computed by the UcerfFilter
            ridx = self.all_ridx[iloc - self.start]
        else:
            ridx = self.get_ridx(h5, iloc)
        mag = self.mags[iloc - self.start]
        if mag < self.min_mag:
            return
        surface_set = []
        indices = self.src_filter.get_indices(self, ridx, mag)
        if len(indices) == 0:
            return
        for trace, plane in self.gen_trace_planes(ridx, h5):
            # build simple fault surface
            for jloc in range(0, plane.shape[2]):
                top_left = Point(
                    plane[0, 0, jloc], plane[0, 1, jloc], plane[0, 2, jloc])
                top_right = Point(
                    plane[1, 0, jloc], plane[1, 1, jloc], plane[1, 2, jloc])
                bottom_right = Point(
                    plane[2, 0, jloc], plane[2, 1, jloc], plane[2, 2, jloc])
                bottom_left = Point(
                    plane[3, 0, jloc], plane[3, 1, jloc], plane[3, 2, jloc])
                try:
                    surface_set.append(
                        ImperfectPlanarSurface.from_corner_points(
                            top_left, top_right, bottom_right, bottom_left))
                except ValueError as err:
                    raise ValueError(err, trace, top_left, top_right,
                                     bottom_right, bottom_left)

        rupture = ParametricProbabilisticRupture(
            mag, self.rake[iloc - self.start], trt,
            surface_set[len(surface_set) // 2].get_middle_point(),
            MultiSurface(surface_set), self.rate[iloc - self.start], self.tom)

        return rupture

    def iter_ruptures(self, **kwargs):
        """
        Yield ruptures for the current set of indices
        """
        with h5py.File(self.source_file, "r") as hdf5:
            for ridx in range(self.start, self.stop):
                if self.rate[ridx - self.start]:  # may have have zero rate
                    rup = self.get_ucerf_rupture(ridx, hdf5)
                    if rup:
                        yield rup

    # called upfront, before classical_split_filter
    def __iter__(self):
        if self.stop - self.start <= self.ruptures_per_block:  # already split
            yield self
            return
        for start in range(self.start, self.stop, self.ruptures_per_block):
            stop = min(start + self.ruptures_per_block, self.stop)
            new = copy.copy(self)
            new.id = self.id
            new.source_id = '%s:%d-%d' % (self.source_id, start, stop)
            new.start = start
            new.stop = stop
            yield new

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.source_id)

    def get_background_sources(self):
        """
        Turn the background model of a given branch into a set of point sources
        """
        background_sids = self.get_background_sids()
        with h5py.File(self.source_file, "r") as hdf5:
            grid_loc = "/".join(["Grid", self.idx_set["grid_key"]])
            # for instance Grid/FM0_0_MEANFS_MEANMSR_MeanRates
            mags = hdf5[grid_loc + "/Magnitude"][()]
            mmax = hdf5[grid_loc + "/MMax"][background_sids]
            rates = hdf5[grid_loc + "/RateArray"][background_sids, :]
            locations = hdf5["Grid/Locations"][background_sids, :]
            sources = []
            for i, bg_idx in enumerate(background_sids):
                src_id = "_".join([self.idx_set["grid_key"], str(bg_idx)])
                src_name = "|".join([self.idx_set["total_key"], str(bg_idx)])
                mag_idx = (self.min_mag <= mags) & (mags < mmax[i])
                src_mags = mags[mag_idx]
                src_mfd = EvenlyDiscretizedMFD(
                    src_mags[0],
                    src_mags[1] - src_mags[0],
                    rates[i, mag_idx].tolist())
                ps = PointSource(
                    src_id, src_name, self.tectonic_region_type, src_mfd,
                    self.mesh_spacing, self.msr, self.aspect, self.tom,
                    self.usd, self.lsd,
                    Point(locations[i, 0], locations[i, 1]),
                    self.npd, self.hdd)
                ps.checksum = zlib.adler32(pickle.dumps(vars(ps), protocol=4))
                ps._wkt = ps.wkt()
                ps.id = self.id
                ps.et_id = self.et_id
                ps.num_ruptures = ps.count_ruptures()
                ps.nsites = 1  # anything <> 0 goes
                sources.append(ps)
        return sources

    def get_one_rupture(self):
        raise ValueError('Unsupported option')

    def generate_event_set(self, background_sids, eff_num_ses):
        """
        Generates the event set corresponding to a particular branch
        """
        # get rates from file
        with h5py.File(self.source_file, 'r') as hdf5:
            occurrences = self.tom.sample_number_of_occurrences(
                self.rate * eff_num_ses, self.serial)
            indices, = numpy.where(occurrences)
            logging.debug(
                'Considering "%s", %d ruptures', self.source_id, len(indices))

            # get ruptures from the indices
            ruptures = []
            rupture_occ = []
            for iloc, n_occ in zip(indices, occurrences[indices]):
                ucerf_rup = self.get_ucerf_rupture(iloc, hdf5)
                if ucerf_rup:
                    ruptures.append(ucerf_rup)
                    rupture_occ.append(n_occ)

            # sample background sources
            background_ruptures, background_n_occ = sample_background_model(
                hdf5, self.idx_set["grid_key"], self.tom, eff_num_ses,
                self.serial, background_sids, self.min_mag, self.npd,
                self.hdd, self.usd, self.lsd, self.msr, self.aspect,
                self.tectonic_region_type)
            ruptures.extend(background_ruptures)
            rupture_occ.extend(background_n_occ)
        return ruptures, rupture_occ

    def _sample_ruptures(self, eff_num_ses):
        background_sids = self.get_background_sids()
        n_occ = AccumDict(accum=0)
        rups, occs = self.generate_event_set(background_sids, eff_num_ses)
        for rup, occ in zip(rups, occs):
            n_occ[rup] += occ
        yield from n_occ.items()


def sample_background_model(
        hdf5, branch_key, tom, eff_num_ses, seed, filter_idx, min_mag, npd,
        hdd, upper_seismogenic_depth, lower_seismogenic_depth, msr=WC1994(),
        aspect=1.5, trt=DEFAULT_TRT):
    """
    Generates a rupture set from a sample of the background model

    :param branch_key:
        Key to indicate the branch for selecting the background model
    :param tom:
        Temporal occurrence model as instance of :class:
        openquake.hazardlib.tom.TOM
    :param seed:
        Random seed to use in the call to tom.sample_number_of_occurrences
    :param filter_idx:
        Sites for consideration (can be None!)
    :param float min_mag:
        Minimim magnitude for consideration of background sources
    :param npd:
        Nodal plane distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param hdd:
        Hypocentral depth distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param float aspect:
        Aspect ratio
    :param float upper_seismogenic_depth:
        Upper seismogenic depth (km)
    :param float lower_seismogenic_depth:
        Lower seismogenic depth (km)
    :param msr:
        Magnitude scaling relation
    :param float integration_distance:
        Maximum distance from rupture to site for consideration
    """
    bg_magnitudes = hdf5["/".join(["Grid", branch_key, "Magnitude"])][()]
    # Select magnitudes above the minimum magnitudes
    mag_idx = bg_magnitudes >= min_mag
    mags = bg_magnitudes[mag_idx]
    rates = hdf5["/".join(["Grid", branch_key, "RateArray"])][filter_idx, :]
    rates = rates[:, mag_idx]
    valid_locs = hdf5["Grid/Locations"][filter_idx, :]
    # Sample remaining rates
    sampler = tom.sample_number_of_occurrences(rates * eff_num_ses, seed)
    background_ruptures = []
    background_n_occ = []
    for i, mag in enumerate(mags):
        rate_idx = numpy.where(sampler[:, i])[0]
        rate_cnt = sampler[rate_idx, i]
        occurrence = rates[rate_idx, i]
        locations = valid_locs[rate_idx, :]
        ruptures = generate_background_ruptures(
            tom, locations, occurrence,
            mag, npd, hdd, upper_seismogenic_depth,
            lower_seismogenic_depth, msr, aspect, trt)
        background_ruptures.extend(ruptures)
        background_n_occ.extend(rate_cnt.tolist())
    return background_ruptures, background_n_occ

# #################################################################### #


def build_idx_set(branch_id, start_date):
    """
    Builds a dictionary of keys based on the branch code
    """
    code_set = branch_id.split("/")
    code_set.insert(3, "Rates")
    idx_set = {
        "sec": "/".join([code_set[0], code_set[1], "Sections"]),
        "mag": "/".join([code_set[0], code_set[1], code_set[2], "Magnitude"])}
    idx_set["rate"] = "/".join(code_set)
    idx_set["rake"] = "/".join([code_set[0], code_set[1], "Rake"])
    idx_set["msr"] = "-".join(code_set[:3])
    idx_set["geol"] = code_set[0]
    if start_date:  # time-dependent source
        idx_set["grid_key"] = "_".join(
            branch_id.replace("/", "_").split("_")[:-1])
    else:  # time-independent source
        idx_set["grid_key"] = branch_id.replace("/", "_")
    idx_set["total_key"] = branch_id.replace("/", "|")
    return idx_set


def get_rupture_dimensions(mag, nodal_plane, msr, rupture_aspect_ratio,
                           upper_seismogenic_depth, lower_seismogenic_depth):
    """
    Calculate and return the rupture length and width
    for given magnitude ``mag`` and nodal plane.

    :param nodal_plane:
        Instance of :class:`openquake.hazardlib.geo.nodalplane.NodalPlane`.
    :returns:
        Tuple of two items: rupture length in width in km.

    The rupture area is calculated using method
    :meth:`~openquake.hazardlib.scalerel.base.BaseMSR.get_median_area`
    of source's
    magnitude-scaling relationship. In any case the returned
    dimensions multiplication is equal to that value. Than
    the area is decomposed to length and width with respect
    to source's rupture aspect ratio.

    If calculated rupture width being inclined by nodal plane's
    dip angle would not fit in between upper and lower seismogenic
    depth, the rupture width is shrunken to a maximum possible
    and rupture length is extended to preserve the same area.
    """
    area = msr.get_median_area(mag, nodal_plane.rake)
    rup_length = math.sqrt(area * rupture_aspect_ratio)
    rup_width = area / rup_length
    seismogenic_layer_width = (lower_seismogenic_depth -
                               upper_seismogenic_depth)
    max_width = (seismogenic_layer_width /
                 math.sin(math.radians(nodal_plane.dip)))
    if rup_width > max_width:
        rup_width = max_width
        rup_length = area / rup_width
    return rup_length, rup_width


def get_rupture_surface(mag, nodal_plane, hypocenter, msr,
                        rupture_aspect_ratio, upper_seismogenic_depth,
                        lower_seismogenic_depth, mesh_spacing=1.0):
    """
    Create and return rupture surface object with given properties.

    :param mag:
        Magnitude value, used to calculate rupture dimensions,
        see :meth:`_get_rupture_dimensions`.
    :param nodal_plane:
        Instance of :class:`openquake.hazardlib.geo.nodalplane.NodalPlane`
        describing the rupture orientation.
    :param hypocenter:
        Point representing rupture's hypocenter.
    :returns:
        Instance of
        :class:`~openquake.hazardlib.geo.surface.planar.PlanarSurface`.
    """
    assert (upper_seismogenic_depth <= hypocenter.depth
            and lower_seismogenic_depth >= hypocenter.depth)
    rdip = math.radians(nodal_plane.dip)

    # precalculated azimuth values for horizontal-only and vertical-only
    # moves from one point to another on the plane defined by strike
    # and dip:
    azimuth_right = nodal_plane.strike
    azimuth_down = (azimuth_right + 90) % 360
    azimuth_left = (azimuth_down + 90) % 360
    azimuth_up = (azimuth_left + 90) % 360

    rup_length, rup_width = get_rupture_dimensions(
        mag, nodal_plane, msr, rupture_aspect_ratio, upper_seismogenic_depth,
        lower_seismogenic_depth)
    # calculate the height of the rupture being projected
    # on the vertical plane:
    rup_proj_height = rup_width * math.sin(rdip)
    # and it's width being projected on the horizontal one:
    rup_proj_width = rup_width * math.cos(rdip)

    # half height of the vertical component of rupture width
    # is the vertical distance between the rupture geometrical
    # center and it's upper and lower borders:
    hheight = rup_proj_height / 2
    # calculate how much shallower the upper border of the rupture
    # is than the upper seismogenic depth:
    vshift = upper_seismogenic_depth - hypocenter.depth + hheight
    # if it is shallower (vshift > 0) than we need to move the rupture
    # by that value vertically.
    if vshift < 0:
        # the top edge is below upper seismogenic depth. now we need
        # to check that we do not cross the lower border.
        vshift = lower_seismogenic_depth - hypocenter.depth - hheight
        if vshift > 0:
            # the bottom edge of the rupture is above the lower sesmogenic
            # depth. that means that we don't need to move the rupture
            # as it fits inside seismogenic layer.
            vshift = 0
        # if vshift < 0 than we need to move the rupture up by that value.

    # now we need to find the position of rupture's geometrical center.
    # in any case the hypocenter point must lie on the surface, however
    # the rupture center might be off (below or above) along the dip.
    rupture_center = hypocenter
    if vshift != 0:
        # we need to move the rupture center to make the rupture fit
        # inside the seismogenic layer.
        hshift = abs(vshift / math.tan(rdip))
        rupture_center = rupture_center.point_at(
            horizontal_distance=hshift, vertical_increment=vshift,
            azimuth=(azimuth_up if vshift < 0 else azimuth_down))

    # from the rupture center we can now compute the coordinates of the
    # four coorners by moving along the diagonals of the plane. This seems
    # to be better then moving along the perimeter, because in this case
    # errors are accumulated that induce distorsions in the shape with
    # consequent raise of exceptions when creating PlanarSurface objects
    # theta is the angle between the diagonal of the surface projection
    # and the line passing through the rupture center and parallel to the
    # top and bottom edges. Theta is zero for vertical ruptures (because
    # rup_proj_width is zero)
    theta = math.degrees(
        math.atan((rup_proj_width / 2.) / (rup_length / 2.)))
    hor_dist = math.sqrt(
        (rup_length / 2.) ** 2 + (rup_proj_width / 2.) ** 2)
    left_top = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=-rup_proj_height / 2,
        azimuth=(nodal_plane.strike + 180 + theta) % 360)
    right_top = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=-rup_proj_height / 2,
        azimuth=(nodal_plane.strike - theta) % 360)
    left_bottom = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=rup_proj_height / 2,
        azimuth=(nodal_plane.strike + 180 - theta) % 360)
    right_bottom = rupture_center.point_at(
        horizontal_distance=hor_dist,
        vertical_increment=rup_proj_height / 2,
        azimuth=(nodal_plane.strike + theta) % 360)
    return PlanarSurface(nodal_plane.strike, nodal_plane.dip,
                         left_top, right_top, right_bottom, left_bottom)


def generate_background_ruptures(tom, locations, occurrence, mag, npd,
                                 hdd, upper_seismogenic_depth,
                                 lower_seismogenic_depth, msr=WC1994(),
                                 aspect=1.5, trt=DEFAULT_TRT):
    """
    :param tom:
        Temporal occurrence model as instance of :class:
        openquake.hazardlib.tom.TOM
    :param numpy.ndarray locations:
        Array of locations [Longitude, Latitude] of the point sources
    :param numpy.ndarray occurrence:
        Annual rates of occurrence
    :param float mag:
        Magnitude
    :param npd:
        Nodal plane distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param hdd:
        Hypocentral depth distribution as instance of :class:
        openquake.hazardlib.pmf.PMF
    :param float upper_seismogenic_depth:
        Upper seismogenic depth (km)
    :param float lower_seismogenic_depth:
        Lower seismogenic depth (km)
    :param msr:
        Magnitude scaling relation
    :param float aspect:
        Aspect ratio
    :param str trt:
        Tectonic region type
    :returns:
        List of ruptures
    """
    ruptures = []
    n_vals = len(locations)
    depths = hdd.sample_pairs(n_vals)
    nodal_planes = npd.sample_pairs(n_vals)
    for i, (x, y) in enumerate(locations):
        hypocentre = Point(x, y, depths[i][1])
        surface = get_rupture_surface(mag, nodal_planes[i][1],
                                      hypocentre, msr, aspect,
                                      upper_seismogenic_depth,
                                      lower_seismogenic_depth)
        rupture_probability = (occurrence[i] * nodal_planes[i][0] *
                               depths[i][0])
        ruptures.append(ParametricProbabilisticRupture(
            mag, nodal_planes[i][1].rake, trt, hypocentre, surface,
            rupture_probability, tom))
    return ruptures
