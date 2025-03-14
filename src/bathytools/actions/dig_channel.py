import logging
import warnings

import numpy as np
from bitsea.basins.region import Polygon

from bathytools.actions import SimpleAction
from bathytools.utilities.points import Point
from bathytools.utilities.points import Segment


LOGGER = logging.getLogger(__name__)


class DigChannel(SimpleAction):
    """
    The DigChannel action allows users to modify bathymetry data by
    creating a channel.

    A channel is defined using two points and a width (in meters). This action
    generates a channel between the two points and adjusts the bathymetry for
    all cells located within the channel. The channel is represented as a
    rectangle, where the start and end points are the midpoints of two
    opposite sides whose lengths match the specified width.

    The bathymetric values of the cells within the channel are updated
    linearly,transitioning from the depth value of the start point to that of
    the end point.
    """

    def __init__(
        self,
        name: str,
        description: str,
        start_point: dict[str, float],
        end_point: dict[str, float],
        width: float,
    ):
        super().__init__(name, description)

        self._start_point = Point(**start_point)
        self._end_point = Point(**end_point)
        self._width = float(width)

    def __call__(self, bathymetry):
        # Create a normalized vector representing the direction of the channel
        # from the start point to the end point (centerline of the channel).
        channel_direction = self._end_point - self._start_point
        channel_direction /= np.linalg.norm(channel_direction)

        # Create a vector perpendicular to the channel direction, which
        # represents the orientation of the channel's width across its
        # centerline.
        channel_width_direction = np.array(
            [-channel_direction[1], channel_direction[0]]
        )

        # Construct the segments representing the start and end of the channel.
        # These are centered on the start and end points and extended laterally
        # by half the channel width in both directions, perpendicular to the
        # channel direction.
        s1 = Segment.build_from_center_and_length(
            self._start_point, channel_width_direction, self._width
        )
        s2 = Segment.build_from_center_and_length(
            self._end_point, channel_width_direction, self._width
        )

        # Define a polygon that encloses the channel, using the start and end
        # segments to form its boundaries.
        poly_lon = [
            s1.start.lon,
            s1.end.lon,
            s2.end.lon,
            s2.start.lon,
            s1.start.lon,
        ]
        poly_lat = [
            s1.start.lat,
            s1.end.lat,
            s2.end.lat,
            s2.start.lat,
            s1.start.lat,
        ]
        p = Polygon(lon_list=poly_lon, lat_list=poly_lat)

        # Reshape and broadcast the latitude and longitude arrays to match the
        # shape of the bathymetry elevation array, ensuring compatibility for
        # spatial operations.
        domain_lats = bathymetry.latitude.values[:, np.newaxis]
        domain_lons = bathymetry.longitude.values[np.newaxis, :]
        domain_lats, domain_lons = np.broadcast_arrays(
            domain_lats, domain_lons
        )

        # Identify the bathymetry cells that fall within the channel polygon.
        cells_in_poly = p.is_inside(lon=domain_lons, lat=domain_lats)
        n_cells = np.count_nonzero(cells_in_poly)

        if n_cells == 0:
            warnings.warn(
                f"The action DigChannel from {self._start_point} to "
                f"{self._end_point} did not find any cell inside the channel. "
                "Ensure that you choose an appropriate width and the correct "
                "boundary points. The execution will continue ignoring this "
                "action."
            )
            return bathymetry

        LOGGER.debug("Digging %s cells for the channel", n_cells)

        # Determine the depth values at the start and end points of the channel
        # by approximating them to the nearest neighboring bathymetry cells.
        s_depth = bathymetry.elevation.sel(
            longitude=self._start_point.lon,
            latitude=self._start_point.lat,
            method="nearest",
        ).values
        e_depth = bathymetry.elevation.sel(
            longitude=self._end_point.lon,
            latitude=self._end_point.lat,
            method="nearest",
        ).values
        LOGGER.debug("Start depth: %s", s_depth)
        LOGGER.debug("End depth: %s", e_depth)

        # Extract coordinates (latitude and longitude) of the bathymetry cells
        # that are located inside the channel polygon.
        dig_lats = domain_lats[cells_in_poly]
        dig_lons = domain_lons[cells_in_poly]

        # Initialize an array to temporarily store the updated depth values for
        # the cells within the channel.
        new_depths = np.empty(
            shape=dig_lats.shape, dtype=bathymetry.elevation.dtype
        )

        main_segment = Segment(self._start_point, self._end_point)
        for i, (lat, lon) in enumerate(zip(dig_lats, dig_lons)):
            p = Point(lat=lat, lon=lon)
            _, t = main_segment.project(p)
            new_depths[i] = s_depth + (e_depth - s_depth) * t

        # Update the bathymetry elevation data with the newly computed depth
        # values for the channel cells.
        bathymetry["elevation"].transpose("latitude", "longitude").values[
            cells_in_poly
        ] = new_depths

        return bathymetry
