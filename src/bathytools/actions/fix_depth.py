from os import PathLike
from typing import Callable

import numpy as np
import xarray as xr
from bitsea.basins.region import Polygon

from bathytools.actions import MultipleChoiceAction


class FixDepth(MultipleChoiceAction):
    """
    Changes the values of the bathymetry dataset at a specified location to a
    fixed value.
    """

    @classmethod
    def get_choices(cls) -> dict[str, Callable]:
        return {
            "slice": cls.fix_value_on_slice,
            "polygon": cls.fix_value_on_polygon,
        }

    @classmethod
    def get_choice_field(cls) -> str:
        return "where"

    @staticmethod
    def fix_value_on_slice(
        bathymetry: xr.Dataset,
        *,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        value: float,
    ) -> xr.Dataset:
        """
        Fix the value of the bathymetry dataset on a rectangle whose corners
        are specified by the min_lat, max_lat, min_lon, max_lon coordinates.

        Args:
            bathymetry: the bathymetry that must be modified
            min_lat: the minimum latitude of the rectangle
            max_lat: the maximum latitude of the rectangle
            min_lon: the minimum longitude of the rectangle
            max_lon: the maximum longitude of the rectangle
            value: the value to be assigned to the bathymetry dataset in the
                specified rectangle
        """
        bathymetry["elevation"].sel(
            latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
        ).values[:] = -value
        return bathymetry

    @staticmethod
    def fix_value_on_polygon(
        bathymetry: xr.Dataset,
        *,
        polygon_name: str,
        value: float,
        wkt_file: PathLike,
    ):
        """
        Fix the value of the bathymetry dataset on a polygon defined in a
        csv file downloaded from Google My Maps.

        Args:
            bathymetry: the bathymetry that must be modified
            polygon_name: the name of the polygon to be used
            value: the value to be assigned to the bathymetry dataset in the
                specified Polygon. It is expected to be positive.
            wkt_file: the path to the csv file containing the polygons.
        """
        with open(wkt_file, "r") as f:
            available_polys = Polygon.read_WKT_file(f)

        try:
            poly = available_polys[polygon_name]
        except KeyError as e:
            available_polys_str = ('"' + pl + '"' for pl in available_polys)
            error_message = (
                f'Polygon "{polygon_name}" not found in {wkt_file}; available '
                f"choices: {', '.join(available_polys_str)}"
            )
            raise KeyError(error_message) from e

        is_inside = poly.is_inside(
            lon=bathymetry.longitude.values,
            lat=bathymetry.latitude.values[:, np.newaxis],
        )
        bathymetry["elevation"].values[is_inside] = -value
        return bathymetry
