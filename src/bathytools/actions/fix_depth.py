from typing import Callable

import xarray as xr

from bathytools.actions import MultipleChoiceAction


class FixDepth(MultipleChoiceAction):
    """
    Changes the values of the bathymetry dataset at a specified location to a
    fixed value.
    """

    @classmethod
    def get_choices(cls) -> dict[str, Callable]:
        return {"slice": cls.fix_value_on_slice}

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
