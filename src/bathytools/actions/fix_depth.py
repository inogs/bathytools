from typing import Callable

from bathytools.actions import MultipleChoiceAction


class FixDepth(MultipleChoiceAction):
    @classmethod
    def get_choices(cls) -> dict[str, Callable]:
        return {"slice": cls.fix_value_on_slice}

    @classmethod
    def get_choice_field(cls) -> str:
        return "where"

    @staticmethod
    def fix_value_on_slice(
        bathymetry, *, min_lat, max_lat, min_lon, max_lon, value
    ):
        bathymetry["elevation"].sel(
            latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
        ).values[:] = -value
        return bathymetry
