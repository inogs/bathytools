from typing import Callable

from bathytools.actions import CellBroadcastAction


class FixDepth(CellBroadcastAction):
    """
    Changes the values of the bathymetry dataset at a specified location to a
    fixed value.
    """

    def build_callable(self, **kwargs) -> Callable:
        if "value" not in kwargs:
            raise ValueError(
                'Missing required argument "value" for class '
                f"{self.__class__.__name__}"
            )
        value = float(kwargs["value"])

        def fix_value(x):
            x[:] = -value
            return x

        return fix_value
