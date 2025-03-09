from typing import Callable

import numpy as np

from bathytools.actions import CellBroadcastAction


class SetDepth(CellBroadcastAction):
    """
    Modifies the values of the bathymetry dataset at a specified location by
    setting them to a fixed value. This action also allows specifying a range
    of depths to limit the modification to cells within that range. As a result,
    this action can be used to cap bathymetry values at a specific depth
    or exclude cells shallower than a given threshold.

    This action accepts the following arguments:
      - value (float): The fixed value used to replace the original values.
      - shallower_than (float, optional): The minimum depth to modify. Only
          cells with bathymetry values shallower than this will be changed.
          If not provided, all cells in the domain are subject to modification.
      - deeper_than (float, optional): The maximum depth to modify. Only
          cells with bathymetry values deeper than this will be changed.
          If not provided, all cells in the domain are subject to modification.
    """

    def build_callable(self, **kwargs) -> Callable:
        if "value" not in kwargs:
            raise ValueError(
                'Missing required argument "value" for class '
                f"{self.__class__.__name__}"
            )
        value = float(kwargs["value"])

        if "shallower_than" in kwargs:
            shallower_than = float(kwargs["shallower_than"])
        else:
            shallower_than = None

        if "deeper_than" in kwargs:
            deeper_than = float(kwargs["deeper_than"])
        else:
            deeper_than = None

        if shallower_than is not None and deeper_than is not None:

            def fix_value(x):
                x[:] = -value
                return x

            return fix_value

        def fix_value(x):
            if shallower_than is None:
                k1 = np.min(x) - 1
            else:
                k1 = -shallower_than

            if deeper_than is None:
                k2 = np.max(x) + 1
            else:
                k2 = -deeper_than

            return np.where(np.logical_and(x < 0, x <= k2, x >= k1), -value, x)

        return fix_value
