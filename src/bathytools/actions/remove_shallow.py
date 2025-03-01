import numpy as np
import xarray as xr

from bathytools.actions import SimpleAction


class RemoveShallow(SimpleAction):
    """
    Represents an action that removes all the cells shallower than a specified
    value.

    This class is designed to modify bathymetry data by setting to zero the
    depth values that are shallower than a specific threshold.

    Attributes:
        threshold (float): The threshold depth value. All the cells with a depth
        between this threshold and zero will be set to zero.
    """

    def __init__(self, name: str, description: str, threshold: float):
        super().__init__(name, description)
        self.threshold = float(threshold)

    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
        threshold_cells = np.logical_and(
            bathymetry.elevation < 0, bathymetry.elevation > -self.threshold
        )
        bathymetry["elevation"].values[threshold_cells] = 0.0
        return bathymetry
