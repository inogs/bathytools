from itertools import product as cart_product
from logging import getLogger
from warnings import warn

import numpy as np
import xarray as xr

from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix


LOGGER = getLogger(__name__)


class RemoveDeepTrappedWater(SimpleAction):
    """
    Ensures that the bathymetry values of the cells located on the boundary
    increase monotonically while moving towards the boundaries.

    This action is very similar to RemoveTrappedWater. The main difference is
    that RemoveTrappedWater removes water cells located on the boundary that
    are adjacent to land cells in the direction perpendicular to the boundary;
    if a cell on the boundary is a water cell, its value will never be
    modified.

    This action, instead, also removes the holes that are near the boundary
    and are under the sea level. In other words, it ensures that
    a cell on the boundary is always higher than its neighbor cells in
    the direction perpendicular to the boundary.

    Args:
        n_cells: The size of the boundary. If `n_cells` is 1, only the cells
            that are exactly on the boundary are modified. If `n_cells` is 2,
            we modify the cells that are exactly on the boundary and the ones
            that have a distance of 1 from the boundary, and so on.
    """

    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        n_cells: int = 2,
    ):
        super().__init__(name, description, output_appendix=output_appendix)
        self._n_cells = n_cells
        if self._n_cells < 0:
            raise ValueError(
                f"The number of cells to remove deep trapped water must be "
                f"greater than 0, but is {self._n_cells}."
            )
        if self._n_cells == 0:
            warn(
                f"Executing the {self.__class__.__name__} action with n_cells "
                f"set to 0 has no effect. Are you sure you want to do this?"
            )

    def __call__(self, bathymetry: xr.Dataset) -> xr.Dataset:
        if self._n_cells == 0:
            return bathymetry

        bathy_values = bathymetry.elevation.transpose(
            "latitude", "longitude"
        ).values
        min_axis = min(bathy_values.shape)
        min_cells_per_axis = (self._n_cells + 1) * 2
        if min_axis < min_cells_per_axis:
            if bathy_values.shape[0] < bathy_values.shape[1]:
                wrong_axis = "latitude"
            else:
                wrong_axis = "longitude"
            raise ValueError(
                f"The minimum number of cells on each axis required to remove "
                f"deep trapped water with n_cells set to {self._n_cells} is "
                f"{min_cells_per_axis}, but on the axis of the {wrong_axis} "
                f"there are only {min_axis} cells."
            )

        # We use the axis do determine whether we are iterating along
        # latitude (0) or longitude (1); we use side to decide on which side
        # of the boundary we are processing:
        # for latitude: axis = 0 is South, axis = -1 is North
        # for longitude: axis = 0 is West, axis = -1 is East.
        for axis, side in cart_product((0, 1), (0, -1)):
            LOGGER.debug(
                "Removing deep trapped water for axis %s on side %s",
                axis,
                side,
            )
            for cell in reversed(range(self._n_cells)):
                if side == 0:
                    current_index = cell
                    neighbor_index = current_index + 1
                else:
                    current_index = -1 - cell
                    neighbor_index = current_index - 1

                # Build a slice to select elements on the boundary we are
                # currently processing.
                slice_current = [slice(None), slice(None)]
                slice_current[axis] = current_index
                slice_current = tuple(slice_current)

                # Construct a slice to select elements from the neighboring row
                # or column.
                slice_near = [slice(None), slice(None)]
                slice_near[axis] = neighbor_index
                slice_near = tuple(slice_near)

                bathy_values[slice_current] = np.where(
                    bathy_values[slice_current] < bathy_values[slice_near],
                    bathy_values[slice_near],
                    bathy_values[slice_current],
                )

        return bathymetry
