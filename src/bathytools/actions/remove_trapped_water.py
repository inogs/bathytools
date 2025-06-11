from logging import getLogger

import numpy as np

from bathytools.actions import SimpleAction


LOGGER = getLogger(__name__)


class RemoveTrappedWater(SimpleAction):
    """
    Removes all water cells located on the boundary that come into contact
    with a land cell in the direction perpendicular to the boundary
    (e.g., north-south or east-west). Inside these cells, the fluxes imposed by
    the boundary conditions have no space to dissipate, and removing these
    cells prevents trapped water from destabilizing numerical simulations.
    """

    def __call__(self, bathymetry):
        water_cells = (
            bathymetry.elevation.transpose("latitude", "longitude").values < 0
        )
        # Determine whether we are iterating along latitude (0) or longitude (1)
        for axis in (0, 1):
            # Specify which side of the boundary is being processed:
            # for latitude: South (0) or North (-1),
            # for longitude: West (0) or East (-1).
            for side in (0, -1):
                LOGGER.debug(
                    "Removing trapped water for axis %s on side %s", axis, side
                )
                # Determine the neighbor row or column that is adjacent to the
                # specified boundary, identified by the variable `side`.
                side_near = 1 if side == 0 else -2

                # Build a slice to select elements on the boundary we are
                # currently processing.
                slice_current = [slice(None), slice(None)]
                slice_current[axis] = side
                slice_current = tuple(slice_current)

                # Construct a slice to select elements from the neighboring row
                # or column.
                slice_near = [slice(None), slice(None)]
                slice_near[axis] = side_near
                slice_near = tuple(slice_near)

                land_near = ~water_cells[slice_near]
                boundary_water = water_cells[slice_current]

                # Identify "trapped cells" - water cells that are adjacent to
                # land cells.
                trapped_cells = np.logical_and(land_near, boundary_water)
                n_trapped_cells = np.count_nonzero(trapped_cells)
                LOGGER.debug(
                    "Detected %s trapped water cells", n_trapped_cells
                )

                if n_trapped_cells == 0:
                    LOGGER.debug(
                        "No trapped water cells found; skipping operation."
                    )

                # Create a slice to target the trapped cells specifically.
                trapped_cells_slice = [slice(None), slice(None)]
                trapped_cells_slice[axis] = side
                trapped_cells_slice[1 - axis] = trapped_cells
                trapped_cells_slice = tuple(trapped_cells_slice)

                # Set the bathymetry values of the trapped cells to 0 to remove
                # water from them.
                bathymetry["elevation"].transpose("latitude", "longitude")[
                    trapped_cells_slice
                ] = 0.0

        return bathymetry
