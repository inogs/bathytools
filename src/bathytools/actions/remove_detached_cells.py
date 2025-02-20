from logging import getLogger

import numpy as np
from bitsea.components.component_mask import ComponentMask

from bathytools.actions import SimpleAction
from bathytools.utilities.logtools import count_common_cells


LOGGER = getLogger(__name__)


class RemoveDetachedCells(SimpleAction):
    """
    Removes all the cells that are not connected to the sea (like the lakes
    or parts of the ocean that are not connected to the main sea we are
    modeling). The sea is defined as the connected component with the highest
    number of cells.
    """

    def __call__(self, bathymetry):
        LOGGER.debug("Computing the connected component of the mask")
        water_cells = bathymetry.elevation.values < 0.0
        components = ComponentMask(water_cells)
        LOGGER.debug(
            "There are %s connected components", components.n_components
        )

        sea_cells = components.get_component(
            components.get_biggest_component()
        )
        outside_main_component = np.logical_not(sea_cells)

        bathymetry["elevation"].values[outside_main_component] = 0.0

        LOGGER.debug(
            "Removed %s detached cells",
            count_common_cells(outside_main_component, water_cells),
        )

        return bathymetry
