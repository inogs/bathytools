from logging import getLogger
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bathytools.domain_discretization import DomainDiscretization
from bathytools.filters import Filter
from bathytools.output_appendix import OutputAppendix


LOGGER = getLogger(__name__)


class RemoveStagnations(Filter):
    """
    Removes "stagnations" from the domain.

    A "stagnation" refers to a vertical column of water cells that is deeper
    than any neighboring columns and is entirely surrounded by land cells
    at the bottom. As a result, water in this column becomes trapped,
    with the only potential interaction occurring through vertical diffusion.
    This can adversely affect the accuracy and stability of numerical
    simulations.

    To address this issue, the filter identifies all stagnation columns within
    the bathymetry. For each stagnation, the depth of its water cells is
    replaced with the depth of the deepest water cell found in the neighboring
    columns.
    This adjustment ensures smooth water depth transitions and prevents isolated
    water volumes from distorting simulation results.
    """

    def __call__(
        self, domain_discretization: DomainDiscretization
    ) -> DomainDiscretization:
        LOGGER.debug('Starting "RemoveStagnation" filter')

        mesh_mask = domain_discretization.build_mesh_mask(
            output_appendix=OutputAppendix(Path(".")),
        )
        cell_mask = mesh_mask.tmask.values[0, :, :, :]

        LOGGER.debug("Computing the number of cells per column")
        cells_per_column = np.count_nonzero(cell_mask, axis=0)

        assert np.all(cells_per_column >= 0)

        # Create a new array that has two more axis;
        # windowed_cells_per_column[i, j, :, :] is a 3x3 square that contains
        # the same values of cells_per_column[i - 1: i + 1, j - 1, j + 1]
        windowed_cells_per_column = sliding_window_view(
            cells_per_column, (3, 3), writeable=False
        )

        # Create a 3x3 mask that excludes the center and the diagonals
        cross_neighbours = np.zeros((3, 3), dtype=bool)
        cross_neighbours[:, 1] = True
        cross_neighbours[1, :] = True
        cross_neighbours[1, 1] = False

        # For every column, here we save the number of cell of the deepest
        # nearby column
        neighbour_max_columns = np.max(
            windowed_cells_per_column,
            axis=(-2, -1),
            where=cross_neighbours,
            initial=0,
        )
        # This mask is `True` where we have a stagnation. There can not be
        # stagnations on the boundaries
        stagnation_mask = np.zeros(shape=cells_per_column.shape, dtype=bool)
        stagnation_mask[1:-1, 1:-1] = (
            neighbour_max_columns < cells_per_column[1:-1, 1:-1]
        )

        n_stagnations = int(np.sum(stagnation_mask))
        LOGGER.debug(f"Found {n_stagnations} stagnations")

        if n_stagnations == 0:
            LOGGER.debug("No stagnations found; returning the original domain")
            return domain_discretization

        # This is the indices of the stagnation columns. We remove 1 from the
        # indices later to have also indices that work when using the
        # slicing_window_view (that removes the values on the boundaries)
        stagnation_indices = np.nonzero(stagnation_mask)
        windowed_indices = tuple(k - 1 for k in stagnation_indices)

        # This is the windowed representation of the actual bathymetry
        windowed_depth = sliding_window_view(
            domain_discretization.bathymetry,
            window_shape=(3, 3),
            writeable=False,
        )
        # And here we compute the new depth for the points
        stagnation_new_depth = np.max(
            windowed_depth[windowed_indices],
            axis=(-2, -1),
            where=cross_neighbours,
            initial=0.0,
        )

        # Copy the values inside the new bathymetry. We must take into account
        # the change of sign. The "elevation" variable of the Xarray Dataset
        # has negative values, while the numpy array are positive.
        new_bathymetry = domain_discretization.original_bathymetry.copy()
        new_bathymetry_data = new_bathymetry.elevation.transpose(
            "latitude", "longitude"
        ).values
        new_bathymetry_data[:] = -domain_discretization.bathymetry
        new_bathymetry_data[stagnation_indices] = -stagnation_new_depth

        return DomainDiscretization.build(
            new_bathymetry,
            domain_geometry=domain_discretization.geo_arrays.domain_geometry,
        )

    @classmethod
    def from_dict(cls, init_dict: dict, output_appendix: OutputAppendix):
        name = init_dict["name"]
        description = init_dict.get("description", "")

        if len(init_dict.get("args", {})) > 0:
            raise ValueError(
                'No "args" are accepted for filter "RemoveStagnation"'
            )

        for key in init_dict:
            if key not in ["name", "description", "args"]:
                raise ValueError(
                    f'Invalid argument "{key}" in filter "{name}"'
                )

        # noinspection PyArgumentList
        return cls(
            name=name, description=description, output_appendix=output_appendix
        )
