from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import Dict

import xarray as xr


class OutputAppendix:
    """
    The `OutputAppendix` class allows `Action` classes to save additional data
    during the execution of their `__call__` method, beyond altering the
    bathymetry.

    This class provides a clear interface for managing additional output by
    enabling actions to:
    1. Access the path to the output directory and create custom files there.
    2. Append metadata to the final `meshmask.nc` file via a dictionary.
    """

    def __init__(self, output_dir: PathLike):
        """
        Initialize the `OutputAppendix` object with the output directory.

        Args:
            output_dir: The path to the directory where output files will
                be stored.
        """
        self._output_dir = Path(output_dir)
        self._meshmask_metadata: Dict[str:Any] = {}
        self._additional_vars: Dict[str : xr.DataArray] = {}

    @property
    def output_dir(self) -> Path:
        """
        Get the path to the output directory.

        Returns:
            The directory where output files are stored.
        """
        return self._output_dir

    def add_meshmask_metadata(self, name: str, content: Any):
        """
        Add metadata to the `meshmask` file.

        Args:
            name: The key or name for the metadata to be added.
            content: The content associated with the metadata key.
        """
        self._meshmask_metadata[name] = content

    def add_additional_variable(self, name: str, content: xr.DataArray):
        """
        Adds an xarray DataArray to the meshmask file.

        Args:
            name: The name of the DataArray to be added to the meshmask file.
            content: The value of the DataArray to be added to the meshmask.
        """
        self._additional_vars[name] = content

    def get_additional_variable(self, name: str) -> xr.DataArray:
        """
        Retrieves an additional variable by its name.

        This method searches for a specific variable within the collection of
        additional variables. If the variable is found, it returns it;
        otherwise, an IndexError is raised.

        Args:
            name: The name of the variable to retrieve.

        Returns:
           The requested variable with the specified name.

        Raises:
            IndexError: If the variable with the specified name is not found.
        """
        if "name" in self._additional_vars:
            return self._additional_vars[name]
        else:
            raise IndexError(f"No variable named {name} found.")

    def get_n_additional_variables(self) -> int:
        """
        Gets the count of additional variables.

        This method calculates and returns the number of additional variables.
        """
        return len(self._additional_vars)

    def get_additional_variables(self) -> MappingProxyType[str, xr.DataArray]:
        """
        Returns all the DataArrays added to the meshmask file
        """
        return MappingProxyType(self._additional_vars)
