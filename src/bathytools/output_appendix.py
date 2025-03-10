from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict


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
