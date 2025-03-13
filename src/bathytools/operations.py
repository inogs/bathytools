from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path


LOGGER = logging.getLogger(__name__)


class Operation(ABC):
    """
    The `Operation` class represents an optional function applied during the
    execution of the algorithm that generates the bathymetry mask.

    Operations are categorized into two main groups: `Actions` and `Filters`.
    `Actions` are applied to the bathymetry data prior to its discretization
    across the domain, while `Filters` modify the values of individual cells
    on the already discretized domain.

    This interface serves as a framework for converting a section of the YAML
    configuration file into an `Operation`, which could be either an `Action`
    or a `Filter`. To enable this, every `Operation` subclass must implement a
    `from_dict` method that accepts a dictionary containing the relevant
    configuration options specified in the YAML file.

    The `build` class method interprets the dictionary, determines the
    appropriate `Operation` subclass, and invokes its `from_dict` method for
    instantiation. To identify the relevant subclass, the `get_subclasses`
    method scans for all concrete (non-abstract) implementations within the
    module.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @classmethod
    def get_subclasses(cls) -> dict[str, type]:
        """
        Retrieves all subclasses of the given class, including both direct and
        indirect subclasses.

        This method dynamically imports all Python scripts located in the
        directory of the class definition and identifies subclasses of the
        provided class. Only concrete (non-abstract) subclasses found in those
        scripts are included in the result.

        Returns:
            A dictionary mapping subclass names to their corresponding class
            types.
        """
        # Identify the module where the current class is defined.
        cls_module = inspect.getmodule(cls)
        # Resolve the absolute path of the module file.
        cls_module_path = Path(cls_module.__file__).resolve()

        # Define the directory to search for additional scripts. It is the one
        # where the module is saved.
        script_dir = cls_module_path.parent

        # Determine the module import path based on the location of the
        # current class.
        # If the class is in an __init__.py file or a module without a nested
        # namespace (no dot in the name), subclasses are assumed to be in a
        # more internal module. Otherwise, search at the same directory level.
        if (
            cls_module_path.name == "__init__.py"
            or "." not in cls_module_path.name
        ):
            base_module_name = cls_module.__name__
        else:
            base_module_name = ".".join(cls_module.__name__.split(".")[:-1])

        # Initialize a dictionary to store identified subclasses.
        subclasses = {}

        for file_path in script_dir.iterdir():
            f_name = file_path.name
            if f_name.lower().endswith(".py") and not f_name.startswith("_"):
                module_name = f"{base_module_name}.{f_name[:-3]}"
                LOGGER.debug("Importing module %s", module_name)
                current_module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(
                    current_module, inspect.isclass
                ):
                    if issubclass(obj, cls) and not inspect.isabstract(obj):
                        subclasses[name] = obj
                        LOGGER.debug("Found class %s", name)

        return subclasses

    @classmethod
    def build(cls, init_dict: dict, subclasses: dict[str, type] | None = None):
        """
        Instantiates a subclass of `Operation` based on the `name` field
        provided in the input dictionary, and initializes it using the
        subclass's `from_dict` method.

        The `init_dict` should contain the following keys:
          - `name` (required): A string indicating the name of the subclass
             of `Operation` to instantiate.
          - `description` (optional): A human-readable description of the
            operation. Defaults to an empty string if not provided.
          - `args` (optional): A dictionary of arguments required for the
            subclass initialization. Defaults to an empty dictionary if not
            provided.

        This method identifies the appropriate subclass of `Operation` by name
        and uses its `from_dict` method to create an instance.

        Args:
            init_dict (dict): Dictionary containing initialization data for
                the operation.
            subclasses (dict, optional): A dictionary mapping subclass names
                to their types. If not provided, the method will automatically
                retrieve subclasses using `get_subclasses`.

        Returns:
            Operation: An instance of the appropriate `Operation` subclass,
            initialized with the provided data.
        """

        if subclasses is None:
            subclasses = cls.get_subclasses()
        operation = subclasses[init_dict["name"]]

        if "description" not in init_dict:
            init_dict["description"] = ""

        if "args" not in init_dict:
            init_dict["args"] = {}

        # noinspection PyUnresolvedReferences
        return operation.from_dict(init_dict)

    @classmethod
    @abstractmethod
    def from_dict(cls, init_dict: dict):
        """
        Defines how each `Operation` subclass should be initialized from a
        dictionary, typically loaded from a configuration file.

        Each subclass must implement this method to handle its specific
        initialization logic, validating and extracting the fields required
        for creating its instance.

        Args:
            init_dict (dict): A dictionary containing the necessary fields
                for initializing the operation.

        Returns:
            Operation: An instance of the subclass initialized with the
                provided dictionary data.
        """
        raise NotImplementedError
