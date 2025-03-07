from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable
from warnings import warn

import xarray as xr


if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


class Action(ABC):
    """
    The `Action` class represents an operation performed on bathymetry data
    prior to its discretization into a mask.

    The primary functionality of an `Action` is defined by the `__call__`
    method, which modifies the input bathymetry and returns the result.
    From a programmatic perspective, bathymetry data is represented as a
    `DataArray` with the following structure:
      - Two dimensions: latitude and longitude.
      - A single scalar dimension for elevation.

    The elevation values are defined as 0 for land and negative for ocean.

    This is an abstract base class providing a general interface for actions
    and a mechanism to locate specific subclasses by their names.

    A mandatory requirement for all subclasses is that there must be possible
    to initialize them from a dictionary. We will refer to such dictionaries
    as `initialization dictionaries`. The only mandatory field in such a
    dictionary is the `name` field, which specifies the name of the action
    (the name of the class). Even if it is not mandatory, another special
    field is `description`, which contains a human-readable description of the
    reason for performing the action. If the description is not provided, itù
    will be set to an empty string.
    Moreover, another reserved field is `args`, which contains the
    arguments that must be passed to the action constructor. If not submitted,
    this will be considered as an empty dictionary.
    Other fields are action-specific.
    """

    # A dictionary that associates the name of a subclass of Actions to its
    # implementation. This dictionary is filled by the `get_subclasses()`
    # method
    _SUBCLASSES: dict[str, type] | None = None

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, init_dict: dict):
        """
        Each action class must support initialization from a dictionary,
        which is typically loaded from a configuration file. This method
        defines the blueprint for creating an object of the class using such
        a dictionary. The dictionary must contain the required fields
        specific to each action type.
        """
        raise NotImplementedError

    @staticmethod
    def _check_args_dict(args_dict: dict):
        """
        Ensure that the args dictionary does not contain any reserved field.
        """
        if "name" in args_dict:
            raise ValueError(
                'A "name" argument cannot be specified in the args of an action'
            )
        if "description" in args_dict:
            raise ValueError(
                'A "description" argument cannot be specified in the args of an action'
            )

    @staticmethod
    def build(init_dict: dict) -> Action:
        """
        Creates an object of a subclass of `Action` based on the `name`
        specified in the provided dictionary and initializes it using the
        subclass's `from_dict` method.

        The `init_dict` should have the following structure:
          - `name` (required): A string representing the name of the
            target subclass of `Action`.
          - `description` (optional): A human-readable description of the
            action. Defaults to an empty string if not provided.
          - `args` (optional): A dictionary containing any specific
            arguments required for initializing the action. Defaults to an
            empty dictionary if not provided.

        The method internally resolves the appropriate subclass. This subclass
        is expected to implement the `from_dict` method, which is used to create
        and return an instance of the class.

        Args:
            init_dict: The dictionary containing the initialization data for
                the action.

        Returns:
            An object of the subclass of `Action` initialized based on the
            provided dictionary.
        """

        if Action._SUBCLASSES is None:
            Action._SUBCLASSES = Action.get_subclasses()
        action = Action._SUBCLASSES[init_dict["name"]]

        # noinspection PyUnresolvedReferences
        return action.from_dict(init_dict)

    @staticmethod
    def get_subclasses() -> dict[str, type]:
        """
        Returns all the subclasses of Action. This includes both the direct
        subclasses of Action and their subclasses.

        This method also imports all the Python scripts inside the 'actions'
        directory, to ensure all the classes are available.
        """
        actions_dir = Path(__file__).parent

        action_classes = {}

        for file_path in actions_dir.iterdir():
            f_name = file_path.name
            if f_name.lower().endswith(".py") and not f_name.startswith("_"):
                module_name = f"bathytools.actions.{f_name[:-3]}"
                LOGGER.debug("Importing module %s", module_name)
                current_module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(
                    current_module, inspect.isclass
                ):
                    if issubclass(obj, Action) and not inspect.isabstract(obj):
                        action_classes[name] = obj
                        LOGGER.debug("Found class %s", name)

        return action_classes


class SimpleAction(Action, ABC):
    """
    A `SimpleAction` represents an action that does not require additional
    arguments apart from the basic fields `name`, `description` and `args` in
    its initialization dictionary. Its `from_dict` method directly calls the
    constructor of the base class, passing the fields from the args
    dictionary as arguments, along with the name and the description.
    """

    @classmethod
    def from_dict(cls, init_dict: dict):
        if "args" not in init_dict:
            init_dict["args"] = {}
        name = init_dict["name"]
        description = init_dict.get("description", "")

        Action._check_args_dict(init_dict["args"])

        for key in init_dict:
            if key not in ["name", "description", "args"]:
                raise ValueError(
                    f'Invalid argument "{key}" in action "{name}"'
                )

        # noinspection PyArgumentList
        return cls(name=name, description=description, **init_dict["args"])


class MultipleChoiceAction(Action, ABC):
    """
    The `MultipleChoiceAction` class is an extension of the `Action` class that
    allows for multiple operations, with the specific operation determined by an
    additional field in its initialization dictionary. The intended use case
    is to associate several different Python functions to the same action name.

    A `MultipleChoiceAction` expects the initialization dictionary to have four
    fields:
      - `name` (required): The name of the action, identifying its class.
      - `description` (optional): A human-readable description for the action.
            Defaults to an empty string.
      - `args` (optional): A dictionary containing arguments specific to the
            operation being performed. Defaults to an empty dictionary.
      - `<choice_field>` (required): The name of this field is defined by the
            `get_choice_field` method. The value of this field determines the
            operation to be performed by the action.

    The `get_choice_field` method specifies the name of the additional field
    required in the initialization dictionary.
    The `get_choices` method returns a mapping of values for the
    `<choice_field>` to the corresponding functions implementing the operations.
    In other words, the allowed values for the `<choice_field>` in the
    initialization dictionary are the keys of the mapping returned by the
    `get_choices` method.
    Each value in this mapping, instead, represents a possible function that
    can be called by this action.
    When the `__call__` method of this class is invoked, the class resolves the
    function to execute based on the value of the `<choice_field>` and executes
    it. The resolved function receives the bathymetry data and the arguments
    (`args`) specified in the initialization dictionary.
    """

    @classmethod
    @abstractmethod
    def get_choice_field(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_choices(cls) -> dict[str, Callable]:
        raise NotImplementedError

    @classmethod
    def default_choice(cls) -> str | None:
        return None

    def __init__(self, name: str, description: str, choice: str, kwargs: dict):
        super().__init__(name, description)
        self._choice = choice
        self._kwargs = kwargs

    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
        choice_func = self.get_choices()[self._choice]
        return choice_func(bathymetry, **self._kwargs)

    @classmethod
    def from_dict(cls, init_dict: dict):
        if "args" not in init_dict:
            init_dict["args"] = {}
        name = init_dict["name"]
        description = init_dict.get("description", "")

        choice_field = cls.get_choice_field()

        Action._check_args_dict(init_dict["args"])

        if choice_field in init_dict["args"]:
            raise ValueError(
                f'The "{choice_field}" argument cannot be specified in the '
                f'args of an action of type "{cls.__name__}", but it must '
                "be specified as a field in the action definition."
            )

        if choice_field not in init_dict:
            default_choice = cls.default_choice()
            if default_choice is None:
                raise ValueError(
                    f'The "{choice_field}" argument must be specified when '
                    f'defining an action of type "{cls.__name__}".'
                )
            choice = default_choice
        else:
            choice = init_dict[choice_field]

        valid_choices = sorted(tuple(cls.get_choices().keys()))
        if init_dict[choice_field] not in valid_choices:
            valid_choices_str = ", ".join([f'"{v}"' for v in valid_choices])
            raise ValueError(
                f'Invalid value "{choice}" received for field '
                f'"{choice_field}"; the value of this field must be one '
                f"among the followings: {valid_choices_str}"
            )

        for key in init_dict:
            if key not in ["name", "description", choice_field, "args"]:
                warn(
                    f'Unknown field "{key}" in action "{name}"; it will be '
                    f"ignored"
                )

        return cls(
            name=name,
            description=description,
            choice=choice,
            kwargs=init_dict["args"],
        )
