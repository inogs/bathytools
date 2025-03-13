from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable
from warnings import warn

import numpy as np
import xarray as xr
from bitsea.basins.region import Polygon

from bathytools.operations import Operation
from bathytools.utilities.relative_paths import read_path


LOGGER = logging.getLogger(__name__)


class Action(Operation, ABC):
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
    reason for performing the action. If the description is not provided, it
    will be set to an empty string.
    Moreover, another reserved field is `args`, which contains the
    arguments that must be passed to the action constructor. If not submitted,
    this will be considered as an empty dictionary.
    Other fields are action-specific.
    """

    @abstractmethod
    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
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
        return choice_func(self, bathymetry, **self._kwargs)

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
        if choice not in valid_choices:
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


class CellBroadcastAction(MultipleChoiceAction, ABC):
    """
    A common use case for a MultipleChoiceAction is applying the same operation
    to a subset of bathymetry data, cell by cell.

    In this scenario, each selected cell in the subset is processed independently.
    This class provides a convenient way to implement such algorithms, ensuring
    that the same function can be applied to each cell of the domain, a slice,
    or a polygon.

    To create a class that inherits from this one, the `build_callable` method
    must be implemented. This method returns a callable to be applied to each
    cell (i.e., the callable operates on a NumPy array and is expected to return
    a NumPy array with the same shape).

    The `build_callable` method is invoked with the arguments passed to the
    `__init__` method, excluding those in the `STANDARD_KEYS` tuple. These
    arguments are reserved for specifying the domain in which the callable
    is applied. Thus, they should not be used as argument names in your
    implementation of `build_callable`.

    The `build_callable` method can utilize all initialization values to create
    a callable that accepts a single argument: the bathymetry data values to
    be processed.

    The resulting action will apply the callable to all cells in the domain if
    the `where` field is set to "everywhere" or is not specified. If the `where`
    field is set to "slice", the `args` field must include the following keys:
    `min_lat`, `max_lat`, `min_lon`, `max_lon`, which define the slice's position.
    Any additional values in `args` are passed to the `build_callable` method.
    Similarly, if the `where` field is set to "polygon", the `args` field must
    contain the following keys: `polygon_name` and `wkt_file`. These specify
    the polygon to be used. Any other values in `args` are passed to
    the `build_callable` method.
    """

    STANDARD_KEYS = (
        "min_lat",
        "max_lat",
        "min_lon",
        "max_lon",
        "polygon_name",
        "wkt_file",
    )

    def __init__(self, name: str, description: str, choice: str, kwargs: dict):
        build_callable_kwargs = kwargs.copy()
        for standard_key in self.STANDARD_KEYS:
            if standard_key in build_callable_kwargs:
                del build_callable_kwargs[standard_key]
        self._callable = self.build_callable(**build_callable_kwargs)

        kwargs_domain = {
            c: v for c, v in kwargs.items() if c in self.STANDARD_KEYS
        }
        if "wkt_file" in kwargs_domain:
            kwargs_domain["wkt_file"] = read_path(kwargs_domain["wkt_file"])

        super().__init__(name, description, choice, kwargs_domain)

    @abstractmethod
    def build_callable(
        self, **kwargs
    ) -> Callable[[np.ndarray | xr.DataArray], np.ndarray | xr.DataArray]:
        raise NotImplementedError

    @classmethod
    def get_choice_field(cls) -> str:
        return "where"

    @classmethod
    def get_choices(cls) -> dict[str, Callable]:
        return {
            "everywhere": cls.apply_everywhere,
            "slice": cls.apply_on_slice,
            "polygon": cls.apply_on_polygon,
        }

    @classmethod
    def default_choice(cls) -> str | None:
        return "everywhere"

    def apply_everywhere(self, bathymetry: xr.DataArray) -> xr.DataArray:
        bathymetry["elevation"].values = self._callable(
            bathymetry.elevation.values.copy()
        )
        return bathymetry

    def apply_on_slice(
        self,
        bathymetry: xr.DataArray,
        *,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> xr.DataArray:
        data = (
            bathymetry["elevation"]
            .sel(
                latitude=slice(min_lat, max_lat),
                longitude=slice(min_lon, max_lon),
            )
            .values.copy()
        )
        bathymetry["elevation"].sel(
            latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
        )[:] = self._callable(data)
        return bathymetry

    def apply_on_polygon(
        self, bathymetry: xr.DataArray, *, polygon_name: str, wkt_file: Path
    ):
        with open(wkt_file, "r") as f:
            available_polys = Polygon.read_WKT_file(f)

        try:
            poly = available_polys[polygon_name]
        except KeyError as e:
            available_polys_str = ('"' + pl + '"' for pl in available_polys)
            error_message = (
                f'Polygon "{polygon_name}" not found in {wkt_file}; available '
                f"choices: {', '.join(available_polys_str)}"
            )
            raise KeyError(error_message) from e

        is_inside = poly.is_inside(
            lon=bathymetry.longitude.values,
            lat=bathymetry.latitude.values[:, np.newaxis],
        )
        data = (
            bathymetry["elevation"]
            .transpose("latitude", "longitude")
            .values[is_inside]
            .copy()
        )
        bathymetry["elevation"].transpose("latitude", "longitude").values[
            is_inside
        ] = self._callable(data)
        return bathymetry
