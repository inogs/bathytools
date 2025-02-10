import importlib
import inspect
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import xarray as xr


if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


class Action(ABC):
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
        raise NotImplementedError

    @staticmethod
    def read_description(init_dict: dict) -> str:
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
    @classmethod
    def from_dict(cls, init_dict: dict):
        if "args" not in init_dict:
            init_dict["args"] = {}
        name = init_dict["name"]
        description = init_dict.get("description", "")

        if "name" in init_dict["args"]:
            raise ValueError(
                'A "name" argument cannot be specified in the args of an action'
            )
        if "description" in init_dict["args"]:
            raise ValueError(
                'A "description" argument cannot be specified in the args of an action'
            )

        for key in init_dict:
            if key not in ["name", "description", "args"]:
                raise ValueError(
                    f'Invalid argument "{key}" in action "{name}"'
                )

        # noinspection PyArgumentList
        return cls(name=name, description=description, **init_dict["args"])
