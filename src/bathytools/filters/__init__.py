from abc import ABC
from abc import abstractmethod

from bathytools.domain_discretization import DomainDiscretization
from bathytools.operations import Operation


class Filter(Operation, ABC):
    """
    The `Filter` class represents an operation applied to a
    `DomainDiscretization` object.

    This class takes a `DomainDiscretization` object as input and
    returns a modified `DomainDiscretization` object. This happens inside its
    `__call__` method.

    Similar to `Actions`, all subclasses of `Filter` must be initialized
    using a dictionary, referred to as an `initialization dictionary`.
    The `initialization dictionary` must contain a mandatory field, `name`,
    which specifies the name of the filter (i.e., the class name).
    Optionally, the dictionary can include a `description` field, which
    provides a human-readable explanation of the purpose of the filter.
    If no description is provided, it defaults to an empty string.

    Another reserved field in the dictionary is `args`, which holds arguments
    to pass to the filter constructor. If not provided, `args` defaults to an
    empty dictionary. Any additional fields in the dictionary are specific to
    the particular filter being implemented.
    """

    @abstractmethod
    def __call__(
        self, domain_discretization: DomainDiscretization
    ) -> DomainDiscretization:
        raise NotImplementedError
