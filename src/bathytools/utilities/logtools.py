from typing import Callable

import numpy as np


def build_logging_lazy_function(name: str, f: Callable):
    """
    This function builds a lazy function that warps the f function. This new
    function can be used to log the results of f. Indeed, the
    return value of the f function is not evaluated until the lazy function
    is converted to a string. This means that the f function is only evaluated
    if the log is printed.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __str__(self):
        return str(f(*self._args, **self._kwargs))

    lazy_class = type(
        name, (object,), {"__init__": __init__, "__str__": __str__}
    )

    return lazy_class


LoggingNanMin = build_logging_lazy_function("LoggingNanMin", np.nanmin)
LoggingNanMax = build_logging_lazy_function("LoggingNanMax", np.nanmax)
