import logging

import numpy as np


class DepthLevels:
    """
    Represents the vertical depth levels within a computational model.

    The `DepthLevels` class models the thickness of vertical layers (in meters)
    and facilitates computations related to the positions of cell faces and
    centers within a domain or grid.

    The objects of this class are initialized from a 1D NumPy array containing
    the thickness of each level. All the other attributes are computed based
    on those values.
    """

    def __init__(self, levels: np.ndarray):
        self._levels = np.asarray(levels)

        if len(self._levels.shape) != 1:
            raise ValueError(
                f"`levels` must be a 1D array; its current shape is {self._levels.shape}"
            )

        face_positions = np.zeros(
            (self._levels.shape[0] + 1,), dtype=levels.dtype
        )
        np.cumsum(self._levels, out=face_positions[1:])

        self._faces = face_positions

        self._levels.setflags(write=False)
        self._faces.setflags(write=False)

    def __repr__(self):
        return f"DepthLevels(levels={repr(self._levels)})"

    def __str__(self):
        return str(self._levels)

    @property
    def thickness(self):
        return self._levels

    @property
    def centers(self):
        return (self._faces[:-1] + self._faces[1:]) / 2.0

    @property
    def bottom_faces(self):
        return self._faces[1:]

    @property
    def top_faces(self):
        return self._faces[:-1]

    @property
    def face_positions(self):
        return self._faces


def generate_level_heights(
    first_layer_height: float = 1.0,
    max_depth: float = 218.0,
    extra_refined_depth: float = 20.0,
) -> DepthLevels:
    """Generate  the height of the levels for the  z-axis of the model.

    The function is a Python reimplementation of the one found in the
    `domzgr.F90` [1] file of the NEMO model. Variable names have been kept
    the same as in the original code to maintain consistency and facilitate
    comparison between the two implementations.

    This function requires to specify the height of the first layer and the
    maximum depth of the domain. It implements an algorithm that propagate
    the size of the first layer, increasing it logarithmically, until the
    maximum depth is reached.

    If a higher resolution is needed on the uppermost part of the domain,
    it is possible to use the `extra_refined_depth` parameter. The first
    `extra_refined_depth` meters of the model will have layers of the same
    height of the first one.

    Args:
        first_layer_height (float): the first layer's thickness
        max_depth (float): maximum depth of the current domain
        extra_refined_depth (float): thickness of extra slab to be stacked
            on top of the depth-increasing layers, allowing high resolution
            in near the surface

    Returns:
        levels: one dimensional array containing the thickness of all the
            vertical domain layers; these increase with depth, allowing for
            higher resolution near the surface

    Raises:
        ValueError: when `first_layer_height` and/or `max_depth` are not
        strictly positive


    NOTE: Conversion table between legacy and new variable names

        first_layer_height = ppdzmin
        MEDSEA_MAX_DEPTH = PPHMAX

    [1] https://forge.nemo-ocean.eu/nemo/nemo/-/blob/789f105d635d17e0e1fd41a002d44c8fc5b045d6/src/OCE/DOM/domzgr.F90
    """
    logger = logging.getLogger(f"{__name__}:generate_level_heights")
    if (first_layer_height <= 0.0) or (max_depth <= 0.0):
        raise ValueError(
            "`first_layer_height` and `max_depth` must be strictly positive."
        )
    if extra_refined_depth < 0.0:
        raise ValueError("`extra_refined_depth` must be non-negative.")
    if extra_refined_depth > max_depth:
        raise ValueError(
            "`extra_refined_depth` must be smaller than `max_depth`"
        )

    # same parameters from the NEMO Fortran functions
    PPKTH = 111.0
    PPACR = 64.0
    # max depth of the Mediterranean Sea
    MEDSEA_MAX_DEPTH = 5700.0

    # number of z-levels in the NEMO model
    JPK = 141
    JPKM1 = JPK - 1

    ZKTH = PPKTH
    ZACR = PPACR

    # first_layer_height is the height of the first layer. In the original grid
    # used by the CMS physics products this value is set to 2. In the MER project
    # usually we want a finer grid, so for example a typical value could be 1.
    za1 = (first_layer_height - MEDSEA_MAX_DEPTH / JPKM1) / (
        np.tanh((1 - PPKTH) / PPACR)
        - PPACR
        / JPKM1
        * np.log(np.cosh((JPK - PPKTH) / PPACR) / np.cosh((1 - PPKTH) / PPACR))
    )
    za0 = first_layer_height - za1 * np.tanh((1 - PPKTH) / PPACR)
    zsur = -za0 - za1 * PPACR * np.log(np.cosh((1 - PPKTH) / PPACR))

    _zw = np.arange(1, JPK - 1)
    gdepw_1d = (
        zsur + za0 * _zw + za1 * ZACR * np.log(np.cosh((_zw - ZKTH) / ZACR))
    )

    # logger.debug("NEMO-like grid: %s", gdepw_1d)

    # At this point we have a nonlinear grid with the first layer at 0 m.
    # We add on top `n_extra_levels` equally spaced levels and shift the
    # nonlinear layers down accordingly.
    n_extra_levels = round(extra_refined_depth / first_layer_height)
    extra_depth = n_extra_levels * first_layer_height
    stop_depth = max_depth - extra_depth

    stop_lvl = next(i for i, d in enumerate(gdepw_1d) if d > stop_depth)
    logger.debug("Last index = %s", stop_lvl)
    logger.debug(
        "Last NEMO-like depth (+%.0f m) = %.4f m",
        extra_depth,
        gdepw_1d[stop_lvl] + extra_depth,
    )
    logger.debug("Last old CADEAU depth = %.4f m", max_depth)

    # stack together equally spaced and nonlinear grids, taking care of not
    # double counting the level at extra_depth.
    mixed_zb = np.hstack(
        [
            np.arange(n_extra_levels) * first_layer_height,
            extra_depth + gdepw_1d[: stop_lvl + 1],
        ]
    )
    del_z = np.diff(mixed_zb)

    # logger.debug("Mixed grid borders: %s", mixed_zb)
    # logger.debug("Mixed grid \u0394z: %s", del_z)
    logger.debug("Number of cells = %s", len(del_z))

    return DepthLevels(del_z)
