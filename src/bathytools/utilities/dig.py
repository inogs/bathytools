import re
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import List
from typing import Tuple

import numpy as np


DIG = List[Tuple[int, int]]
LOGGER = getLogger(__name__)


class Direction(Enum):
    EAST = "E"
    WEST = "W"
    NORTH = "N"
    SOUTH = "S"

    def move_indices(self, i, j):
        if self == Direction.EAST:
            i = i + 1
        elif self == Direction.WEST:
            i = i - 1
        elif self == Direction.NORTH:
            j = j + 1
        elif self == Direction.SOUTH:
            j = j - 1
        else:
            raise ValueError(f"Invalid direction: {self}")
        return i, j

    def __neg__(self):
        if self == Direction.EAST:
            return Direction.WEST
        elif self == Direction.WEST:
            return Direction.EAST
        elif self == Direction.NORTH:
            return Direction.SOUTH
        elif self == Direction.SOUTH:
            return Direction.NORTH
        else:
            raise ValueError(f"Invalid direction: {self}")


@dataclass
class Movement:
    length: int
    direction: Direction

    @staticmethod
    def from_str(s: str):
        string_mask = re.compile(r"^(\d+)([EWNS])$")
        match = string_mask.match(s)
        if match is None:
            raise ValueError(f"Invalid movement string: {s}")
        return Movement(int(match.group(1)), Direction(match.group(2)))

    def from_zonal_meridional_description(s: str):
        string_mask = re.compile(r"^([+-]?\d+)([mMzZ])$")
        match = string_mask.match(s)
        if match is None:
            raise ValueError(f"Invalid movement string: {s}")
        value = int(match.group(1))
        if match.group(2) in ["m", "M"]:
            direction = Direction.NORTH
        elif match.group(2) in ["z", "Z"]:
            direction = Direction.WEST
        else:
            raise ValueError(f"Invalid direction: {match.group(2)}")

        if value < 0:
            direction = -direction
            value = -value

        return Movement(value, direction)


def main_river_cell_list(
    i_start: int, j_start: int, movement_list: List[Movement]
) -> DIG:
    """
    Draws the path of a river whose width is exactly one cell.

    Args:
        i_start: start point first coordinate
        j_start: start point second coordinate
        movement_list: list of movements

    Returns:
        a list of tuples(i,j) with the ordered positions of the river
    """
    output_indices = [(i_start, j_start)]
    i = i_start
    j = j_start
    for segment in movement_list:
        length = segment.length
        direction = segment.direction
        for k in range(length):
            i, j = direction.move_indices(i, j)
            output_indices.append((i, j))
    return output_indices


def lateral_point(L, segno: int = 1) -> Tuple:
    """
    For a first point of a given segment, finds the nearest lateral point
    - on the right, if segno=1, on the left otherwise
    Arguments:
    L: list of tuples(i,j) of positions
    segno: 1 or -1
    Returns: (i,j)
    """
    assert segno in [-1, 1]
    i0, j0 = L[0]
    i1, j1 = L[1]
    versor1 = (i1 - i0, j1 - j0, 0)
    vers_to_apply = segno * np.cross(versor1, (0, 0, 1))
    i_side = i0 + vers_to_apply[0]
    j_side = j0 + vers_to_apply[1]
    return i_side, j_side


def insert(i: int, j: int, L_orig: DIG, L: DIG) -> Tuple[int, int, DIG]:
    """
    Inserts a new point (i,j) a list of positions of a new river
    by taking in account the segment we come up beside, in order to
    - never overlap it. Returns i = 0, as an exception for the caller.
    - don't proceed if we are already at the end of original river
    """
    LOGGER.debug("--- %s %s", i, j)

    if (i, j) in L_orig:
        LOGGER.warning("hai sbattuto sull'originale")
        return 0, 0, L

    hypothesis = [L[-1], (i, j)]
    I1, J1 = lateral_point(hypothesis, 1)
    I2, J2 = lateral_point(hypothesis, -1)
    if not ((L_orig[-1] == (I1, J1)) | (L_orig[-1] == (I2, J2))):
        L.append((i, j))

    LOGGER.debug("Final output: %s", L)
    return i, j, L


def cells_side(L: DIG, segno: int = 1) -> DIG:
    """
    Draws a new path placed side by side with the original

    Arguments:
        L: list of tuples(i,j) of positions
        segno: if 1, the new path is on the right of the original
               if -1, on the left

    Returns:
        l_side: list of tuples(i,j)
    """
    assert segno in [-1, 1]
    n = len(L)
    l_side = []

    i_side, j_side = lateral_point(L, segno)
    l_side.append((i_side, j_side))
    skip_list = []
    for k in range(1, n - 2):
        i0, j0 = L[k]
        i1, j1 = L[k + 1]
        i2, j2 = L[k + 2]
        # v1 and v2 are the two unit vectors of the two curves
        v1 = (i1 - i0, j1 - j0, 0)
        v2 = (i2 - i1, j2 - j1, 0)
        CURVA_davanti = np.cross(v1, v2)
        if k in skip_list:
            continue
        if CURVA_davanti[2] == 0:
            i_side, j_side, l_side = insert(
                i_side + v1[0], j_side + v1[1], L, l_side
            )
            if i_side == 0:
                return l_side
        if CURVA_davanti[2] == segno:
            # print("curva a sfavore (esterna)",i0,j0)
            for _ in range(3 * abs(segno)):
                i_side, j_side, l_side = insert(
                    i_side + v1[0], j_side + v1[1], L, l_side
                )
                if i_side == 0:
                    return l_side
        if CURVA_davanti[2] == -segno:
            # print("curva a favore (interna)")
            i_side, j_side, l_side = insert(
                i_side + v1[0], j_side + v1[1], L, l_side
            )
            assert i_side != 0
            skip_list = [k + 1, k + 2]

    # last two points, straight ahead
    for k in range(2):
        i_side, j_side, l_side = insert(
            i_side + v1[0], j_side + v1[1], L, l_side
        )

    return l_side


def apply_dig(A, L: DIG, v: float):
    """
    Applies a constant value the bathymetry, on a list of positions
    Arguments:
    A : 2D ndarray, original bathymetry
    L: list of tuples(i,j) of positions
    Returns: 2D ndarray of the corrected bathymetry
    """
    n = len(L)
    for k in range(n):
        i, j = L[k]
        A[j, i] = v
    return A


def sequence_side(
    nHorCells: int, i_start: int, j_start: int, movements: List[Movement]
) -> DIG:
    """
    Draws the path of the river having width expressed in cells.
    The sequence is:
     - draw the main path
     - then a path on the right
     - then a path of the left
     and so on up to 5.

    Arguments:
    nHorCells: width in cells of the river, max=5
    i_start,j_start: start point
    Segmentlist:  list of movements

    Return:
    List of tuples (i,j)
    """
    if nHorCells not in range(1, 6):
        raise ValueError(f"nHorCells must be in range 1..5, got {nHorCells}")

    L_out = []
    for k in range(nHorCells):
        if k == 0:
            L = main_river_cell_list(i_start, j_start, movements)
            L_out.extend(L)
        if k == 1:
            L1 = cells_side(L, segno=1)
            L_out.extend(L1)
        if k == 2:
            L2 = cells_side(L, segno=-1)
            L_out.extend(L2)
        if k == 3:
            L3 = cells_side(L1, segno=1)
            L_out.extend(L3)
        if k == 4:
            L4 = cells_side(L2, segno=-1)
            L_out.extend(L4)
    return L_out


if __name__ == "__main__":
    import pylab as pl

    A = np.zeros((50, 50))

    # DIGlist=["5E","3S","10E","3N","3W","4N"]
    DIGlist = ["30E", "20S", "10W", "5N"]

    seedx, seedy = 10, 40
    v = 10
    nHcells = 3

    L = sequence_side(nHcells, seedx, seedy, DIGlist)

    # L = main_river_cell_list(seedx, seedy, DIGlist)
    # L1 = cells_side(L,segno=1)
    # L2 = cells_side(L,segno=-1)
    # L3 = cells_side(L1,segno=1)
    #
    A = apply_dig(A, L, v / 2)
    # A = apply_dig(A,L1,v)
    # A = apply_dig(A,L2,v*2)
    # A = apply_dig(A,L3,v*3)

    pl.close("all")
    fig, ax = pl.subplots()
    ax.imshow(A)
    ax.invert_yaxis()
    fig.show()
