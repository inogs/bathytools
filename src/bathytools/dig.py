from typing import List
from typing import Tuple

import numpy as np


DIG = List[Tuple[int, int]]


def main_river_cell_list(
    i_start: int, j_start: int, Segmentlist: List = ["20E", "10N"]
) -> DIG:
    """
    Draws the path of the river
    Arguments:
    i_start,j_start: start point
    Segmentlist:  list of segments in format "n_cells,direction"
               example: ["30E", "20S", "10W", "5N"]
    Returns a list of tuples(i,j) with the ordered positions of the river
    """
    L = [(i_start, j_start)]
    i = i_start
    j = j_start
    for segment in Segmentlist:
        length = int(segment[:-1])
        direction = segment[-1]
        for k in range(length):
            if direction == "E":
                i = i + 1
            if direction == "W":
                i = i - 1
            if direction == "N":
                j = j + 1
            if direction == "S":
                j = j - 1
            L.append((i, j))
    return L


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


def insert(
    i: int, j: int, L_orig: DIG, L: DIG, verbose=False
) -> Tuple[int, int, DIG]:
    """
    Inserts a new point (i,j) a list of positions of a new river
    by taking in account the segment we come up beside, in order to
    - never overlap it. Returns i = 0, as an exception for the caller.
    - don't proceed if we are already at the end of original river
    """
    if verbose:
        print("---", i, j)
    if (i, j) in L_orig:
        print("hai sbattuto sull'originale")
        return 0, 0, L
    hypothesis = [L[-1], (i, j)]
    I1, J1 = lateral_point(hypothesis, 1)
    I2, J2 = lateral_point(hypothesis, -1)
    if not ((L_orig[-1] == (I1, J1)) | (L_orig[-1] == (I2, J2))):
        L.append((i, j))
    if verbose:
        print(L)
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
    nHorCells: int, i_start: int, j_start: int, Segmentlist: List = ["20E,10N"]
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
    Segmentlist:  list of segments in format "n_cells,direction"

    Return:
    List of tuples (i,j)
    """
    if nHorCells not in range(1, 6):
        raise ValueError(f"nHorCells must be in range 1..5, got {nHorCells}")

    L_out = []
    for k in range(nHorCells):
        if k == 0:
            L = main_river_cell_list(i_start, j_start, Segmentlist)
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
