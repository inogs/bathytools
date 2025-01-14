from typing import List
from typing import Tuple

import numpy as np
import pylab as pl


def main_river_cell_list(i: int, j: int, DIGlist: List) -> List:
    """Returns a list of tuples(i,j)"""
    L = [(i, j)]
    for segment in DIGlist:
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


def lateral_point(L, segno) -> Tuple:
    i0, j0 = L[0]
    i1, j1 = L[1]
    versor1 = (i1 - i0, j1 - j0, 0)
    vers_to_apply = segno * np.cross(versor1, (0, 0, 1))
    i_side = i0 + vers_to_apply[0]
    j_side = j0 + vers_to_apply[1]
    return i_side, j_side


def insert(i, j, L_orig, L):
    # print("---",i,j)
    if (i, j) in L_orig:
        print("hai sbattuto sull'originale")
        return 0, 0, L
    hypothesis = [L[-1], (i, j)]
    I1, J1 = lateral_point(hypothesis, 1)
    I2, J2 = lateral_point(hypothesis, -1)
    if not ((L_orig[-1] == (I1, J1)) | (L_orig[-1] == (I2, J2))):
        L.append((i, j))
    # print(L)
    return i, j, L


def cells_side(L: List, segno=1) -> List:
    """Arguments:
    L: list of tuples(i,j)
    Returns:
    L_trasv: list of tuples(i,j)
    """
    assert segno in [-1, 1]
    n = len(L)
    L_TRASV = []

    i_trasv, j_trasv = lateral_point(L, segno)
    L_TRASV.append((i_trasv, j_trasv))
    SKIP_LIST = []
    for k in range(1, n - 2):
        i0, j0 = L[k]
        i1, j1 = L[k + 1]
        i2, j2 = L[k + 2]
        versor1 = (i1 - i0, j1 - j0, 0)
        versor2 = (i2 - i1, j2 - j1, 0)
        CURVA_davanti = np.cross(versor1, versor2)
        if k in SKIP_LIST:
            continue
        if CURVA_davanti[2] == 0:
            i_trasv, j_trasv, L_TRASV = insert(
                i_trasv + versor1[0], j_trasv + versor1[1], L, L_TRASV
            )
            if i_trasv == 0:
                return L_TRASV
        if CURVA_davanti[2] == segno:
            # print("curva a sfavore",i0,j0)
            for _ in range(3 * abs(segno)):
                i_trasv, j_trasv, L_TRASV = insert(
                    i_trasv + versor1[0], j_trasv + versor1[1], L, L_TRASV
                )
                if i_trasv == 0:
                    return L_TRASV
        if CURVA_davanti[2] == -segno:
            # print("curva a favore")
            i_trasv, j_trasv, L_TRASV = insert(
                i_trasv + versor1[0], j_trasv + versor1[1], L, L_TRASV
            )
            if i_trasv == 0:
                return L_TRASV
            SKIP_LIST = [k + 1, k + 2]

    # last two points, straight ahead
    for k in range(2):
        i_trasv, j_trasv, L_TRASV = insert(
            i_trasv + versor1[0], j_trasv + versor1[1], L, L_TRASV
        )

    return L_TRASV


def apply_dig(A, L, v):
    n = len(L)
    for k in range(n):
        i, j = L[k]
        A[j, i] = v
    return A


def sequence_side(nHorCells, i, j, DIGlist):
    L_out = []
    for k in range(nHorCells):
        if k == 0:
            L = main_river_cell_list(i, j, DIGlist)
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
