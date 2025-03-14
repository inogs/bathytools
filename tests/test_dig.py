from typing import List

import numpy as np
from pytest import fixture

from bathytools.utilities.dig import apply_dig
from bathytools.utilities.dig import Direction
from bathytools.utilities.dig import Movement
from bathytools.utilities.dig import sequence_side


@fixture
def list_of_movements():
    return [
        Movement(30, Direction.EAST),
        Movement(20, Direction.SOUTH),
        Movement(10, Direction.WEST),
        Movement(5, Direction.NORTH),
    ]


def test_sequence_side_thin(list_of_movements: List[Movement]):
    # Compute the length of the curve; we start from one because we have for
    # sure the first cell
    curve_length = 1
    for movement in list_of_movements:
        curve_length += movement.length

    seed_x, seed_y = 10, 40
    n_h_cells = 1

    dig_cells, river_source = sequence_side(
        n_h_cells, seed_x, seed_y, list_of_movements
    )

    assert len(dig_cells) == curve_length


def test_sequence_side_fat(list_of_movements: List[Movement]):
    # Compute the length of the curve; we start from one because we have for
    # sure the first cell
    curve_length = 1
    for movement in list_of_movements:
        curve_length += movement.length

    seed_x, seed_y = 10, 40
    n_h_cells = 5

    dig_cells, river_sources = sequence_side(
        n_h_cells, seed_x, seed_y, list_of_movements
    )

    assert len(dig_cells) == curve_length * n_h_cells


def test_apply_dig(list_of_movements: List[Movement]):
    test_matrix = np.zeros((50, 50))

    seed_x, seed_y = 10, 40
    v = 10
    n_h_cells = 3

    curve_cells, riversources = sequence_side(
        n_h_cells, seed_x, seed_y, list_of_movements
    )

    n_cells = len(curve_cells)

    apply_dig(test_matrix, curve_cells, v)

    assert np.sum(test_matrix) == n_cells * v
