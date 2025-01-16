import numpy as np

from bathytools.dig import apply_dig
from bathytools.dig import sequence_side


def test_sequence_side_thin():
    dig_list = ["30E", "20S", "10W", "5N"]

    # Compute the length of the curve; we start from one because we have for
    # sure the first cell
    curve_length = 1
    for movement in dig_list:
        current_length = int(movement[:-1])
        curve_length += current_length

    seed_x, seed_y = 10, 40
    n_h_cells = 1

    dig_cells = sequence_side(n_h_cells, seed_x, seed_y, dig_list)

    assert len(dig_cells) == curve_length


def test_sequence_side_fat():
    dig_list = ["30E", "20S", "10W", "5N"]

    # Compute the length of the curve; we start from one because we have for
    # sure the first cell
    curve_length = 1
    for movement in dig_list:
        current_length = int(movement[:-1])
        curve_length += current_length

    seed_x, seed_y = 10, 40
    n_h_cells = 5

    dig_cells = sequence_side(n_h_cells, seed_x, seed_y, dig_list)

    assert len(dig_cells) == curve_length * n_h_cells


def test_apply_dig():
    test_matrix = np.zeros((50, 50))

    # DIGlist=["5E","3S","10E","3N","3W","4N"]
    dig_list = ["30E", "20S", "10W", "5N"]

    seed_x, seed_y = 10, 40
    v = 10
    n_h_cells = 3

    curve_cells = sequence_side(n_h_cells, seed_x, seed_y, dig_list)

    n_cells = len(curve_cells)

    apply_dig(test_matrix, curve_cells, v)

    assert np.sum(test_matrix) == n_cells * v
