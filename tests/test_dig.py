from itertools import product as cart_prod

import numpy as np
import pytest

from bathytools.utilities.dig import Dig
from bathytools.utilities.dig import Direction
from bathytools.utilities.dig import Movement
from bathytools.utilities.dig import StartIndexStrategy


def test_dig_single_line():
    """Test digging a single line in a 2D grid."""
    list_of_movements = [
        Movement(4, Direction.EAST),
        Movement(3, Direction.SOUTH),
        Movement(6, Direction.WEST),
        Movement(2, Direction.NORTH),
    ]

    dig = Dig(movements=list_of_movements)

    grid = np.zeros((10, 10), dtype=int)

    dig.fill_dig_mask((5, 5), out=grid, value=7, allow_outside=False)

    expected_output = np.zeros_like(grid)
    expected_output[5, 5:10] = 7
    expected_output[2:6, 9] = 7
    expected_output[2, 3:10] = 7
    expected_output[2:5, 3] = 7

    assert np.all(grid == expected_output)


def test_dig_survives_outside_domain():
    list_of_movements = [
        Movement(11, Direction.EAST),
        Movement(5, Direction.NORTH),
        Movement(14, Direction.WEST),
        Movement(5, Direction.NORTH),
        Movement(4, Direction.EAST),
    ]

    dig = Dig(movements=list_of_movements, thick=3)

    grid = np.zeros((15, 12), dtype=int)

    dig.fill_dig_mask((1, 1), out=grid, value=7, allow_outside=True)

    expected_output = np.zeros_like(grid)
    expected_output[1:4, 1:] = 7
    expected_output[6:9, :] = 7
    expected_output[6:14, 0] = 7
    expected_output[11:14, 0:5] = 7
    assert np.all(grid == expected_output)


@pytest.mark.parametrize("last_direction", Direction)
def test_dig_sources(last_direction: Direction):
    list_of_movements = [
        Movement(4, Direction.EAST),
        Movement(3, Direction.SOUTH),
        Movement(7, Direction.WEST),
        Movement(7, Direction.NORTH),
        Movement(1, last_direction),
    ]

    # if last_direction is south, then we cannot come from the North otherwise
    # there is an overlap. So we add an extra movement to the list.
    if last_direction == Direction.SOUTH:
        list_of_movements.insert(-1, Movement(4, Direction.EAST))

    thick = 2

    dig = Dig(movements=list_of_movements[:], thick=thick)

    sources = dig.get_dig_source((4, 4), (12, 12))

    assert len(sources) == thick

    if last_direction == Direction.EAST:
        assert (8, 3) in sources
        assert (9, 3) in sources
    if last_direction == Direction.WEST:
        assert (8, 0) in sources
        assert (9, 0) in sources
    if last_direction == Direction.NORTH:
        assert (10, 1) in sources
        assert (10, 2) in sources
    if last_direction == Direction.SOUTH:
        assert (7, 5) in sources
        assert (7, 6) in sources


@pytest.mark.parametrize(
    ["direction", "thick", "mov_length"],
    cart_prod(Direction, (1, 3, 5), (2, 3, 4)),
)
def test_starting_index_strategies_odd(
    direction: Direction, thick: int, mov_length: int
):
    x, y = (11, 15)
    dig = Dig(
        movements=[Movement(mov_length, direction)],
        thick=thick,
        start_index_strategy=StartIndexStrategy.SIDE_CENTERED,
    )

    test_grid = np.zeros((29, 37), dtype=bool)
    dig.fill_dig_mask((x, y), out=test_grid, value=True, allow_outside=False)

    expected_output = np.zeros_like(test_grid)

    if direction == Direction.EAST:
        expected_output[
            x - thick // 2 : x + thick // 2 + 1, y : y + thick + mov_length
        ] = True
    elif direction == Direction.WEST:
        expected_output[
            x - thick // 2 : x + thick // 2 + 1,
            y - thick - mov_length + 1 : y + 1,
        ] = True
    elif direction == Direction.NORTH:
        expected_output[
            x : x + thick + mov_length, y - thick // 2 : y + thick // 2 + 1
        ] = True
    elif direction == Direction.SOUTH:
        expected_output[
            x - thick - mov_length + 1 : x + 1,
            y - thick // 2 : y + thick // 2 + 1,
        ] = True

    assert np.all(test_grid == expected_output)
