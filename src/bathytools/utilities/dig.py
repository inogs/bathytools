import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product as cart_prod
from logging import getLogger

import numpy as np


LOGGER = getLogger(__name__)


class DigOutsideShape(ValueError):
    pass


class Direction(Enum):
    """Represents cardinal directions for movement in a 2D grid.

    The Direction enum provides four cardinal directions (EAST, WEST,
    NORTH, SOUTH) and methods for manipulating grid indices based on these
    directions.

    Attributes:
        EAST: East direction, represented as "E"
        WEST: West direction, represented as "W"
        NORTH: North direction, represented as "N"
        SOUTH: South direction, represented as "S"
    """

    EAST = "E"
    WEST = "W"
    NORTH = "N"
    SOUTH = "S"

    def move_indices(self, i: int, j: int, amount: int = 1) -> tuple[int, int]:
        """Calculates new grid indices after moving in the direction by a
        specified amount.

        Args:
            i: Current x-coordinate/column index (represent latitude)
            j: Current y-coordinate/row index (represent longitude)
            amount (optional): Distance to move. Defaults to 1.

        Returns:
            tuple[int, int]: New (i, j) coordinates after movement
        """
        if self == Direction.EAST:
            j = j + amount
        elif self == Direction.WEST:
            j = j - amount
        elif self == Direction.NORTH:
            i = i + amount
        elif self == Direction.SOUTH:
            i = i - amount
        else:
            raise ValueError(f"Invalid direction: {self}")
        return i, j

    def __neg__(self):
        """Returns the opposite cardinal direction.

        Returns:
            The opposite direction (EAST->WEST, WEST->EAST, NORTH->SOUTH,
            SOUTH->NORTH)
        """
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
    """Represents a single movement in a 2D grid with length and direction.

    This class defines a movement by combining a distance (length) and a
    cardinal direction. It provides methods to calculate new grid indices
    after movement and parse movement descriptions from different string
    formats.

    Usually, length is a positive integer. It cannot be negative unless it
    is -1. In this case, the movement is infinite in that direction (up to the
    end of the domain)

    Attributes:
        length: The distance to move in grid units
        direction: The cardinal direction of movement (EAST, WEST, NORTH,
            SOUTH)
    """

    length: int
    direction: Direction

    def move_indices(self, i, j) -> tuple[int, int]:
        """Calculate new grid indices after applying the movement.

        Args:
            i: Current x-coordinate/column index
            j: Current y-coordinate/row index

        Returns:
            New (i, j) coordinates after movement
        """
        if self.length < 0:
            if self.length != -1:
                raise ValueError(f"Invalid movement length: {self.length}")
            raise ValueError(
                "This movement is infinite in direction "
                f"{self.direction.name.lower()}; this object cannot be used "
                "directly but must be copied with the correct length when "
                "the dimensions of the domain are known"
            )

        return self.direction.move_indices(i, j, self.length)

    @staticmethod
    def from_str(s: str):
        """Create a Movement from a string in format '<length><direction>'.

        Args:
            s: String in format like "30E", "20N", where the number is the
                length and the letter is the direction (E, W, N, S)

        Returns:
            A new `Movement` instance

        Raises:
            ValueError: If the string format is invalid
        """
        string_mask = re.compile(r"^(\d+)([EWNS])$")
        match = string_mask.match(s)
        if match is None:
            raise ValueError(f"Invalid movement string: {s}")
        return Movement(int(match.group(1)), Direction(match.group(2)))

    @staticmethod
    def from_zonal_meridional_description(s: str):
        """Create a Movement from a zonal/meridional description string.

        Args:
            s: String in format like "30z", "-20m", where the number is the
                length and 'z'/'Z' indicates zonal (East/West) movement,
                'm'/'M' indicates meridional (North/South) movement

        Returns:
            A new `Movement` instance

        Raises:
            ValueError: If the string format is invalid
        """
        string_mask = re.compile(
            r"^(?P<length>[+-]?(\d+|\*))(?P<direction>[mMzZ])$"
        )
        match = string_mask.match(s)
        if match is None:
            raise ValueError(f"Invalid movement string: {s}")
        if match.group("direction") in ["m", "M"]:
            direction = Direction.NORTH
        elif match.group("direction") in ["z", "Z"]:
            direction = Direction.EAST
        else:
            raise ValueError(f"Invalid direction: {match.group(2)}")

        # In case there is a "*" in the length, we put a -1 in the length to
        # indicate that the movement is infinite in that direction
        if match.group("length").endswith("*"):
            if match.group("length").startswith("-"):
                direction = -direction
            return Movement(-1, direction)

        # If we can read the value, we ensure that is always positive in the
        # right direction
        value = int(match.group("length"))

        if value < 0:
            direction = -direction
            value = -value

        return Movement(value, direction)

    def copy(self, length: int = None, direction: Direction = None):
        """Create a copy of this Movement with modified attributes."""
        if length is None:
            length = self.length
        if direction is None:
            direction = self.direction
        return Movement(length, direction)


class StartIndexStrategy(Enum):
    """Defines different strategies for choosing starting indices for digs.

    When we define a dig, we need to specify where to start digging. If the dig
    thickness is greater than 1, we face the problem that the starting point
    is a square made of many cells, but the starting point is only defined by
    a single cell. This enum defines the different strategies for choosing the
    starting point of the dig with respect to a single cell.

    BOTTOM_LEFT: the cell (i, j) is the bottom-left cell of the starting square,
        i.e., the starting point is a pair with the smaller indices in both
        the direction of the cells that define the starting square.
    CENTERED: the cell (i, j) is the center of the starting square.
    SIDE_CENTERED: Start digs from the center of the side of the square that
        is opposite to the direction of the first movement of the dig.
    """

    BOTTOM_LEFT = 0
    CENTERED = 1
    SIDE_CENTERED = 2


class Dig:
    """Represents a path-based mask generator for 2D grids.

    The Dig class creates masks in 2D arrays by following a sequence of
    movements and filling the path with a specified value. Each dig can have
    its own thickness and can handle movements within or outside the domain
    boundaries.

    Args:
        movements: Sequence of Movement objects defining the path to dig
        thick: Width of the path in grid units (default: 1)
        start_index_strategy: Strategy for choosing starting indices
            (default: BOTTOM_LEFT)

    Attributes:
        movements: A sequence of Movement objects defining the path
        thick: Width of the path in grid units (default: 1)
    """

    def __init__(
        self,
        movements: Sequence[Movement],
        thick: int = 1,
        start_index_strategy: StartIndexStrategy = StartIndexStrategy.BOTTOM_LEFT,
    ):
        self.movements: tuple[Movement, ...] = tuple(movements)
        self.thick: int = thick
        self._start_index_strategy = start_index_strategy

    def _transform_starting_indices(
        self, start_indices: tuple[int, int]
    ) -> tuple[int, int]:
        """Transform starting indices according to the chosen strategy.

        The method "_get_dig_slices" (which actually computes where this
        object must dig) implements an algorithm that assumes that the starting
        indices are the bottom-left corner of the initial square (i.e., the
        smaller indices inside the starting square).
        If the starting indices aren't in the bottom-left corner, we need to
        convert them to. This method solves this problem, moving the starting
        indices from the position specified by the user to the bottom-left.
        """
        initial_strategy = self._start_index_strategy
        if initial_strategy == StartIndexStrategy.BOTTOM_LEFT:
            return start_indices

        if initial_strategy == StartIndexStrategy.CENTERED:
            shift = self.thick // 2
            return start_indices[0] - shift, start_indices[1] - shift

        if initial_strategy == StartIndexStrategy.SIDE_CENTERED:
            if len(self.movements) == 0:
                raise ValueError(
                    "Cannot choose side-centered starting indices for empty dig"
                )
            if self.movements[0].direction == Direction.EAST:
                return start_indices[0] - self.thick // 2, start_indices[1]
            elif self.movements[0].direction == Direction.WEST:
                return start_indices[0] - self.thick // 2, start_indices[
                    1
                ] - self.thick + 1
            elif self.movements[0].direction == Direction.NORTH:
                return start_indices[0], start_indices[1] - self.thick // 2
            else:
                return start_indices[0] - self.thick + 1, start_indices[
                    1
                ] - self.thick // 2

        raise ValueError(f"Invalid initial strategy: {initial_strategy}")

    def _check_index_inside(
        self, i: int, j: int, shape: tuple[int, int]
    ) -> None:
        """Validate if the given indices are within the domain boundaries.

        This function expects i and j to be the position of the bottom-left
        corner of the square that defines a specific position (and that
        the side of the square has length `self.thick`).

        Args:
            i: x-coordinate/column index to check
            j: y-coordinate/row index to check
            shape: Tuple of (height, width) defining the domain size

        Raises:
            DigOutsideShape: If the indices are outside the domain
        """
        if i < 0:
            raise DigOutsideShape(
                f"First index {i} of position {(i, j)} is outside the domain"
            )
        if i + self.thick - 1 >= shape[0]:
            raise DigOutsideShape(
                f"The first index of position {(i, j)} draws a rectangle that "
                f"is too large for the domain (shape: {shape}, side: "
                f"{self.thick})"
            )
        if j < 0:
            raise DigOutsideShape(
                f"Second index {j} of position {(i, j)} is outside the domain"
            )
        if j + self.thick - 1 >= shape[1]:
            raise DigOutsideShape(
                f"The second index of position {(i, j)} draws a rectangle "
                f"that is too large for the domain (shape: {shape}, side: "
                f"{self.thick})"
            )

    def _get_dig_slices(
        self,
        start_indices: tuple[int, int],
        domain_shape: tuple[int, int],
        allow_outside: bool = False,
    ) -> tuple[tuple[slice, slice], ...]:
        """Get slices for filling the mask with the path defined by this dig.

        Produces a sequence of slices that define the area to fill when the dig
        is executed. Each element of the output sequence corresponds to a 2D
        slice (a tuple with two slices) that defines a rectangle in the dig.
        """
        i, j = self._transform_starting_indices(start_indices)

        # Check if the starting indices are inside the domain
        if not allow_outside:
            try:
                self._check_index_inside(i, j, domain_shape)
            except DigOutsideShape as e:
                raise DigOutsideShape(
                    "Start indices are outside the domain"
                ) from e

        # This is the final output. Each slice here corresponds to a rectangle
        slices = []

        # If there are no movements, the dig is a square of side length thick
        if len(self.movements) == 0:
            slices.append((slice(i, i + self.thick), slice(j, j + self.thick)))
            return tuple(slices)

        for movement_index, movement in enumerate(self.movements):
            # If the movement is infinite in a direction, we must create a
            # new movement of the right length in the same direction.
            if movement.length == -1:
                new_direction = movement.direction

                # If the direction is East-Weast, we modify j (the second
                # index); otherwise we modify i
                if movement.direction in (Direction.EAST, Direction.WEST):
                    indx = 1
                    starting_point = j
                else:
                    indx = 0
                    starting_point = i

                # If the direction moves us toward the last index of the domain
                if movement.direction in (Direction.EAST, Direction.NORTH):
                    length = domain_shape[indx] - self.thick - starting_point
                    # if this is negative, it means that we started from
                    # outside the domain. Then we need to invert the direction
                    # and reach the last index of the domain
                    if length < 0:
                        length = -length
                        new_direction = -new_direction
                else:
                    # In this case, we need to reach the first index; if
                    # starting_point is negative, it means that we started from
                    # outside the domain (and we need to invert the direction)
                    length = starting_point
                    if starting_point < 0:
                        new_direction = -new_direction
                        length *= -1
                movement = Movement(direction=new_direction, length=length)

            # (i, j) is the current position (top-left corner). (new_i, new_j)
            # is the new one after the current movement
            new_i, new_j = movement.move_indices(i, j)

            # Check if the new position is inside the domain
            if not allow_outside:
                try:
                    self._check_index_inside(new_i, new_j, domain_shape)
                except DigOutsideShape as e:
                    raise DigOutsideShape(
                        f"Movement {movement_index} ({movement}) moves the "
                        f"dig outside the domain"
                    ) from e

            min_i, max_i = min(new_i, i), max(new_i, i)
            min_j, max_j = min(new_j, j), max(new_j, j)

            i, j = new_i, new_j

            # If this slice is completely out of the domain, we ignore it, and
            # we continue with the next movement
            if max_i + self.thick <= 0:
                continue
            if min_i >= domain_shape[0]:
                continue
            if max_j + self.thick <= 0:
                continue
            if min_j >= domain_shape[1]:
                continue

            # We create the corresponding slices. We cut the slices so to be
            # always inside the domain
            slices.append(
                (
                    slice(
                        max(0, min_i), min(max_i + self.thick, domain_shape[0])
                    ),
                    slice(
                        max(0, min_j), min(max_j + self.thick, domain_shape[1])
                    ),
                )
            )
        return tuple(slices)

    def fill_dig_mask(
        self,
        start_indices: tuple[int, int],
        out: np.ndarray,
        value=True,
        allow_outside: bool = False,
    ):
        """Fill a mask array with a path following the specified movements.

        Creates a path in the output array starting from start_indices and
        following the sequence of movements. The path is filled with the
        specified value and can have variable thickness.
        If the thickness of this path is larger than 1, the starting point
        (i.e., the points that would be filled if the sequence of movements
        was empty) is a square of side length thick. In this case, the meaning
        of the coordinates (i, j) depends on the start_index_strategy chosen
        when this object has been defined.

        Args:
            start_indices: Starting position (i, j) for the path
            out: Output numpy array to fill with the mask
            value: Value to set in the mask along the path (default: True)
            allow_outside: If True, allow path to extend outside domain
                         (default: False)

        Raises:
            DigOutsideShape: If allow_outside is False and the path extends
                            outside domain
        """
        slices = self._get_dig_slices(
            start_indices, out.shape[-2:], allow_outside
        )

        for movement_slice in slices:
            out[..., movement_slice[0], movement_slice[1]] = value

    def get_dig_cells(
        self,
        start_indices: tuple[int, int],
        domain_shape: tuple[int, int],
        allow_outside: bool = False,
    ) -> set[tuple[int, int]]:
        """Returns the indices of the cells that are part of the current dig

        This method returns the indices of the cells that would have been
        filled if the method `fill_dig_mask` had been called with the same
        start_indices and domain_shape.

        The result is returned as a tuple of coordinate pairs, each
        representing a cell in the path of the dig.

        Args:
            start_indices: The starting indices (row, column) for the dig
            domain_shape: The shape of the grid as a tuple (rows, columns)

        Returns:
            A set of tuples, where each tuple represents the (row, column)
            coordinates of a cell of the dig.
        """
        cells: set[tuple[int, int]] = set()

        slices = self._get_dig_slices(
            start_indices, domain_shape, allow_outside=allow_outside
        )

        for slice_x, slice_y in slices:
            assert slice_x.step is None or slice_x.step == 1
            assert slice_y.step is None or slice_y.step == 1
            cells.update(
                cart_prod(
                    range(slice_x.start, slice_x.stop),
                    range(slice_y.start, slice_y.stop),
                )
            )

        return cells

    def get_dig_source(
        self, start_indices: tuple[int, int], domain_shape: tuple[int, int]
    ) -> set[tuple[int, int]]:
        """Returns the last cells of the dig that are part of the domain

        We can imagine a dig as a path drawn by a square that moves from the
        position described by the starting indices to its last position
        following the movements. This method returns the indices of the last
        cells that have been dig. This means that the output is a vertical or
        horizontal line made of `thick` cells and with width equal to one.
        If this dig represents a river, then those cells are the river's source.

        Args:
            start_indices: The starting indices (row, column) for the dig
            domain_shape: The shape of the grid as a tuple (rows, columns)

        Returns:
            A set of tuples, where each tuple represents the (row, column)
            coordinates of the last cells dug by this object.
        """
        slices = self._get_dig_slices(
            start_indices, domain_shape, allow_outside=False
        )

        assert len(slices) == len(self.movements)

        # We get the last slice associated with this dig. This is the last
        # rectangle that we have dug.
        last_slice = slices[-1]

        # We identify the indices along with the last movement has dug. The
        # line of the cells of the source will be perpendicular to this index.
        if self.movements[-1].direction in (Direction.EAST, Direction.WEST):
            moving_index = 1
        else:
            moving_index = 0

        # This is the other index, the one the source cells are aligned with.
        fixed_index: int = 1 - moving_index

        # The dig related to the last slice moves between
        # last_slice[moving_index].start and last_slice[moving_index].stop - 1;
        # how do we identify which one we must choose?
        if self.movements[-1].direction in (Direction.WEST, Direction.SOUTH):
            get_index = min
        else:

            def get_index(*args):
                return max(*args) - 1

        # Along the fixed_index, the width of the dig must be equal to the
        # thickness of the dig.
        assert (
            last_slice[fixed_index].stop - last_slice[fixed_index].start
            == self.thick
        )

        # This is the position of the sources (one of their two indices)
        d_index = get_index(
            last_slice[moving_index].start, last_slice[moving_index].stop
        )

        cells = set()
        for i in range(
            last_slice[fixed_index].start, last_slice[fixed_index].stop
        ):
            current_cell = [None, None]
            current_cell[moving_index] = d_index
            current_cell[fixed_index] = i
            cells.add(tuple(current_cell))

        return cells
