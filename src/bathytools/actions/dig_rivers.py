from __future__ import annotations

import json
import warnings
from collections import namedtuple
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger
from operator import attrgetter
from operator import itemgetter
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
import xarray as xr
from bitsea.commons.geodistances import compute_geodesic_distance
from bitsea.commons.grid import RegularGrid
from bitsea.commons.mask import Mask

from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix
from bathytools.utilities.dig import apply_dig
from bathytools.utilities.dig import Direction
from bathytools.utilities.dig import Movement
from bathytools.utilities.dig import sequence_side
from bathytools.utilities.relative_paths import read_path


LOGGER = getLogger(__name__)


class Interval(namedtuple("Interval", ["start", "end"])):
    """
    Represents an interval with a start and an end.

    The Interval class encapsulates a range of integers defined by a start
    (inclusive) and an end (exclusive). It provides methods to convert the
    interval to a string, check if an item belongs to the interval, and
    determine whether two intervals intersect.

    Attributes:
        start (int): The start of the interval (inclusive).
        end (int): The end of the interval (exclusive).
    """

    def __str__(self):
        return f"({self.start}, {self.end})"

    def __contains__(self, item):
        try:
            if item == int(item):
                return self.start <= int(item) < self.end
        except (TypeError, ValueError):
            return False
        return False

    def intersect(self, other):
        # If `other` is an integer and falls within this interval,
        # an intersection exists
        try:
            if other == int(other):
                return other in self
        except (TypeError, ValueError):
            pass

        if not isinstance(other, Interval):
            raise TypeError(f"Cannot compare Interval with {type(other)}")
        # If `other` completely encompasses this interval, check if this
        # interval is non-empty
        if other.start <= self.start and other.end >= self.end:
            return self.start < self.end

        # If `other` does not completely encompass this interval, check if this
        # interval contains at least one boundary of `other`
        return other.start in self or other.end in self


@dataclass
class RiverDig:
    """
    Represents a river that will be dug into a bathymetry domain

    Attributes:
        id (int): Unique identifier for the river.
        name (str): Name of the river.
        model (str): How the river is modelled.
        mouth_latitude (float): Latitude of the river's mouth.
        mouth_longitude (float): Longitude of the river's mouth.
        width (float): Width of the river channel in meters.
        depth (float): Depth of the river channel in meters.
        side (Optional[Literal["N", "S", "E", "W"]]): The source side of the
            river, specifying the direction from where the river originates
            relative to the grid (North, South, East, West).
        stem (Optional[list[Movement]]): Sequence of movements representing the
            course of the river through the bathymetry grid.
    """

    id: int
    name: str
    model: str
    mouth_latitude: float
    mouth_longitude: float
    width: float
    depth: float
    side: Optional[Literal["N", "S", "E", "W"]] = None
    stem: Optional[list[Movement]] = None


@dataclass
class RiverSource:
    """
    Represent the source of a river, i.e., the cells from which the water
    originates. This object contains the id of the river and the name of the
    river, together with the x and y coordinates of the cells from which the
    river starts. If the river originate from only one cell, then the `x` and
    `y` attributes are integers, otherwise only one of them is an `Interval`.
    It also contains the model of the original river, if the source needs to
    behave differently depending on the model.
    """

    id: int
    name: str
    model: str
    x: int | Interval
    y: int | Interval

    def to_json(self) -> OrderedDict[str, int | str]:
        """
        Returns an OrderedDict that contains the same information of the
        original object but that can be easily serialized into a JSON file
        (because the intervals are converted to strings).
        """
        x_str = str(self.x) if isinstance(self.x, Interval) else self.x
        y_str = str(self.y) if isinstance(self.y, Interval) else self.y

        return OrderedDict(
            [
                ("id", self.id),
                ("name", self.name),
                ("model", self.model),
                ("latitude", x_str),
                ("longitude", y_str),
            ]
        )


class RiverSourceJSONEncoder(json.JSONEncoder):
    """
    An encoder that can serialize `RiverSource` objects into a JSON file by
    using their `to_json` method.
    """

    def default(self, o):
        if isinstance(o, RiverSource):
            return o.to_json()
        return super().default(o)


class DigRivers(SimpleAction):
    """
    Dig the rivers on the domain
    """

    @staticmethod
    def _steam_length_to_stem(
        steam_length: int, side: Literal["N", "S", "E", "W"]
    ) -> List[Movement]:
        """
        Converts the "stem_length" parameter into a list of `Movement` objects.

        The river's stem can either be explicitly described as a list of
        movements or simply as a length. In cases where the explicit stem is
        not provided, this function generates a stem based on its length,
        with the assumption that the river flows in the direction opposite to
        its defined side.

        Args:
            steam_length (int): The length of the river's stem measured in
                grid cells.
            side (Literal["N", "S", "E", "W"]): The side of the grid
                (North, South, East, or West) representing the river's origin.

        Returns:
            List[Movement]: A single `Movement`, representing the river's stem
            based on its direction and length.

        Raises:
            ValueError: If the provided side is invalid or `steam_length` is not
            a positive integer.
        """
        if side not in ("N", "S", "E", "W"):
            raise ValueError(f"Invalid side: {side}")
        if steam_length <= 0:
            raise ValueError(
                "Steam length must be a positive number of cell: received "
                f"{steam_length}"
            )
        river_side = Direction(side)

        output = [Movement(steam_length, river_side)]
        LOGGER.debug(
            'Converting steam length "%s" from side "%s" to stem %s',
            steam_length,
            side,
            output,
        )
        return output

    @staticmethod
    def _update_stem_value(
        previous_stem_value: Mapping, new_values: Dict
    ) -> Dict:
        """
        Updates the geometry dictionary with the most recent description of the
        river's stem.

        River stem information can originate from multiple sources, including:
        the "defaults" section of the main file, the river-specific section of
        the main file, and the domain file. While other attributes can simply
        be updated with new values, handling the stem information needs special
        care, since it may be described as either a list of movements (`stem`)
        or its length (`stem_length`).

        This function checks if the incoming dictionary (`new_values`)
        specifies either the `stem` or the `stem_length`. If so, it updates the
        geometry appropriately while removing the previous stem or stem_length
        entries.

        Args:
            previous_stem_value (Mapping): The existing dictionary containing
                the stem geometry.
            new_values (Dict): The dictionary with the latest geometry updates.

        Returns:
            Dict: A combined dictionary representing the current state of the
            stem geometry.

        Raises:
            ValueError: If both `stem` and `stem_length` are defined in the
            incoming values.
        """
        if "stem_length" in new_values and "stem" in new_values:
            raise ValueError(
                'Both "stem_length" and "stem" are specified for the same river'
            )
        if "stem_length" in new_values:
            stem_length = new_values["stem_length"]
            del new_values["stem_length"]
            return {"stem_length": stem_length}
        if "stem" in new_values:
            stem_value = new_values["stem"]
            del new_values["stem"]
            return {"stem": stem_value}
        return dict(previous_stem_value)

    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        main_file: str,
        domain_file: Optional[str] = None,
    ):
        super().__init__(name, description, output_appendix=output_appendix)
        self._main_file_path = read_path(main_file)
        self._domain_file_path = (
            read_path(domain_file) if domain_file else None
        )

        # Parse the main file and load its data
        with open(self._main_file_path, "r") as f:
            main_data = json.load(f)

        # Load default geometry values for rivers, keeping stem data separate
        # because it requires custom handling for proper updates.
        if "defaults" in main_data and "geometry" in main_data["defaults"]:
            default_values = dict(main_data["defaults"]["geometry"])

            # Check for inconsistent definitions of stem attributes
            if "stem_length" in default_values and "stem" in default_values:
                raise ValueError(
                    'Both "stem_length" and "stem" are specified in the '
                    "defaults section."
                )
            default_stem = self._update_stem_value({}, default_values)
        else:
            default_values = {}
            default_stem = {}
        LOGGER.debug(
            "Reading rivers with the following defaults: %s; stem: %s",
            default_values,
            default_stem,
        )

        if "rivers" not in main_data:
            raise ValueError("No rivers found in main file")

        # Read the main section
        rivers = main_data["rivers"]
        rivers_dig_data: dict[tuple[int, str], tuple[str, dict, dict]] = {}
        for river in rivers:
            river_id = int(river["id"])
            river_name = str(river["name"])
            river_model = str(river["model"])
            LOGGER.debug("Reading river %s (id = %s)", river_name, river_id)
            geometry = default_values.copy()
            current_geo_values = river.get("geometry", {})
            try:
                stem = self._update_stem_value(
                    default_stem.copy(), current_geo_values
                )
            except Exception as e:
                raise ValueError(
                    "Error while reading geometry section of river "
                    f"{river_name} (id = {river_id}) in file "
                    f"{self._main_file_path}"
                ) from e
            geometry.update(current_geo_values)
            rivers_dig_data[(river_id, river_name)] = (
                river_model,
                geometry,
                stem,
            )

        if self._domain_file_path is not None:
            LOGGER.debug("Reading also domain file %s", self._domain_file_path)
            with open(self._domain_file_path, "r") as f:
                domain_data = json.load(f)
        else:
            LOGGER.debug("No domain file provided; using main file data only")
            domain_data = {"rivers": []}

        if "rivers" not in domain_data:
            raise ValueError('No "rivers" section found in domain file')

        # Incorporate updated geometry and stem values from the domain file
        # into `rivers_dig_data`, ensuring consistency with the main file.
        for river in domain_data["rivers"]:
            river_id = int(river["id"])
            river_name = str(river["name"])
            if (river_id, river_name) not in rivers_dig_data:
                raise ValueError(
                    f"River with id = {river_id} and name = {river_name} "
                    f"defined in file {self._domain_file_path} is not defined "
                    f"in the main file {self._main_file_path}"
                )
            river_model, main_geometry, main_stem = rivers_dig_data[
                (river_id, river_name)
            ]
            domain_geometry = river.get("geometry", {})
            try:
                stem = self._update_stem_value(main_stem, domain_geometry)
            except Exception as e:
                raise ValueError(
                    f"Error while reading geometry section of river {river_id} "
                    f"called ({river_name}) in file {self._domain_file_path}"
                ) from e
            main_geometry.update(domain_geometry)
            if "model" in river:
                river_model = str(river["model"])
            rivers_dig_data[(river_id, river_name)] = (
                river_model,
                main_geometry,
                stem,
            )

        if "enabled_only" in domain_data:
            LOGGER.debug('Reading section "enabled_only" from domain file')
            previous_rivers = rivers_dig_data
            rivers_dig_data = {}
            enabled_only = domain_data["enabled_only"]
            for river in enabled_only:
                river_id = river["id"]
                river_name = river["name"]
                if (river_id, river_name) not in previous_rivers:
                    raise ValueError(
                        f"River with id = {river_id} and name = {river_name} "
                        'defined in the "enabled_only" section of the file '
                        f"{self._domain_file_path} is not defined "
                        f"in the main file {self._main_file_path}"
                    )
                LOGGER.debug(
                    "Enabling river %s (id = %s)", river_name, river_id
                )
                rivers_dig_data[(river_id, river_name)] = previous_rivers[
                    (river_id, river_name)
                ]
        else:
            LOGGER.debug('No "enabled_only" section found in domain file')

        # Generate a `RiverDig` instance for each river using the consolidated
        # geometry and stem data, then store these in a list.
        river_digs: List[RiverDig] = []
        for river_id, river_name in rivers_dig_data:
            river_model, river_geometry, river_stem = rivers_dig_data[
                (river_id, river_name)
            ]

            # If _update_stem_value did correctly its job, this should never
            # happen
            assert "stem_length" not in river_stem or "stem" not in river_stem
            assert "stem_length" not in river_geometry
            assert "stem" not in river_geometry

            # Now it is time to reintroduce the description of the stem into
            # the geometry dictionary. By this point, we should also have the
            # river's side information, as all the data from various files
            # has been consolidated.
            # If some information is missing for certain rivers, it is not
            # an issue. For instance, if the side or the stem geometry is
            # missing, we simply omit those values in the dictionary. As a
            # result, the RiverDig object for such rivers will have `None`
            # values for those fields. This is acceptable, as these rivers
            # are likely outside our current domain, and thus do not have
            # data in the domain file. Consequently, these rivers are not
            # intended to be dug.
            if "stem_length" in river_stem:
                if "side" not in river_geometry:
                    LOGGER.debug(
                        "River %s (id = %s) has a stem_length but no side; "
                        "ignoring stem_length",
                        river_name,
                        river_id,
                    )
                else:
                    # noinspection PyTypeChecker
                    river_geometry["stem"] = self._steam_length_to_stem(
                        river_stem["stem_length"], river_geometry["side"]
                    )
                    del river_stem["stem_length"]
            elif "stem" in river_stem:
                LOGGER.debug(
                    "Reading stem of river %s (id = %s)", river_name, river_id
                )
                # noinspection PyTypeChecker
                river_geometry["stem"] = [
                    Movement.from_zonal_meridional_description(s)
                    for s in river_stem["stem"]
                ]

            river_dig = RiverDig(
                id=river_id,
                name=river_name,
                model=river_model,
                **river_geometry,
            )
            river_digs.append(river_dig)

        # The rivers are an attribute of this object; we will use them inside
        # the __call__ method
        self.river_digs = tuple(river_digs)

    @staticmethod
    def read_river_atlas(
        river_sources: dict[tuple[int, str], tuple[str, Direction, list]],
    ):
        """
        Creates a "river atlas" by transforming river source data into a
        structured dictionary, categorized by each side of the domain.
        For each domain side, the function aggregates a list of `RiverSource`
        objects representing the river sources originating from that side.

        Args:
            river_sources: A dictionary where the keys are tuples representing
                river identifiers (a river's ID and its name), and the values
                are tuples containing:
                - The river model
                - The side of the river's origin (as a `Direction` enum).
                - A list of grid cells where the river originates.

        Returns:
            dict: A "river atlas" where each key is a `Direction` (domain side),
            and the value is a list of `RiverSource` objects representing the
            rivers originating from that side.

        Raises:
            ValueError: If source cells are missing, misaligned, or overlapping.
        """
        source_atlas = {}
        for side in Direction:
            source_atlas[side] = []
            # Collect all rivers originating from the current side, and sort
            # them by ID.
            side_rivers = (r for r, k in river_sources.items() if k[1] == side)
            side_rivers = sorted(side_rivers, key=itemgetter(0))

            # Define axes based on the river's side: horizontal alignment for
            # north/south and vertical alignment for east/west.
            fixed_axis = 1 if side in (Direction.NORTH, Direction.SOUTH) else 0
            moving_axis = 1 - fixed_axis

            for river_id, river_name in side_rivers:
                river_model, _, source_cells = river_sources[
                    (river_id, river_name)
                ]
                if len(source_cells) == 0:
                    raise ValueError(
                        f"No source cells found for river {river_name} (id = "
                        f"{river_id})"
                    )
                # Verify if source cells align along the fixed coordinate
                # (single unique value).
                fixed_coords = set(map(itemgetter(fixed_axis), source_cells))
                if len(fixed_coords) > 1:
                    raise ValueError(
                        f"River {river_name} (id = {river_id}) has side "
                        f'"{side.value}" but its source cells are not aligned '
                        "with this side: "
                        + ", ".join([str(c) for c in source_cells])
                    )
                fixed_coord = fixed_coords.pop()

                # Collect and verify moving coordinates along the other axis
                # (non-fixed).
                moving_coords = sorted(
                    map(itemgetter(moving_axis), source_cells)
                )
                LOGGER.debug(
                    "River %s (id = %s) has source cells with fixed axis %s = %s "
                    "and moving axis %s = %s",
                    river_name,
                    river_id,
                    fixed_axis,
                    fixed_coord,
                    moving_axis,
                    moving_coords,
                )

                moving_coords_range = moving_coords[-1] - moving_coords[0] + 1
                if moving_coords_range != len(moving_coords):
                    raise ValueError(
                        f"River {river_name} (id = {river_id}) has "
                        f"non-contiguous source cells: {source_cells}"
                    )
                # Convert moving coordinates to an interval or a single integer.
                if moving_coords[-1] == moving_coords[0]:
                    moving_coords = moving_coords[0]
                else:
                    moving_coords = Interval(
                        moving_coords[0], moving_coords[-1] + 1
                    )

                # Create a RiverSource object and add it to the list for
                # this side.
                current_river = [river_id, river_name, river_model, None, None]
                current_river[fixed_axis + 3] = fixed_coord
                current_river[moving_axis + 3] = moving_coords
                current_river_source = RiverSource(*current_river)
                source_atlas[side].append(current_river_source)

            # Ensure no RiverSource objects on this side have overlapping
            # domains.
            intervals = []
            for r_source in source_atlas[side]:
                if fixed_axis == 0:
                    intervals.append((r_source.y, r_source.name))
                else:
                    intervals.append((r_source.x, r_source.name))
            intervals = sorted(
                intervals,
                key=lambda x: x[0].start
                if isinstance(x[0], Interval)
                else x[0],
            )
            for (i1, r1), (i2, r2) in zip(intervals[:-1], intervals[1:]):
                # Check for overlapping intervals or identical positions.
                overlap = isinstance(i1, Interval) and i1.intersect(i2)
                overlap = overlap or i1 == i2
                if overlap:
                    raise ValueError(
                        f'Two rivers on side "{side.value}" have overlapping '
                        f'source domains: "{r1}" on {i1} and "{r2}" on {i2}'
                    )

        return source_atlas

    def save_river_sources(
        self,
        river_sources: dict[tuple[int, str], tuple[str, Direction, list]],
        lat_lon_shape: tuple[int, int],
        open_side: dict[Direction, bool],
    ):
        """
        Saves detailed information about river sources into a JSON file and
        generates namelists to configure boundary conditions for the MITgcm
        model.

        This function organizes and structures the river data to specify where
        boundary conditions related to rivers need to be applied in the model.
        It also produces namelist files that define the open boundaries for
        rivers.

        Args:
            river_sources (dict): A dictionary where the keys represent a unique
                river identifier (a tuple containing the river ID and name), and
                the values are tuples with:
                - A `Direction` indicating the originating side of the river.
                - A list of grid cells from which the river originates.
            lat_lon_shape (tuple): The shape of the domain grid (number of
                latitude cells, number of longitude cells).
            open_side (dict): A dictionary mapping each domain side
                (`Direction`) to a boolean value indicating whether the
                boundary is open (True for a boundary with water) or closed
                (if there are only land cells).
        """
        river_atlas = self.read_river_atlas(river_sources)

        # Serialize the river source data into a JSON file. Enumerations are
        # converted to their corresponding string labels (via the `.value`
        # attribute).
        river_position_content = json.dumps(
            {a.value: b for a, b in river_atlas.items()},
            indent=2,
            cls=RiverSourceJSONEncoder,
        )

        source_file = (
            self._output_appendix.output_dir / "rivers_positions.json"
        )
        with open(source_file, "w") as f:
            f.write(river_position_content + "\n")

        # Create namelist files to configure river-specific open boundary
        # conditions for the numerical model. We start by writing the function
        # that we will use inside our routine when we have to register a river;
        # it writes
        def write_obj_interval(length: int, position: int) -> str:
            return f"{length}*{position},"

        obj_file_content = ""
        for side in Direction:
            # Determine the prefix for each line based on the boundary side.
            if side == Direction.NORTH:
                line_prefix = "OB_Jnorth="
            elif side == Direction.SOUTH:
                line_prefix = "OB_Jsouth="
            elif side == Direction.EAST:
                line_prefix = "OB_Ieast="
            elif side == Direction.WEST:
                line_prefix = "OB_Iwest="
            else:
                raise ValueError(f"Unexpected side: {side}")

            # Calculate the total number of boundary cells for the current side.
            if side in (Direction.NORTH, Direction.SOUTH):
                side_length = lat_lon_shape[1]
                other_side_length = lat_lon_shape[0]
            else:
                side_length = lat_lon_shape[0]
                other_side_length = lat_lon_shape[1]

            # Determine the default position for boundary cells with no rivers.
            # For open sides, this is 1 (start of axis) or the side length
            # (end of axis); for closed sides, this is set to 0.
            if side in (Direction.NORTH, Direction.EAST):
                no_river_position = other_side_length if open_side[side] else 0
            else:
                no_river_position = 1 if open_side[side] else 0

            # Define helper functions to retrieve the river's position on the
            # side and its offset (translation) with respect to the boundary.
            if side in (Direction.NORTH, Direction.SOUTH):
                get_side_coord = attrgetter("x")
                get_translation = attrgetter("y")
            else:
                get_side_coord = attrgetter("y")
                get_translation = attrgetter("x")

            # Determine the starting position of a river source along the
            # domain side.
            def get_start_point(r: RiverSource):
                side_position = get_side_coord(r)
                if isinstance(side_position, Interval):
                    return side_position.start
                return side_position

            # Exclude rivers that are modeled as rains
            side_rivers = [
                r for r in river_atlas[side] if r.model != "rain_like"
            ]

            # Sort the sources based on their starting point.
            current_side_sources = sorted(side_rivers, key=get_start_point)

            # If the side has no rivers, write a single line to assign all
            # boundary cells to the default value (either open or closed).
            if len(current_side_sources) == 0:
                obj_file_content += (
                    line_prefix
                    + write_obj_interval(side_length, no_river_position)
                    + "\n"
                )
                continue

            # Begin writing the line for the current domain side.
            obj_file_content += f"{line_prefix}"

            # Track where the previous river ended. At the start, this is set
            # to 0.
            previous_position = 0
            for river in current_side_sources:
                river_start = get_start_point(river)
                river_translation = get_translation(river)
                river_interval = get_side_coord(river)
                # Ensure the river's position along the side is an interval;
                # convert it to an interval if it's a single integer.
                if not isinstance(river_interval, Interval):
                    river_interval = Interval(
                        river_interval, river_interval + 1
                    )

                # Calculate the number of grid cells occupied by the current
                # river.
                river_range = river_interval.end - river_start

                # If there is a gap between the current river and the previous
                # one, add it to the line being written.
                if river_start != previous_position:
                    obj_file_content += write_obj_interval(
                        river_start - previous_position, no_river_position
                    )

                # Update the ending position for the next river.
                previous_position = river_interval.end

                # Write the cells occupied by the current river. Add "+ 1" to
                # the translation for Fortran's 1-based indexing.
                obj_file_content += write_obj_interval(
                    river_range, river_translation + 1
                )

            # If the end of the side has not been reached, fill the remaining
            # cells with the "no_river" value.
            if previous_position != side_length:
                obj_file_content += write_obj_interval(
                    side_length - previous_position, no_river_position
                )

            # Finalize and close the line for this side.
            obj_file_content += "\n"

        obj_file = (
            self._output_appendix.output_dir / "rivers_open_boundaries.txt"
        )
        LOGGER.info("Writing rivers open boundaries to %s", obj_file)
        with open(obj_file, "w") as f:
            f.write(obj_file_content)

    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
        """
        Perform the river digging action on a bathymetry dataset.

        This method digs rivers into a provided bathymetry.
        It validates each river, determines its location with respect to the
        bathymetry grid, and then applies the digging process for all rivers
        within the defined domain.

        Args:
            bathymetry: The bathymetry in which the rivers will be dug.

        Returns:
            The modified bathymetry with the dug rivers.

        Raises:
            ValueError: If crucial river details (e.g., side or stem) are
                missing in the configuration.
        """
        # Create a fake mask with only 1 level ad depth 0.5 meter. We will use
        # it to locate the closest wet cell near the river mouth
        grid = RegularGrid(lon=bathymetry.longitude, lat=bathymetry.latitude)
        mask_v = bathymetry.elevation.transpose("latitude", "longitude") < -0.5
        mask = Mask(
            grid=grid,
            zlevels=[0.5],
            mask_array=mask_v,
            allow_broadcast=True,
        )

        # Check if a side is "open", i.e. if it has at least one water cell
        b_array = bathymetry.elevation.transpose(
            "latitude", "longitude"
        ).values
        water_cells = b_array < 0
        open_sides = {
            Direction.SOUTH: bool(np.any(water_cells[0, :])),
            Direction.NORTH: bool(np.any(water_cells[-1, :])),
            Direction.WEST: bool(np.any(water_cells[:, 0])),
            Direction.EAST: bool(np.any(water_cells[:, -1])),
        }
        LOGGER.debug("Open sides: %s", open_sides)

        # Filter rivers to include only those within the domain, ensuring each
        # has the required side and stem data for successful processing.
        rivers_inside = []
        for river in self.river_digs:
            river_lat = river.mouth_latitude
            river_lon = river.mouth_longitude
            is_inside = mask.is_inside_domain(lon=river_lon, lat=river_lat)
            if is_inside:
                if river.side is None:
                    raise ValueError(
                        f"River {river.name} (id = {river.id}) is inside the "
                        f"domain but there is no information about its side; "
                        f"ensure to have specified a side in the main file"
                    )
                if river.stem is None:
                    raise ValueError(
                        f"River {river.name} (id = {river.id}) is inside the "
                        f"domain but there is no information about its stem;"
                        f"ensure to have specified a stem or a stem_length "
                        f"(together with a side) in the main file"
                    )
                rivers_inside.append(river)
            else:
                LOGGER.debug(
                    "River %s (id = %s) is outside the domain and it will "
                    "*NOT* be dug",
                    river.name,
                    river.id,
                )

        LOGGER.debug("%s rivers must be dug", len(rivers_inside))

        river_sources = {}
        for river in rivers_inside:
            LOGGER.debug("Digging river %s (id = %s)", river.name, river.id)
            river_lat = river.mouth_latitude
            river_lon = river.mouth_longitude

            # Find the closest water cell for the river
            river_cell = mask.convert_lon_lat_wetpoint_indices(
                lon=river_lon, lat=river_lat, max_radius=None
            )
            new_lon = mask.xlevels[river_cell[::-1]]
            new_lat = mask.ylevels[river_cell[::-1]]

            # We log how much we have to move from the original point
            LOGGER.debug(
                "River %s (id = %s) mouth is at cell %s (LAT %.4f, LON %.4f); "
                "in configuration file it was (LAT %.4f, LON %.4f)",
                river.name,
                river.id,
                river_cell,
                new_lat,
                new_lon,
                river_lat,
                river_lon,
            )
            mouth_distance = compute_geodesic_distance(
                lat1=river_lat, lon1=river_lon, lat2=new_lat, lon2=new_lon
            )
            # From metres to km
            mouth_distance /= 1e3
            if mouth_distance > 20.0:
                warnings.warn(
                    f"Mouth of river {river.name} (id = {river.id}) has been "
                    f"moved more than 20 km ({mouth_distance:.3f} km) from "
                    f"its original position; it is probably an error in the "
                    "configuration file"
                )

            # Approximate the face size using the center of the cell, which
            # provides a reasonable estimate due to the grid's regular
            # structure.
            if river.side == "N" or river.side == "S":
                face_size = mask.e1t[river_cell[::-1]]
            elif river.side == "E" or river.side == "W":
                face_size = mask.e2t[river_cell[::-1]]
            else:
                raise ValueError(f"Invalid side: {river.side}")
            LOGGER.debug("The length of one face is %s m", face_size)
            n_cells = max(round(river.width / face_size), 1)
            LOGGER.debug(
                "River %s will be thick %s cells", river.name, n_cells
            )

            LOGGER.debug(
                "Starting digging a river from (%s, %s) using the "
                "following movements: %s",
                river_cell[0],
                river_cell[1],
                river.stem,
            )
            digging_cells, current_river_sources = sequence_side(
                n_cells, river_cell[0], river_cell[1], river.stem
            )
            LOGGER.debug(
                "%s cells will be dug",
                len(digging_cells),
            )
            river_sources[(river.id, river.name)] = (
                river.model,
                Direction(river.side),
                current_river_sources,
            )

            apply_dig(
                bathymetry.elevation.transpose("latitude", "longitude"),
                digging_cells,
                -river.depth,
            )

        self.save_river_sources(river_sources, b_array.shape, open_sides)

        return bathymetry
