import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

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


@dataclass
class RiverDig:
    """
    Represents a river that will be dug into a bathymetry domain

    Attributes:
        id (int): Unique identifier for the river.
        name (str): Name of the river.
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
    mouth_latitude: float
    mouth_longitude: float
    width: float
    depth: float
    side: Optional[Literal["N", "S", "E", "W"]] = None
    stem: Optional[list[Movement]] = None


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

        # Read the default river geometry values; we keep the description of the
        # stem into a separate dictionary because it needs a custom procedure to
        # be updated
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
        rivers_dig_data = {}
        for river in rivers:
            river_id = river["id"]
            river_name = river["name"]
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
            rivers_dig_data[(river_id, river_name)] = (geometry, stem)

        if self._domain_file_path is not None:
            LOGGER.debug("Reading also domain file %s", self._domain_file_path)
            with open(self._domain_file_path, "r") as f:
                domain_data = json.load(f)
        else:
            LOGGER.debug("No domain file provided; using main file data only")
            domain_data = {"rivers": []}

        if "rivers" not in domain_data:
            raise ValueError('No "rivers" section found in domain file')

        # Update rivers_dig_data with the new values from the domain file
        for river in domain_data["rivers"]:
            river_id = river["id"]
            river_name = river["name"]
            if (river_id, river_name) not in rivers_dig_data:
                raise ValueError(
                    f"River with id = {river_id} and name = {river_name} "
                    f"defined in file {self._domain_file_path} is not defined "
                    f"in the main file {self._main_file_path}"
                )
            main_geometry, main_stem = rivers_dig_data[(river_id, river_name)]
            domain_geometry = river.get("geometry", {})
            try:
                stem = self._update_stem_value(main_stem, domain_geometry)
            except Exception as e:
                raise ValueError(
                    f"Error while reading geometry section of river {river_id} "
                    f"called ({river_name}) in file {self._domain_file_path}"
                ) from e
            main_geometry.update(domain_geometry)
            rivers_dig_data[(river_id, river_name)] = (main_geometry, stem)

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

        # We create a new RiverDig for each river, and we save each one
        # into a list
        river_digs: List[RiverDig] = []
        for river_id, river_name in rivers_dig_data:
            river_geometry, river_stem = rivers_dig_data[
                (river_id, river_name)
            ]

            # If _update_stem_value did correctly its job, this should never
            # happen
            assert "stem_length" not in river_stem or "stem" not in river_stem
            assert "stem_length" not in river_geometry
            assert "stem" not in river_geometry

            # It's time to put back the description of the stem into the
            # geometry dictionary. Now we should also know the side of the
            # river (because we have collected all the information from the
            # different files).
            # We do not care too much if some data is missing for some rivers.
            # For example, we do not know the side or the stem geometry, we
            # simply do not put those values inside the dictionary, and then we
            # produce a RiverDig with those values set to `None`. Indeed, we
            # expect those rivers to be rivers that are outside our current
            # domain and, therefore, our domain file does not give us data
            # about those rivers and, on the other hand, we do not have to
            # dig them.
            if "stem_length" in river_stem:
                if "side" not in river_geometry:
                    LOGGER.debug(
                        "River %s (id = %s) has a stem_length but no side; "
                        "ignoring stem_length",
                        river_name,
                        river_id,
                    )
                else:
                    river_geometry["stem"] = self._steam_length_to_stem(
                        river_stem["stem_length"], river_geometry["side"]
                    )
                    del river_stem["stem_length"]
            elif "stem" in river_stem:
                LOGGER.debug(
                    "Reading stem of river %s (id = %s)", river_name, river_id
                )
                river_geometry["stem"] = [
                    Movement.from_zonal_meridional_description(s)
                    for s in river_stem["stem"]
                ]

            river_dig = RiverDig(
                id=river_id,
                name=river_name,
                **river_geometry,
            )
            river_digs.append(river_dig)

        # The rivers are an attribute of this object; we will use them inside
        # the __call__ method
        self.river_digs = tuple(river_digs)

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

        # We produce a list with only the rivers that are inside the domain; we
        # also ensure to have all the information that we need to dig those
        # rivers
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

            # We approximate the size of the face with the size in the center
            # of the cell. This is a good approximation, since the grid is
            # regular
            if river.side == "N" or river.side == "S":
                face_size = mask.e2t[river_cell[::-1]]
            elif river.side == "E" or river.side == "W":
                face_size = mask.e1t[river_cell[::-1]]
            else:
                raise ValueError(f"Invalid side: {river.side}")
            LOGGER.debug("The length of one face is %s m", face_size)
            n_cells = max(round(river.width / face_size), 1)
            LOGGER.debug("The river will be thick %s cells", n_cells)

            LOGGER.debug(
                "Starting digging a river from (%s, %s) using the "
                "following movements: %s",
                river_cell[0],
                river_cell[1],
                river.stem,
            )
            digging_cells = sequence_side(
                n_cells, river_cell[0], river_cell[1], river.stem
            )
            LOGGER.debug(
                "%s cells will be dug",
                len(digging_cells),
            )

            apply_dig(
                bathymetry.elevation.transpose("latitude", "longitude"),
                digging_cells,
                -river.depth,
            )

        return bathymetry
