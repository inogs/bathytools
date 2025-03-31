import argparse
import dataclasses
import json
import warnings
from collections import OrderedDict
from sys import exit as sys_exit
from typing import Any
from typing import Literal

import pandas as pd
from bitsea.utilities.argparse_types import dir_to_be_created_if_not_exists
from bitsea.utilities.argparse_types import existing_file_path
from bitsea.utilities.argparse_types import path_inside_an_existing_dir

MAIN_SHEET_NAME = "ITA"
BGC_SHEET_NAME = "BGC_table_ref"
DOMAINS = ("NAD", "SAD", "ION", "SIC", "TYR", "LIG", "SAR")


@dataclasses.dataclass
class EFASDataSource:
    """
    Represents all information related to the data source for a river,
    specifically data downloaded from the EFAS archive.

    All the properties stored in this object correspond to the EFAS domain.
    For example, the longitude and latitude indices are applicable to the EFAS
    data but are not valid for other model domains.
    """

    longitude: float
    latitude: float
    longitude_index: int
    latitude_index: int

    @staticmethod
    def ordered_keys() -> tuple[str, str, str, str]:
        """
        Returns the fields of this object in the order in which they should
        be serialized.
        """
        return "longitude", "latitude", "longitude_index", "latitude_index"

    def to_dict(self) -> OrderedDict:
        """
        Converts this object into an ordered dictionary, making it easier
        to serialize the data.
        """
        data = [(t, getattr(self, t)) for t in self.ordered_keys()]
        return OrderedDict(data)


@dataclasses.dataclass
class RiverGeometry:
    """
    This class represents the geometry of a river, including information
    about its position, the width of its mouth, and the depth at its mouth.

    An optional field, `side`, specifies the side of the model from which the
    river flows (e.g., "N" for North, "E" for East, "S" for South, or "W"
    for West).
    If `side` is not set (i.e., it is `None`), it indicates that the river's
    direction may vary across different model domains.
    """

    mouth_longitude: float
    mouth_latitude: float
    width: float
    depth: float
    side: Literal["N", "E", "S", "W"] | None = None

    def to_dict(self) -> OrderedDict:
        """
        Converts this object to an ordered dictionary, facilitating
        easier serialization of the river geometry data.

        A key named "side" with the value of self.side is only added if
        `self.side` is not `None`.
        """
        data = [
            ("mouth_longitude", self.mouth_longitude),
            ("mouth_latitude", self.mouth_latitude),
            ("width", self.width),
            ("depth", self.depth),
        ]
        if self.side is not None:
            data.append(("side", self.side))
        return OrderedDict(data)


@dataclasses.dataclass
class River:
    """
    Represents a river and its associated attributes, data source, and geometry.

    This class models a river with its unique identifier, name, and various
    characteristics including runoff factor, data source, geometry, and
    biogeochemical properties. It is designed to serialize the river object
    into an ordered dictionary for consistent representation and further
    processing.

    Attributes:
        id (int): Unique identifier for the river.
        name (str): Name of the river.
        runoff_factor (float): If this number is different from 1, it indicates
            that the values for the runoff discharge of the river read from the
            data source must be multiplied by this factor.
        data_source (EFASDataSource): Source of the river-related data.
        geometry (RiverGeometry): Geometric attributes of the river's
            physical structure.
        biogeochemical (OrderedDict): Biogeochemical properties of the
            river stored in a key-value format. This dictionary may contain the
            name of the profile that will be associated to this river or the
            name of a biogeochemical variable and its associated value.
    """

    id: int
    name: str
    runoff_factor: float
    data_source: EFASDataSource
    geometry: RiverGeometry
    biogeochemical: OrderedDict

    def to_dict(self) -> OrderedDict:
        data = [
            ("id", self.id),
            ("name", self.name),
            ("runoff_factor", self.runoff_factor),
            ("data_source", self.data_source),
            ("geometry", self.geometry),
            ("biogeochemical", self.biogeochemical),
        ]
        return OrderedDict(data)


def build_json_serializer(geometry_defaults):
    class RiverJSONEncoder(json.JSONEncoder):
        """
        This is an improved version of the standard JSONEncoder class that
        has some special instructions for handling the `River` class. For example,
        in the serialized dictionary generated for a river, the `runoff_factor` is
        only included if its value is different from 1.
        """

        def __init__(self, *args, **kwargs):
            self._geometry_defaults = geometry_defaults
            super().__init__(*args, **kwargs)

        def default(self, o):
            if isinstance(o, River):
                river_dict = o.to_dict()
                if abs(o.runoff_factor - 1) < 1e-6:
                    del river_dict["runoff_factor"]
                final_out = {}
                for key, value in river_dict.items():
                    if dataclasses.is_dataclass(value):
                        final_out[key] = self.default(value)
                    else:
                        final_out[key] = value
                return final_out
            if isinstance(o, EFASDataSource):
                final_out = o.to_dict()
                final_out["type"] = "EFAS"
                return OrderedDict(
                    [(i, final_out[i]) for i in ("type",) + o.ordered_keys()]
                )
            if isinstance(o, RiverGeometry):
                raw_dict = o.to_dict()
                final_dict = OrderedDict()
                for key, value in raw_dict.items():
                    # Avoid to copy values that are equal to defaults
                    if key in self._geometry_defaults:
                        default_value = self._geometry_defaults[key]
                        if value == default_value:
                            continue
                    final_dict[key] = value
                return final_dict
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    return RiverJSONEncoder


def argument():
    parser = argparse.ArgumentParser(
        description="This script reads data from an XLSX file and  outputs a "
        "formatted JSON file tha can be used within the bathytools "
        "scripts"
    )

    parser.add_argument(
        "--input-file",
        "-i",
        type=existing_file_path,
        required=True,
        help="""
        The XLSX file that must be read as input
        """,
    )

    parser.add_argument(
        "--output-file",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="""
        The path where the main output file will be written
        """,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help="""
        If this flag is specified and the output file already exists, it will
        be overwritten. Otherwise, an error will be raised.
        """,
    )

    parser.add_argument(
        "--domain-specific-output-files",
        "-d",
        type=dir_to_be_created_if_not_exists,
        required=False,
        default=None,
        help="""
        If this flag is specified, the script will also create a separate JSON
        file for each domain inside the specified directory. The name of the
        file of each domain will be the same as the name of the sheet in the
        XLSX file. The output file will only include rivers that have different
        options on that domain respect to the main file. If a file with the same
        name already exists, it will not be overwritten unless the --overwrite
        flag is specified.
        """,
    )

    return parser.parse_args()


def read_stem_value(raw_value: str):
    raw_value = raw_value.strip()
    if raw_value.startswith("[") and raw_value.endswith("]"):
        values = raw_value[1:-1].split(";")
        return [read_stem_value(v) for v in values]

    # Remove mac curly quotes
    raw_value = raw_value.replace("\u201c", '"').replace("\u201d", '"')

    # Remove quotes
    while raw_value.startswith('"') and raw_value.endswith('"'):
        raw_value = raw_value[1:-1].strip()
    return raw_value


def read_bgc_reference_table(bgc_sheet):
    # The first row (with the name of the variables) is used as titles for
    # the columns. So the following line get the second row (the one with the
    # units of measure).
    units_data = bgc_sheet.iloc[0]

    # We save the name of the variables, so we preserve the order
    variable_index = units_data.index.tolist()
    units = units_data.to_dict()

    # Find the row that contains the BFM factors
    bfm_factors_row_name = "conversion_factor_to_BFM_units"
    conversion_factors = bgc_sheet.loc[bfm_factors_row_name].to_dict()

    # Here we create a dictionary with the same order that we have used
    # for the list. This dictionary associates to each variable name two
    # fields: the unit of that variable and the conversion factor for the BFM;
    # if the conversion factor is 1 (or very close to 1), we skip that field.
    variables = OrderedDict()
    for v in variable_index:
        v_data = OrderedDict([("unit", units[v])])
        conversion_factor = conversion_factors[v]
        if abs(conversion_factor - 1) > 1e-12:
            v_data["BFM_conversion_factor"] = conversion_factor
        variables[v] = v_data

    # Now we read the profiles
    profiles = OrderedDict()
    for i, (row_name, row) in enumerate(bgc_sheet.iterrows()):
        # Skip the first line with the units of measure
        if i == 0:
            continue
        # Skip the row with the bfm conversion factors
        if row_name == bfm_factors_row_name:
            continue
        row_dict = row.to_dict()
        profiles[row_name] = OrderedDict(
            [(v, row_dict[v]) for v in variable_index]
        )

    return variables, profiles


def generate_domain_specific_file(
    domain_sheet, rivers
) -> OrderedDict[str, Any]:
    domain_rivers = []
    enabled_only = []
    for _, row in domain_sheet.iterrows():
        river_id = row["ir"]
        river_name = row["rivername"]
        for river in rivers:
            if river.id == river_id:
                if river.name != river_name:
                    raise ValueError(
                        f"River with id = {river_id} is named "
                        f"\"{river_name}\" into the domain specific file, "
                        f"while is called \"{row['rivername']}\" into the main "
                        f"file."
                    )
                main_river = river
                break
            if river.name == river_name:
                raise ValueError(
                    f"River with name = {river_name} has id {river_id} into "
                    f"the domain specific file, while has id {river.id} into "
                    f"the main file."
                )
        else:
            raise ValueError(
                f"River with name = {river_name} is not present into the main "
                f"file."
            )

        river_dict = OrderedDict([("id", river_id), ("name", river_name)])

        river_side = row["SIDE"]
        if river_side != main_river.geometry.side:
            if "geometry" not in river_dict:
                river_dict["geometry"] = {}
            river_dict["geometry"]["side"] = river_side

        river_stem = row["shift_from_mouth"].strip()
        if river_stem.lower() != "default" and river_stem.lower() != "none":
            river_stem = read_stem_value(river_stem)
            if "geometry" not in river_dict:
                river_dict["geometry"] = {}
            river_dict["geometry"]["stem"] = river_stem

        # If this river does not contain just its id and its name, save it in
        # the final list
        if len(river_dict) > 2:
            domain_rivers.append(river_dict)

        # Save the fact that this river is enabled for this domain
        enabled_only.append(
            OrderedDict([("id", river_id), ("name", river_name)])
        )

    return OrderedDict(
        [("rivers", domain_rivers), ("enabled_only", enabled_only)]
    )


def generate_main_json(main_sheet, domain_sheets, variables, profiles):
    defaults = OrderedDict()
    geometry_defaults = OrderedDict(
        [
            ("width", 500),
            ("depth", 3),
            ("stem_length", 10),
        ]
    )
    defaults["geometry"] = geometry_defaults
    defaults["profiles"] = profiles

    rivers = []
    for index, row in main_sheet.iterrows():
        efas_data = EFASDataSource(
            longitude_index=row["jir"],
            latitude_index=row["jjr"],
            longitude=row["lon_EFAS"],
            latitude=row["lat_EFAS"],
        )

        river_sides = []
        # Read all the sides in all the domain-specific files
        for domain in domain_sheets:
            if row["ir"] in set(domain["ir"]):
                domain_row = domain.loc[domain["ir"] == row["ir"]].iloc[0]
                if domain_row["rivername"] != row["rivername"]:
                    raise ValueError(
                        f"River with id = {row['ir']} is named "
                        f"\"{row['rivername']}\"into the main file, while is "
                        f"called \"{domain_row['rivername']}\" into the domain "
                        f"specific file."
                    )
                side = domain_row["SIDE"]
                river_sides.append(side)
        side = None
        if len(set(river_sides)) == 1:
            side = river_sides[0]

        geometry = RiverGeometry(
            mouth_latitude=row["lat_mouth"],
            mouth_longitude=row["lon_mouth"],
            width=row["width_meters"],
            depth=row["depth_meters"],
            side=side,
        )
        bgc_data = OrderedDict([("profile", row["BGC_area"])])
        river = River(
            id=row["ir"],
            name=row["rivername"],
            runoff_factor=row["runoffFactor"],
            data_source=efas_data,
            geometry=geometry,
            biogeochemical=bgc_data,
        )
        rivers.append(river)

    json_data = json.dumps(
        {"variables": variables, "defaults": defaults, "rivers": rivers},
        cls=build_json_serializer(geometry_defaults),
        indent=2,
    )

    return json_data, rivers


def main() -> int:
    args = argument()

    output_file = args.output_file
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"The output file {output_file} already exists. "
            "Use the --overwrite flag to overwrite it."
        )

    xl_file = pd.ExcelFile(args.input_file)
    sheets = xl_file.sheet_names
    main_sheet = xl_file.parse(MAIN_SHEET_NAME)
    bgc_sheet = xl_file.parse(BGC_SHEET_NAME, index_col=0)

    variables, profiles = read_bgc_reference_table(bgc_sheet)

    domains = OrderedDict()
    for sheet in sheets:
        if sheet in DOMAINS:
            domains[sheet] = xl_file.parse(sheet)

    main_json, rivers = generate_main_json(
        main_sheet, domains.values(), variables, profiles
    )

    with open(output_file, "w") as f:
        f.write(main_json + "\n")

    # If we do not have to create the domain files, we can stop now the
    # execution
    if not args.domain_specific_output_files:
        return 0

    for domain_name, domain_sheet in domains.items():
        domain_file = args.domain_specific_output_files / f"{domain_name}.json"
        if domain_file.exists() and not args.overwrite:
            warnings.warn(
                f"The output file {domain_file} already exists. "
                "Use the --overwrite flag to overwrite it."
            )
        domain_river = generate_domain_specific_file(domain_sheet, rivers)
        with open(domain_file, "w") as f:
            f.write(json.dumps(domain_river, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys_exit(main())
