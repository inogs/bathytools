import numpy as np
import xarray as xr

from bathytools.bathymetry_config import DomainGeometry
from bathytools.depth_levels import generate_level_heights
from bathytools.geoarrays import GeoArrays
from bathytools.water_fractions import WaterFractions


class DomainDiscretization:
    """Handles the discretization of a geographic domain for numerical
    simulations.

    This class is responsible for converting a bathymetry dataset into a
    grid-based discretization that can be used in numerical models. It provides
    information about the geographic arrays defining the simulation grid and
    the fraction of water present within individual grid cells or faces
    (commonly referred to as hFac).

    The transformation of a bathymetry dataset into a `DomainDiscretization`
    object can be achieved by using the `build` class method. This object
    provides methods to construct the mesh mask and MITgcm static datasets,
    both of which are represented as xarray Datasets and can be utilized in
    simulations.

    Attributes:
        geo_arrays (GeoArrays): Stores the geographical arrays that define the
            discretized grid.
        water_fractions (WaterFractions): Contains data regarding the amount of
            water that every cell stores, based on the calculated depth levels
            from the bathymetry and domain geometry.
    """

    def __init__(
        self,
        geo_arrays: GeoArrays,
        water_fractions: WaterFractions,
        original_bathymetry: xr.Dataset,
    ):
        self.geo_arrays = geo_arrays
        self.water_fractions = water_fractions
        self.original_bathymetry = original_bathymetry

    @staticmethod
    def build(bathymetry: xr.Dataset, domain_geometry: DomainGeometry):
        """
        Creates a `DomainDiscretization` object based on the provided
        bathymetry data and domain geometry configuration.

        This method generates vertical depth levels, water fractions, and
        geographic arrays required for simulation grid discretization.

        Args:
            bathymetry: The input dataset containing bathymetric elevation data.
            domain_geometry: The configuration of the domain geometry, including
                grid information and vertical levels.

        Returns:
            DomainDiscretization: An instance containing the discretized domain
                data.
        """
        first_layer_height = (
            domain_geometry.vertical_levels.first_layer_thickness
        )
        max_depth = domain_geometry.vertical_levels.maximum_depth
        depth_levels = generate_level_heights(first_layer_height, max_depth)

        bathymetry_values = bathymetry.elevation.transpose(
            "latitude", "longitude"
        ).values

        water_fractions = WaterFractions.build(
            depth_levels=depth_levels,
            bathymetry_data=bathymetry_values,
            domain_geometry=domain_geometry,
        )
        geo_arrays = GeoArrays.build(
            domain_geometry=domain_geometry, depth_levels=depth_levels
        )

        return DomainDiscretization(
            geo_arrays, water_fractions, original_bathymetry=bathymetry
        )

    @property
    def bathymetry(self) -> np.ndarray:
        return self.water_fractions.refined_bathymetry

    def build_mesh_mask(self) -> xr.Dataset:
        return self.geo_arrays.build_mesh_mask(self.water_fractions)

    def build_mit_static_data(self) -> xr.Dataset:
        return self.geo_arrays.build_mit_static_data(self.water_fractions)
