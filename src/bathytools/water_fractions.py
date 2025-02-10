from dataclasses import dataclass

import numpy as np

from bathytools.bathymetry_config import DomainGeometry
from bathytools.depth_levels import DepthLevels


@dataclass
class WaterFractions:
    """
    Represents water fraction data derived from bathymetry and depth levels.

    This class encapsulates the calculation and representation of water
    fractions based on bathymetry data, domain geometry, and predefined depth
    levels. The water fraction is defined as the percentage of the volume of
    a specific cell that is occupied by the sea.

    Attributes:
        refined_bathymetry: Refined bathymetry data calculated based on water
            fractions and depth levels.
        on_cells: Water fraction values defined on grid cells.
        on_sn_faces: Water fraction values on south-north faces of the grid.
        on_we_faces: Water fraction values on west-east faces of the grid.
    """

    refined_bathymetry: np.ndarray
    on_cells: np.ndarray
    on_sn_faces: np.ndarray
    on_we_faces: np.ndarray

    @staticmethod
    def build(
        depth_levels: DepthLevels,
        bathymetry_data: np.ndarray,
        domain_geometry: DomainGeometry,
    ):
        """
        Constructs and calculates the water fractions for the provided depth
        levels, bathymetry data, and domain geometry.
        """
        top_faces = -depth_levels.top_faces[:, np.newaxis, np.newaxis]
        dz = depth_levels.thickness[:, np.newaxis, np.newaxis]

        percentage_of_water = np.clip(
            (top_faces - bathymetry_data) / dz, 0.0, 1.0
        )

        min_percentage = domain_geometry.minimum_h_factor
        h_fac_c = percentage_of_water.copy()

        h_fac_c[percentage_of_water <= min_percentage] = min_percentage
        h_fac_c[percentage_of_water <= min_percentage / 2.0] = 0
        h_fac_c[:, bathymetry_data >= 0] = 0

        refined_bathymetry = np.sum(h_fac_c * dz, axis=0)

        on_we_faces = np.zeros_like(h_fac_c)
        on_we_faces[:, :, 0] = h_fac_c[:, :, 0]
        on_we_faces[:, :, 1:] = np.min(
            [h_fac_c[:, :, :-1], h_fac_c[:, :, 1:]], axis=0
        )

        on_sn_faces = np.zeros_like(h_fac_c)
        on_sn_faces[:, 0, :] = h_fac_c[:, 0, :]
        on_sn_faces[:, 1:, :] = np.min(
            [h_fac_c[:, :-1, :], h_fac_c[:, 1:, :]], axis=0
        )

        return WaterFractions(
            refined_bathymetry, h_fac_c, on_sn_faces, on_we_faces
        )
