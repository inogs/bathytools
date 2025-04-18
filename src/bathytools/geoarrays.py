import numpy as np
import xarray as xr

from bathytools.bathymetry_config import DomainGeometry
from bathytools.depth_levels import DepthLevels
from bathytools.water_fractions import WaterFractions


class GeoArrays:
    """
    Represents geospatial grid-based computations over the model domain.

    This class provides a framework for constructing geospatial arrays and
    performing computations such as metric cell sizes, cell areas, mesh mask
    construction, and generating static data for numerical simulations. It
    handles grid boundaries, depth levels, and domain geometry for its
    operations.

    Static attributes:
        R0 (float): Earth's radius in meters.
        G (float): Acceleration due to gravity in meters per second squared.
    """

    # Earth Radius in meters
    R0 = 6371.0e3
    G = 9.81

    def __init__(
        self,
        depth_levels: DepthLevels,
        boundary_longitudes: np.ndarray,
        boundary_latitudes: np.ndarray,
        domain_geometry: DomainGeometry,
    ):
        self._depth_levels = depth_levels
        self._boundary_longitudes = boundary_longitudes
        self._boundary_latitudes = boundary_latitudes
        self._domain_geometry = domain_geometry

    @property
    def domain_geometry(self):
        return self._domain_geometry

    @staticmethod
    def build(domain_geometry: DomainGeometry, depth_levels: DepthLevels):
        n_x = domain_geometry.n_x
        n_y = domain_geometry.n_y

        res = domain_geometry.resolution
        xg0 = domain_geometry.minimum_longitude
        yg0 = domain_geometry.minimum_latitude

        boundary_longitudes = xg0 + np.linspace(0, n_x, n_x + 1) * res
        boundary_latitudes = yg0 + np.linspace(0, n_y, n_y + 1) * res

        return GeoArrays(
            depth_levels,
            boundary_longitudes,
            boundary_latitudes,
            domain_geometry,
        )

    @property
    def cell_size(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the cell size within the domain boundaries using the
        provided resolution.

        The method computes the longitudinal and latitudinal sizes of each cell
        in the grid. It then generates a grid of uniform cell size based on the
        resolution specified in the domain geometry.

        Returns:
            A tuple of numpy arrays representing the broadcasted
            longitudinal and latitudinal cell sizes across the domain grid.

        """
        n_x = self._boundary_longitudes.shape[0] - 1
        n_y = self._boundary_latitudes.shape[0] - 1

        lon_size = np.full((n_x,), fill_value=self._domain_geometry.resolution)
        lat_size = np.full((n_y,), fill_value=self._domain_geometry.resolution)

        # noinspection PyTypeChecker
        return np.broadcast_arrays(lon_size, lat_size[:, np.newaxis])

    def compute_metric_cell_size(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the cell sizes in metres.

        The method calculates the sizes of grid cells in longitudinal
        (meridional) and latitudinal (zonal) directions based on the domain
        geometry and Earth's  radius. This is essential for transformations
        that require grid cell spacing in metric units.

        Returns:
            A tuple containing two numpy arrays. The first array represents the
            meridional metric cell size computed for each grid point. The
            second array represents the zonal metric cell size computed for
            each grid point.
        """
        dxf, dyf = self.cell_size
        latitudes = self._domain_geometry.latitude
        meridional_size = (
            np.deg2rad(dxf)
            * self.R0
            * np.cos(np.deg2rad(latitudes[:, np.newaxis]))
        )
        zonal_size = np.deg2rad(dyf) * self.R0
        return meridional_size, zonal_size

    def compute_cell_area(self):
        x, y = self.compute_metric_cell_size()
        return x * y

    def _build_mesh_mask_mer(
        self, water_fractions: WaterFractions, dtype=np.float64
    ) -> xr.Dataset:
        meridional_size, zonal_size = self.compute_metric_cell_size()

        e1t = meridional_size.astype(dtype, copy=False)
        e2t = zonal_size.astype(dtype=dtype, copy=False)
        e3t_1d = self._depth_levels.thickness.astype(dtype, copy=False)

        h_fac_min = self._domain_geometry.minimum_h_factor
        tmask = np.where(
            water_fractions.on_cells < h_fac_min / 2.0, 0, 1
        ).astype(np.int8)

        # We build e3t by computing the real depth of each cell; we use as
        # initial values the
        e3t = np.full(
            shape=water_fractions.on_cells.shape,
            fill_value=e3t_1d[:, np.newaxis, np.newaxis],
            dtype=dtype,
        )
        # How many cells do we have on each column
        cells_per_column = tmask.sum(axis=0)

        # Let us fix the columns with just one cell; in this case, the size of
        # the cell is the depth of the bathymetry
        only_one_cell = cells_per_column == 1
        e3t[0, only_one_cell] = water_fractions.refined_bathymetry[
            only_one_cell
        ]

        # Now we deal with the more complicated case; when the column has more
        # than one cell; we define a mask to identify those columns
        more_than_one_cells = cells_per_column > 1
        # This gives us the index of the cell of the column that is on the
        # bottom
        last_cell_index = cells_per_column[more_than_one_cells] - 1

        # This array of indices is useful so that
        # `v[:, more_than_one_cells][last_cell_index, column_index]` contains
        # the values of v that are on the cells of the bottom
        column_index = np.arange(np.sum(more_than_one_cells))

        # We broadcast the array with the position of the top faces to the
        # shape of the domain
        top_faces = np.broadcast_to(
            self._depth_levels.top_faces[:, np.newaxis, np.newaxis],
            tmask.shape,
        )

        def define_indices(axis):
            """
            Given a 3D array `v`of the shame shape of tmask, we need to
            build 3 arrays of indices i, j, k so that `v[i, j, k]` is the
            values of v on the cells that are on the bottom. Because the
            routine is the same for all the three axis, we use a closure to
            avoid repeating the same code.
            """
            # This slice ensure that the arrays are always 3d, with the values
            # on the appropriate axis
            slice_template = [
                slice(None) if i == axis else np.newaxis for i in range(3)
            ]
            slice_template = tuple(slice_template)
            all_indices_values = np.arange(tmask.shape[axis])[slice_template]
            return np.broadcast_to(all_indices_values, tmask.shape)[
                :, more_than_one_cells
            ][last_cell_index, column_index]

        depth_indices = define_indices(0)
        lat_indices = define_indices(1)
        lon_indices = define_indices(2)

        # We update the values of e3t
        e3t[depth_indices, lat_indices, lon_indices] = (
            water_fractions.refined_bathymetry[lat_indices, lon_indices]
            - top_faces[depth_indices, lat_indices, lon_indices]
        )

        mesh_mask_coords = {
            "depth": (
                ("depth",),
                self._depth_levels.centers.astype(dtype, copy=False),
                {
                    "units": "m",
                    "positive": "down",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
            "latitude": (
                ("latitude",),
                self._domain_geometry.latitude.astype(dtype, copy=False),
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "axis": "Y",
                },
            ),
            "longitude": (
                ("longitude",),
                self._domain_geometry.longitude.astype(dtype, copy=False),
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "axis": "X",
                },
            ),
        }

        mesh_mask_vars = {
            "e1t": (["latitude", "longitude"], e1t),
            "e2t": (["latitude", "longitude"], e2t),
            "e3t": (["depth", "latitude", "longitude"], e3t),
            "tmask": (["depth", "latitude", "longitude"], tmask),
            "delZ": (["depth"], e3t_1d),
        }

        mesh_mask = xr.Dataset(
            data_vars=mesh_mask_vars, coords=mesh_mask_coords
        )
        mesh_mask.attrs.update(mer_mesh_mask_version="1.0")

        return mesh_mask

    def build_mesh_mask(
        self,
        water_fractions: WaterFractions,
        dtype=np.float64,
        use_mer_format: bool = True,
        new_e3t=True,
    ) -> xr.Dataset:
        if use_mer_format:
            return self._build_mesh_mask_mer(water_fractions, dtype=dtype)

        meridional_size, zonal_size = self.compute_metric_cell_size()

        e1t = np.asarray(
            meridional_size[np.newaxis, np.newaxis, :, :], dtype=dtype
        )
        e2t = np.asarray(zonal_size[np.newaxis, np.newaxis, :, :], dtype=dtype)
        e3t = np.asarray(
            self._depth_levels.thickness[
                np.newaxis, :, np.newaxis, np.newaxis
            ],
            dtype=dtype,
        )

        center_lon, center_lat = np.broadcast_arrays(
            self._domain_geometry.longitude,
            self._domain_geometry.latitude[:, np.newaxis],
        )
        center_depth = self._depth_levels.centers
        glamt = np.asarray(
            center_lon[np.newaxis, np.newaxis, :, :], dtype=dtype
        )
        gphit = np.asarray(
            center_lat[np.newaxis, np.newaxis, :, :], dtype=dtype
        )

        h_fac_min = self._domain_geometry.minimum_h_factor
        tmask = np.where(
            water_fractions.on_cells < h_fac_min / 2.0, dtype(0), dtype(1)
        )

        mesh_mask_dict = {
            "e1t": (["time", "z_a", "y", "x"], e1t),
            "e2t": (["time", "z_a", "y", "x"], e2t),
            "glamt": (["time", "z_a", "y", "x"], glamt),
            "gphit": (["time", "z_a", "y", "x"], gphit),
            "nav_lat": (["y", "x"], np.asarray(center_lat, dtype=dtype)),
            "nav_lev": (["z"], np.asarray(center_depth, dtype=dtype)),
            "nav_lon": (["y", "x"], np.asarray(center_lon, dtype=dtype)),
            "tmask": (["time", "z", "y", "x"], tmask[np.newaxis, :, :, :]),
        }

        if new_e3t:
            e3t_shape = np.broadcast_shapes(e3t.shape, e2t.shape)
            e3t = np.broadcast_to(e3t, e3t_shape)
            mesh_mask_dict["e3t"] = (["time", "z", "y", "x"], e3t)
        else:
            mesh_mask_dict["e3t"] = (["time", "z", "y_a", "x_a"], e3t)

        return xr.Dataset(mesh_mask_dict)

    def build_mit_static_data(
        self, water_fractions: WaterFractions, dtype=np.float32
    ) -> xr.Dataset:
        def typed(arr):
            return arr.astype(dtype, copy=False)

        cell_area = typed(self.compute_cell_area())

        dxc, dyc = self.compute_metric_cell_size()
        dxc = typed(dxc)
        dyc = typed(dyc)

        dzg = typed(self._depth_levels.thickness)

        zc = self._depth_levels.centers
        zg = self._depth_levels.bottom_faces
        dzc = np.ones(zc.shape[0] + 1, dtype=dtype)
        dzc[0] = zc[0] * 0.5
        dzc[1:-1] = np.abs(zc[1:] - zc[:-1])
        dzc[-1] = np.abs(zg[-1] - zc[-1])

        depth = typed(water_fractions.refined_bathymetry)

        # hydrostatic pressure as reference (PHref_)
        hydro_pressure_centers = typed(
            np.abs(self._depth_levels.centers) * self.G
        )
        hydro_pressure_faces = typed(
            np.abs(self._depth_levels.face_positions) * self.G
        )

        h_fac_min = self._domain_geometry.minimum_h_factor
        mask_c = typed(
            np.where(water_fractions.on_cells < h_fac_min, 0.0, 1.0)
        )
        mask_w = typed(
            np.where(water_fractions.on_we_faces < h_fac_min, 0.0, 1.0)
        )
        mask_s = typed(
            np.where(water_fractions.on_sn_faces < h_fac_min, 0.0, 1.0)
        )

        mitgcm_vars = {
            "XC": (["XC"], typed(self._domain_geometry.longitude)),
            "YC": (["YC"], typed(self._domain_geometry.latitude)),
            "XG": (["XG"], typed(self._boundary_longitudes[:-1])),
            "YG": (["YG"], typed(self._boundary_latitudes[:-1])),
            "Z": (["Z"], typed(self._depth_levels.centers)),
            "Zp1": (["Zp1"], typed(self._depth_levels.face_positions)),
            "Zu": (["Zu"], typed(self._depth_levels.top_faces)),
            "Zl": (["Zl"], typed(self._depth_levels.bottom_faces)),
            "rA": (["YC", "XC"], cell_area),
            "dxG": (["YG", "XG"], dxc),
            "dyG": (["YG", "XG"], dyc),
            "Depth": (["YC", "XC"], depth),
            "rAz": (["YG", "XG"], cell_area),
            "dxC": (["YC", "XG"], dxc),
            "dyC": (["YG", "XC"], dyc),
            "rAw": (["YC", "XG"], cell_area),
            "rAs": (["YG", "XC"], cell_area),
            "drC": (["Zp1"], dzc),
            "drF": (["Z"], dzg),
            "PHrefC": (["Z"], hydro_pressure_centers),
            "PHrefF": (["Zp1"], hydro_pressure_faces),
            "hFacC": (["Z", "YC", "XC"], typed(water_fractions.on_cells)),
            "hFacW": (["Z", "YC", "XG"], typed(water_fractions.on_we_faces)),
            "hFacS": (["Z", "YG", "XC"], typed(water_fractions.on_sn_faces)),
            "maskC": (["Z", "YC", "XC"], mask_c),
            "maskW": (["Z", "YC", "XG"], mask_w),
            "maskS": (["Z", "YG", "XC"], mask_s),
            "maskInC": (["YC", "XC"], mask_c[0, :, :]),
            "maskInW": (["YC", "XG"], mask_w[0, :, :]),
            "maskInS": (["YG", "XC"], mask_s[0, :, :]),
            "dxF": (["YC", "XC"], dxc),
            "dyF": (["YC", "XC"], dyc),
            "dxV": (["YG", "XG"], dxc),
            "dyU": (["YG", "XG"], dyc),
        }

        return xr.Dataset(mitgcm_vars)
