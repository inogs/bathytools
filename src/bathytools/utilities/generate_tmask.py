import numpy as np
import xarray as xr


def generate_tmask(meshmask: xr.Dataset, zlevels: np.ndarray | None = None):
    depth = meshmask["nav_lev"].values
    latitude = meshmask["nav_lat"].values[:, 0]
    longitude = meshmask["nav_lon"].values[0, :]
    tmask = (meshmask["tmask"].values[0, :] > 0).astype(np.int8)

    data_vars = {"tmask": (("depth", "latitude", "longitude"), tmask)}
    if zlevels is not None:
        data_vars["zlevels"] = ("depth",), zlevels

    new_mask = xr.Dataset(
        coords={
            "depth": (
                ("depth",),
                depth,
                {
                    "units": "m",
                    "positive": "down",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
            "latitude": (
                ("latitude",),
                latitude,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "axis": "Y",
                },
            ),
            "longitude": (
                ("longitude",),
                longitude,
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "axis": "X",
                },
            ),
        },
        data_vars=data_vars,
    )
    return new_mask
