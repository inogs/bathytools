import xarray as xr

from bathytools.actions import SimpleAction


class CapDepth(SimpleAction):
    """
    Represents an action that caps the depth values in a bathymetry dataset.

    This class is designed to modify bathymetry data by capping the maximum
    depth value to a specified limit. It can be used to enforce a depth
    constraint on bathymetric elevation data, ensuring that values do not
    exceed the specified
    maximum depth.

    Attributes:
        depth_cap (float): The maximum allowable depth. Depth values
            exceeding this limit will be capped.
    """

    def __init__(self, name: str, description: str, max_depth: float):
        super().__init__(name, description)
        self.depth_cap = float(max_depth)

    def __call__(self, bathymetry: xr.DataArray) -> xr.DataArray:
        bathymetry["elevation"] = -(-bathymetry).elevation.clip(
            max=self.depth_cap
        )
        return bathymetry
