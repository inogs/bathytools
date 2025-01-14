from collections.abc import Iterable
import xarray as xr
import json
import numpy as np
from pathlib import Path
import cc3d
from graph_help import imshow, set_same_box, pl


def load_ita_bathy(
        url: str = "https://erddap.emodnet.eu/erddap/griddap/bathymetry_2022",
        x_vertices: Iterable = (5., 21.),
        y_vertices: Iterable = (34., 48.),
        out_file: Path = Path("ITA_bathymetry.nc"),
        full_meta: bool = False,
) -> xr.Dataset:
    """Loads bathymetry from the EMODnet server and subsets to a box containing
    all the regional domains.
    Args:
        url (string): url where to find EMODnet data
        x_vertices (iteratable of floats): min and max longitude
        y_vertices (iteratable of floats): min and max latitude
        outFile (string): file name to write bathymetry to, if absent
        full_meta (bool): flag to keep all the meta- and accessory data (std,
        max-min range, interpolation flags, etc)
    Returns:
        xarray Dataset with longitude, latitude and elevation
    """

    if not out_file.exists():
        ds = xr.open_dataset(url)
    else:
        ds = xr.open_dataset(out_file)

    if full_meta:
        ds = ds.sel(longitude=slice(x_vertices[0], x_vertices[1]),
                    latitude=slice(y_vertices[0], y_vertices[1]))
    else:
        ds = ds.elevation.sel(longitude=slice(x_vertices[0], x_vertices[1]),
                              latitude=slice(y_vertices[0], y_vertices[1])).to_dataset()

    if not out_file.exists():
        ds.to_netcdf(out_file)

    return ds
def interpolate_bathy(
        ds_ita: xr.Dataset,
        x_domain: np.ndarray,
        y_domain: np.ndarray,
        out_file: str = '-',
) -> xr.Dataset:
    """Interpolates the EMODnet bathymetry (resolution ~100 m) to the domain
    grid (~500 m); different methods can be used, but from tests they are
    pretty much equivalent (differences of order 1e-6).
    Args:
        ds_ita (xr.DataArray): EMODnet bathymetry cut to the Italian region
        x_domain (np.ndarray): longitude array of the domain
        y_domain (np.ndarray): latitude array of the domain
        out_file (string): file name to write bathymetry to, if absent
    Returns:
        xarray Dataset with longitude, latitude and elevation
    """

    ds_dom = ds_ita.interp(longitude=x_domain, latitude=y_domain, method='linear')
    ds_dom = xr.where(ds_dom > 0., 0, ds_dom)
    return ds_dom

def remove_puddles(
        ds: xr.Dataset,
        threshold: int = 500,
) -> xr.Dataset:
    """Removes unconnected pixels (puddles) from the domain, leaving only ones above
    a set threshold in pixel number.
    Args:
        ds (xr.DataArray): domain bathymetry
        threshold (np.ndarray): minimum number of pixels of a puddle
    Returns:
        xarray Dataset with longitude, latitude and 'cleaned' elevation
    """

    if isinstance(ds, xr.Dataset):
        mask_puddles = xr.where(ds == ds, 1, 0).elevation.values * np.where(ds.elevation.values == 0., 0, 1)
    elif isinstance(ds, xr.DataArray):
        mask_puddles = xr.where(ds == ds, 1, 0).values * np.where(ds == 0., 0, 1)
    elif isinstance(ds, np.ndarray):
        mask_puddles = np.where(ds == ds, 1, 0) * np.where(ds == 0., 0, 1)

     
    puddles = cc3d.connected_components(mask_puddles, connectivity=4)
    removed_puddles = cc3d.dust(puddles, connectivity=4, threshold=threshold)
    imshow(removed_puddles,"removed_puddles")
    if isinstance(ds, xr.Dataset):
        ds = ds.elevation * removed_puddles
    else:
        ds = ds * removed_puddles

    return ds, puddles


def load_domain(
        config_file: Path,
) -> tuple:
    """Loads info defining the domain of interest from .json file.
    Args:
        config_file (string): file name where to find domain config
    Returns:
        arrays of longitude and latitude for the domain
    """

    with open(config_file, 'r') as jfile:
        jdata = json.load(jfile)
    res = jdata["resolution"]
    n_x = int((jdata["maximum_longitude"] - jdata["minimum_longitude"]) / res)
    n_y = int((jdata["maximum_latitude"] - jdata["minimum_latitude"]) / res)
    lon_domain = np.linspace(jdata["minimum_longitude"] + res * .5, jdata["maximum_longitude"] - res * .5, n_x)
    lat_domain = np.linspace(jdata["minimum_latitude"] + res * .5, jdata["maximum_latitude"] - res * .5, n_y)
    return lon_domain, lat_domain

def extend_box(config_file: Path, extent: float=1.0) -> tuple:
    with open(config_file, 'r') as jfile:
        jdata = json.load(jfile)
    x_vertices = jdata["minimum_longitude"] - extent, jdata["maximum_longitude"] + extent
    y_vertices = jdata["minimum_latitude"] - extent, jdata["maximum_latitude"] + extent
    return (x_vertices, y_vertices)


    


if __name__== "__main__":
    
    pl.close('all')
    config_file = Path('domain_north_adriatic_extended.json')
    downloaded_file=Path('bathy_step0.nc')
    orig_bathy_file=Path('bathy_step1.nc')
    lon_dom, lat_dom = load_domain(config_file)
    xv,yv = extend_box(config_file, 1.0)

    ds = load_ita_bathy(x_vertices=xv, y_vertices=yv, out_file = downloaded_file)
    
    ds_dom = interpolate_bathy(ds, x_domain=lon_dom, y_domain=lat_dom)

    
    #ds_dom.to_netcdf(orig_bathy_file)
    ds2, puddles = remove_puddles(ds_dom, threshold=100)
    #ds2.to_netcdf("No_puddles.nc")
    fig1, ax1 = imshow(ds_dom.elevation.values,'orig')
    imshow(puddles,'puddles')
    A=ds_dom.elevation.values.copy()
    # removing puddles
    ii=puddles>1
    A[ii]=np.nan
    fig2, ax2 = imshow(A,'ripulita')