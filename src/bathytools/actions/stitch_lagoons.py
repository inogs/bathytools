import logging
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from bitsea.basins.region import Polygon
from bitsea.components.component_mask import ComponentMask

#from bathytools import geoarrays
from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix
from bathytools.utilities.points import Point
from bathytools.utilities.points import Segment


LOGGER = logging.getLogger(__name__)

class StitchVeniceLagoon(SimpleAction):
    """
    Test of SimpleAction class.
    """
    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        threshold: float,
        min_depth: float,
        input_files: dict[str, str],
        window: dict[float, float, float, float],
        input_crs: dict[str, str],
    ):
        super().__init__(name, description, output_appendix=output_appendix)

        self._threshold = threshold
        self._min_depth = min_depth
        self._input_files = input_files
        self._window = window
        self._crss = input_crs
    #
    def __call__(self, bathymetry):
        xc = bathymetry.longitude.sel(longitude = slice(self._window['min_lon'], self._window['max_lon'])).values
        yc = bathymetry.latitude.sel(latitude = slice(self._window['min_lat'], self._window['max_lat'])).values
        xg = np.zeros(len(xc)+1)
        xg[0] = xc[0] - 0.5*np.diff(xc).mean()
        xg[1:] = xc + 0.5*np.diff(xc).mean()
        yg = np.zeros(len(yc)+1)
        yg[0] = yc[0] - 0.5*np.diff(yc).mean()
        yg[1:] = yc + 0.5*np.diff(yc).mean()
        #
        df_list = []
        meth1 = True
        if meth1:
            for fl in ['file1', 'file2']:
                df = pd.read_csv(self._input_files[fl])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'], df['Depth']), crs=self._crss[fl]).to_crs(crs='EPSG:4326')
                df_list.append(gdf)
            concat_df = pd.concat(df_list)['geometry']
            print(concat_df.x)
            print(concat_df.y)
            sumZ = np.histogram2d(concat_df.x, concat_df.y, weights=concat_df.z, density=False, bins=(xg, yg))[0].T
            numZ = np.histogram2d(concat_df.x, concat_df.y, density=False, bins=(xg, yg))[0].T
            plt.imshow(sumZ, origin = 'lower'); plt.colorbar(); plt.show()
            plt.imshow(numZ, origin = 'lower'); plt.colorbar(); plt.show()
            lagoon = np.where(numZ > 0, sumZ/numZ, 0.)
        else:
            for fl in ['file1', 'file2']:
                df = pd.read_csv(self._input_files[fl])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'], df['Depth']), crs=self._crss[fl]).to_crs(crs='EPSG:4326')['geometry']
                sumZ = np.histogram2d(gdf.x, gdf.y, weights=gdf.z, density=False, bins=(xg, yg))[0].T
                numZ = np.histogram2d(gdf.x, gdf.y, density=False, bins=(xg, yg))[0].T
                df_list.append(np.where(numZ > 0, sumZ/numZ, 0.))
                plt.imshow(sumZ, origin = 'lower'); plt.colorbar(); plt.show()
                plt.imshow(numZ, origin = 'lower'); plt.colorbar(); plt.show()
            lagoon = np.mean(df_list, axis = 0)
        plt.imshow(lagoon, origin = 'lower', vmin = -15); plt.colorbar(); plt.show()
        print('\n•••••••••••••')
        print(lagoon.shape)
        print(yc[0], yc[-1])
        print(xc[0], xc[-1])
        print(bathymetry['elevation'])
        print(bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]].shape)
        print('•••••••••••••\n')
        #
        R0 = 6.371e6
        Xg, _ = np.meshgrid(xg, yc)
        _, Yg = np.meshgrid(xc, yg)
        dXg = (np.diff(Xg, axis = 1).T * R0 * np.pi/180 * np.cos(yc * np.pi/180)).T
        dYg = np.diff(Yg, axis = 0) * R0 * np.pi/180
        print(dXg.mean())
        print(dYg.mean())
        dA = dXg * dYg
        ini_vol = (lagoon * dA)[lagoon != 0].sum()
        # --remove ponds
        lagoon = np.where(lagoon > -self._threshold, 0., np.min([-self._min_depth * np.ones_like(lagoon), lagoon], axis=0))
        water_cells = bathymetry['elevation'].values < 0.0
        components = ComponentMask(water_cells)
        sea_cells = components.get_component(components.get_biggest_component())
        outside_main_component = np.logical_not(sea_cells)
        bathymetry["elevation"].values[outside_main_component] = 0.0
        # --
        lagoon = np.where(lagoon > -self._threshold, 0., np.min([-self._min_depth * np.ones_like(lagoon), lagoon], axis=0))
        fin_vol = (lagoon * dA)[lagoon != 0].sum()
        d_vol = fin_vol - ini_vol
        tmp = np.zeros_like(lagoon)
        tmp[lagoon != 0] = d_vol / len(lagoon[lagoon != 0]) / dA[lagoon != 0]
        plt.imshow(tmp, origin = 'lower'); plt.colorbar(); plt.show()
        lagoon[lagoon != 0] += d_vol / len(lagoon[lagoon != 0]) / dA[lagoon != 0]
        print('\n•••••••••••••')
        print(f'Initial lagoon volume: V₀ = {np.abs(ini_vol)*1e-9:.3f} km³')
        print(f'Final lagoon volume: V₁ = {np.abs(fin_vol)*1e-9:.3f} km³')
        print(f'Variation in volume: ΔV = {d_vol*1e-9:.3f} km³')
        print('•••••••••••••\n')
        #bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]] += lagoon.T
        tmp_oldbat = bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]]
        tmp_oldbat = tmp_oldbat.where(tmp_oldbat != 0.)
        tmp_newbat = np.nanmean([tmp_oldbat.values, np.where(lagoon != 0, lagoon, np.nan).T], axis = 0)
        bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]] = np.where(tmp_newbat == tmp_newbat, tmp_newbat, 0.)
        bathymetry['elevation'].T.plot.imshow(vmin = -20.); plt.show()
        #
        return bathymetry

class StitchMaranoLagoon(SimpleAction):
    """
    Test of SimpleAction class.
    """
    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        threshold: float,
        min_depth: float,
        input_files: dict[str, str],
        window: dict[float, float, float, float],
        input_crs: dict[str, str],
    ):
        super().__init__(name, description, output_appendix=output_appendix)

        self._threshold = threshold
        self._min_depth = min_depth
        self._input_files = input_files
        self._window = window
        self._crss = input_crs
    #
    def __call__(self, bathymetry):
        xc = bathymetry.longitude.sel(longitude = slice(self._window['min_lon'], self._window['max_lon'])).values
        yc = bathymetry.latitude.sel(latitude = slice(self._window['min_lat'], self._window['max_lat'])).values
        xg = np.zeros(len(xc)+1)
        xg[0] = xc[0] - 0.5*np.diff(xc).mean()
        xg[1:] = xc + 0.5*np.diff(xc).mean()
        yg = np.zeros(len(yc)+1)
        yg[0] = yc[0] - 0.5*np.diff(yc).mean()
        yg[1:] = yc + 0.5*np.diff(yc).mean()
        #
        df_list = []
        meth1 = True
        if meth1:
            for fl in ['file1', 'file2']:
                df = pd.read_csv(self._input_files[fl])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'], df['Depth']), crs=self._crss[fl]).to_crs(crs='EPSG:4326')
                df_list.append(gdf)
            concat_df = pd.concat(df_list)['geometry']
            print(concat_df.x)
            print(concat_df.y)
            sumZ = np.histogram2d(concat_df.x, concat_df.y, weights=concat_df.z, density=False, bins=(xg, yg))[0].T
            numZ = np.histogram2d(concat_df.x, concat_df.y, density=False, bins=(xg, yg))[0].T
            plt.imshow(sumZ, origin = 'lower'); plt.colorbar(); plt.show()
            plt.imshow(numZ, origin = 'lower'); plt.colorbar(); plt.show()
            lagoon = np.where(numZ > 0, sumZ/numZ, 0.)
        else:
            for fl in ['file1', 'file2']:
                df = pd.read_csv(self._input_files[fl])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'], df['Depth']), crs=self._crss[fl]).to_crs(crs='EPSG:4326')['geometry']
                sumZ = np.histogram2d(gdf.x, gdf.y, weights=gdf.z, density=False, bins=(xg, yg))[0].T
                numZ = np.histogram2d(gdf.x, gdf.y, density=False, bins=(xg, yg))[0].T
                df_list.append(np.where(numZ > 0, sumZ/numZ, 0.))
                plt.imshow(sumZ, origin = 'lower'); plt.colorbar(); plt.show()
                plt.imshow(numZ, origin = 'lower'); plt.colorbar(); plt.show()
            lagoon = np.mean(df_list, axis = 0)
        plt.imshow(lagoon, origin = 'lower', vmin = -15); plt.colorbar(); plt.show()
        print('\n•••••••••••••')
        print(lagoon.shape)
        print(yc[0], yc[-1])
        print(xc[0], xc[-1])
        print(bathymetry['elevation'])
        print(bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]].shape)
        print('•••••••••••••\n')
        #
        R0 = 6.371e6
        Xg, _ = np.meshgrid(xg, yc)
        _, Yg = np.meshgrid(xc, yg)
        dXg = (np.diff(Xg, axis = 1).T * R0 * np.pi/180 * np.cos(yc * np.pi/180)).T
        dYg = np.diff(Yg, axis = 0) * R0 * np.pi/180
        print(dXg.mean())
        print(dYg.mean())
        dA = dXg * dYg
        ini_vol = (lagoon * dA)[lagoon != 0].sum()
        # --remove ponds
        lagoon = np.where(lagoon > -self._threshold, 0., np.min([-self._min_depth * np.ones_like(lagoon), lagoon], axis=0))
        water_cells = bathymetry['elevation'].values < 0.0
        components = ComponentMask(water_cells)
        sea_cells = components.get_component(components.get_biggest_component())
        outside_main_component = np.logical_not(sea_cells)
        bathymetry["elevation"].values[outside_main_component] = 0.0
        # --
        fin_vol = (lagoon * dA)[lagoon != 0].sum()
        d_vol = fin_vol - ini_vol
        tmp = np.zeros_like(lagoon)
        tmp[lagoon != 0] = d_vol / len(lagoon[lagoon != 0]) / dA[lagoon != 0]
        plt.imshow(tmp, origin = 'lower'); plt.colorbar(); plt.show()
        lagoon[lagoon != 0] += d_vol / len(lagoon[lagoon != 0]) / dA[lagoon != 0]
        print('\n•••••••••••••')
        print(f'Initial lagoon volume: V₀ = {np.abs(ini_vol)*1e-9:.3f} km³')
        print(f'Final lagoon volume: V₁ = {np.abs(fin_vol)*1e-9:.3f} km³')
        print(f'Variation in volume: ΔV = {d_vol*1e-9:.3f} km³')
        print('•••••••••••••\n')
        #bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]] += lagoon.T
        tmp_oldbat = bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]]
        tmp_oldbat = tmp_oldbat.where(tmp_oldbat != 0.)
        tmp_newbat = np.nanmean([tmp_oldbat.values, np.where(lagoon != 0, lagoon, np.nan).T], axis = 0)
        bathymetry['elevation'].loc[xc[0]:xc[-1], yc[0]:yc[-1]] = np.where(tmp_newbat == tmp_newbat, tmp_newbat, 0.)
        bathymetry['elevation'].T.plot.imshow(vmin = -15.)
        plt.gca().set_aspect(2**.5)
        plt.gcf().set_size_inches((20,12))
        plt.savefig(f'test01/NAD_lag_exmpl_{self._threshold:.1f}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        #
        return bathymetry
