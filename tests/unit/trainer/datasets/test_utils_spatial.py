import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
from numpy.testing import assert_array_equal

from trainer import rasterize


def test_rasterize__burn() -> None:
    data_dir = Path(__file__).parents[3] / 'data'
    input_vector = data_dir / 'box.gpkg'
    output_raster = data_dir / 'output.tif'
    prj = 4326

    try:
        # create a vector
        box = shapely.geometry.box(0, 0, 0.1, 0.1)
        gdf = gpd.GeoDataFrame(geometry=[box], data={'value': [2]}, crs=prj)
        gdf.to_file(input_vector, layer='box')

        rasterize(input_vector, output_raster, pixel_size=0.01, no_data_value=0)

        with rasterio.open(output_raster) as src:
            prof = src.profile
            transform = src.transform

            # affine projection checks pixel resolution
            assert transform[0] == 0.01
            assert transform[4] == -0.01
            assert src.crs == pyproj.crs.CRS(f'EPSG:{prj}').to_wkt()

            assert prof['nodata'] == 0
            assert prof['driver'] == 'GTiff'
            assert prof['tiled'] == True
            assert prof['compress'] == 'lzw'

            # read raster has right values
            ras = src.read(1)
            assert_array_equal(ras, np.ones(shape=[10, 10]))

    finally:
        os.remove(input_vector)
        os.remove(output_raster)


def test_rasteriz__attribute() -> None:
    data_dir = Path(__file__).parents[3] / 'data'
    input_vector = data_dir / 'box.gpkg'
    output_raster = data_dir / 'output.tif'
    prj = 4326

    try:
        # create a vector
        box = shapely.geometry.box(0, 0, 0.01, 0.01)
        box_2 = shapely.geometry.box(0.01, 0.01, 0.02, 0.02)
        gdf = gpd.GeoDataFrame(geometry=[box, box_2], data={'value': [1, 2]}, crs=prj)
        gdf.to_file(input_vector, layer='box')

        rasterize(input_vector, output_raster, attribute='value', pixel_size=0.01, no_data_value=0)

        with rasterio.open(output_raster) as src:
            prof = src.profile
            transform = src.transform

            # affine projection checks pixel resolution
            assert transform[0] == 0.01
            assert transform[4] == -0.01
            assert src.crs == pyproj.crs.CRS(f'EPSG:{prj}').to_wkt()

            assert prof['nodata'] == 0
            assert prof['driver'] == 'GTiff'
            assert prof['tiled'] == True
            assert prof['compress'] == 'lzw'

            # read raster has right values - 2 nodata 0s and 2 values
            ras = src.read(1)
            assert_array_equal(ras, np.array([[0, 2], [1, 0]]))

    finally:
        os.remove(input_vector)
        os.remove(output_raster)
