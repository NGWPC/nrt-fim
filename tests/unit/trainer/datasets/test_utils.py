import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
from numpy.testing import assert_array_equal

from trainer.datasets.utils import rasterize


def test_rasterize() -> None:
    """Builds a simple vector to rasterize. Vector is 4x4 with bottom boxes value=1, top left no data, top right = 2.
    Uses a projected CRS (UTM Zone 31N, at lat-lon 0,0).
    
    Known warning: RuntimeWarning: invalid value encountered in cast
        data = encode_cf_variable(out_data.variable).values.astype
    Output raster confirmed to have all data and set no data value.
    """
    data_dir = Path(__file__).parents[3] / "data"
    output_raster = data_dir / "output.tif"
    prj = 32631 # UTM 31N - NULL ISLAND

    try:
        # create a vector - a 2x2 box near null island (0, 0)
        box = shapely.geometry.box(166022, 0, 166024, 1)
        box_2 = shapely.geometry.box(166023, 1, 166024, 2)
        gdf = gpd.GeoDataFrame(geometry=[box, box_2], data={"value": [1, 2]}, crs=prj)

        rasterize(gdf, output_raster, attribute="value", resolution=1, nodata_value=0, crs=prj, dtype=np.uint8)

        with rasterio.open(output_raster) as src:
            prof = src.profile
            transform = src.transform

            # affine projection checks pixel resolution
            assert transform[0] == 1
            assert transform[4] == -1
            assert src.crs == pyproj.crs.CRS(f"EPSG:{prj}").to_wkt()

            assert prof["nodata"] == 0
            assert prof["driver"] == "GTiff"
            assert prof["tiled"] is True
            assert prof["compress"] == "lzw"

            # read raster has right values - 1 nodata as 0
            ras = src.read(1)
            assert_array_equal(ras, np.array([[0, 2], [1, 1]]))

    finally:
        os.remove(output_raster)
