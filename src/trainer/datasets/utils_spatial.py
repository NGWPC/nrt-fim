"""A separate spatial utils library due to GDAL dependency"""

import pathlib

from osgeo import gdal, ogr


def rasterize(
    input_vector: str | pathlib.Path,
    output_raster: str | pathlib.Path,
    pixel_size: int,
    no_data_value: int = 0,
    output_type=gdal.GDT_Byte,
    attribute: str = None,
) -> None:
    """Wrapper for rasterizing a vector layer with GDAL.
    Hardcoded for compressed (LZW), cloud-optimized (tiled), burning value of 1 if no attribute value to set.

    Args:
        input_vector (str | pathlib.Path): input vector (e.g. .shp, .gpkg)
        output_raster (str | pathlib.Path): output raster (e.g. .tif, .img)
        pixel_size (int): pixel resolution
        no_data_value (int, optional): No data value to set Defaults to 0.
        output_type (_type_, optional): GDAL datatype. Defaults to gdal.GDT_Byte.
        attribute (str, optional): If an attribute should be burned into raster for each polygon.
            Defaults to None.
    """
    try:
        vector = ogr.Open(input_vector)
        lyr = vector.GetLayer()
        xmin, xmax, ymin, ymax = lyr.GetExtent()
    except Exception as e:
        raise Exception('Error reading input vector file') from e

    try:
        # if attribute is passed in containing a value for rater, use it
        if attribute:
            ds = gdal.Rasterize(
                output_raster,
                input_vector,
                xRes=pixel_size,
                yRes=pixel_size,
                attribute='value',
                outputBounds=[xmin, ymin, xmax, ymax],
                noData=no_data_value,
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],
                outputType=output_type,
            )
        # else burn in a single value for the raster (burnValues)
        else:
            ds = gdal.Rasterize(
                output_raster,
                input_vector,
                xRes=pixel_size,
                yRes=pixel_size,
                burnValues=1,
                outputBounds=[xmin, ymin, xmax, ymax],
                noData=no_data_value,
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],
                outputType=output_type,
            )

        # close file
        ds = None  # noqa: F841
    except Exception as e:
        raise Exception('Error writing raster file') from e
