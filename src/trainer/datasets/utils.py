import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import xarray as xr
from geocube.api.core import make_geocube
from matplotlib.colors import LinearSegmentedColormap
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_origin

log = logging.getLogger(__name__)


def read(start_time: datetime, end_time: datetime):
    """Reading the zarr stores for Routing and Soil moisture

    Parameters
    ----------
    start_time : datetime
        start time for slicing the zarr store
    end_time : datetime
        end time for slicing the zarr store

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Routing and Soil moisture datasets
    """
    so = {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}

    base_pattern_routing = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr/"
    base_pattern_sm = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr/"

    ds_r = (
        xr.open_dataset(
            base_pattern_routing,
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        )
        .sel(time=slice(start_time, end_time))
        .qSfcLatRunoff
    )

    ds_sm = (
        xr.open_dataset(
            base_pattern_sm, backend_kwargs={"storage_options": so}, engine="zarr", chunked_array_type="cubed"
        )
        .sel(time=slice(start_time, end_time))
        .SOIL_M
    )

    return ds_r, ds_sm


def save_prediction_image(pred_tensor, epoch, save_dir, statistics, batch=None):
    """
    Plot denormalized prediction with proper colorbar for flood visualization

    Args:
        prediction: Normalized prediction array/tensor
        mean: Mean value used in normalization
        std: Standard deviation used in normalization
        threshold: Threshold value for considering a pixel as flooded
        figsize: Figure size tuple
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    pred_np = pred_tensor.detach().cpu().squeeze().numpy()

    obs_mean = statistics.loc[2]  # mean is at index 2
    obs_std = statistics.loc[3]  # std is at index 3
    pred_np = ((pred_np - 1e-8) * obs_std) + obs_mean

    vmin = np.min(pred_np)
    vmax = np.max(pred_np)

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    colors = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.4, 0.65, 1), (0.2, 0.5, 1), (0, 0.3, 0.8)]
    flood_cmap = LinearSegmentedColormap.from_list("flood_cmap", colors)

    # Plot 1: Denormalized prediction with proper colorbar
    im1 = ax1.imshow(pred_np, cmap=flood_cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Coffeyville Flood Extent [Epoch: {epoch}]")

    cbar1 = fig.colorbar(im1, ax=ax1)

    # Calculate appropriate tick locations
    # Create 5 evenly spaced ticks from vmin to vmax
    tick_count = 5
    ticks = np.linspace(vmin, vmax, tick_count)
    cbar1.set_ticks(ticks)

    # Format tick labels to have consistent decimal places
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    cbar1.set_ticklabels(tick_labels)

    cbar1.set_label("% of pixels flooded")

    filename = f"pred_epoch_{epoch}" + (f"_batch_{batch}" if batch is not None else "") + ".png"
    plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def rasterize(
    input_vector: str | Path | gpd.GeoDataFrame,
    output_raster: str | Path,
    resolution: int | float,
    attribute: str,
    crs: pyproj.CRS | str | int,
    dtype: np.dtype | str,
    nodata_value: int | float = None,
) -> None:
    """
    A wrapper around xarray/geocube to convert a vector file or geopandas dataframe into a raster where the value comes from the vector's 'attribute'.

    Args:
    :param input_vector: A vector dataset as a file or geopandas GeoDataFrame.
    :param output_raster: Output path for the raster file.
    :param resolution: Resolution of cells.
    :param attribute: Column name for the value to rasterize.
    :param crs: A spatial coordinate reference system either as a pyproj object, 'EPSG:####' string, or integer (e.g., 4326).
    :param dtype: Numpy data type for the raster.
    :param nodata_value:
    :return: Value to use for no data in the raster. Defaults to None.
    """
    ds = make_geocube(
        input_vector, measurements=[attribute], resolution=(-resolution, resolution), output_crs=crs
    )

    if nodata_value is not None:
        ds[attribute] = ds[attribute].rio.write_nodata(nodata_value)

    options = {"compress": "LZW", "tiled": "TRUE", "dtype": dtype}

    ds[attribute].rio.to_raster(output_raster, **options)


# if __name__ == "__main__":
#     start_time = datetime.strptime("20190520 000000", "%Y%m%d %H%M%S")
#     end_time = datetime.strptime("20190602 230000", "%Y%m%d %H%M%S")
#     read(start_time, end_time)


def create_master_from_da(
    da,
    master_path: str,
    resolution: float = 250.0,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "lzw",
):
    """
    Create a blank GeoTIFF grid at `resolution` covering the full extent of `da`.
    - da must have .x and .y coordinates in projected units (e.g. meters).
    - da must carry its CRS either in da.rio.crs, da.attrs['esri_pe_string'], or da.attrs['proj4'].
    """
    # 1) grab spatial bounds
    left = float(da.x.min())
    right = float(da.x.max())
    bottom = float(da.y.min())
    top = float(da.y.max())

    # 2) compute output size
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))

    # 3) build Affine transform (upper-left corner)
    transform = from_origin(left, top, resolution, resolution)

    # 4) detect CRS
    crs = None

    # 4a) try rioxarray accessor
    try:
        crs = da.rio.crs
    except Exception:
        crs = None

    # 4b) fallback to ESRI WKT (esri_pe_string or spatial_ref)
    if crs is None:
        wkt = da.attrs.get("esri_pe_string") or da.attrs.get("spatial_ref")
        if wkt:
            try:
                crs = CRS.from_wkt(wkt)
            except Exception as e:
                log.warning(f"Failed to parse WKT CRS: {e}; will try PROJ4 next")

    # 4c) fallback to PROJ4 string
    if crs is None:
        proj4 = da.attrs.get("proj4")
        if proj4:
            try:
                crs = CRS.from_string(proj4)
            except Exception as e:
                log.warning(f"Failed to parse PROJ4 CRS: {e}; will default to EPSG:4326")

    # 4d) final default
    if crs is None:
        log.warning("No CRS found on DataArray; defaulting to EPSG:4326")
        crs = CRS.from_epsg(4326)

    # 5) write blank raster
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata,
        "compress": compress,
    }

    with rasterio.open(master_path, "w", **profile) as dst:
        blank = np.full((height, width), nodata, dtype=np.dtype(dtype))
        dst.write(blank, 1)

    print(f"→ Master grid written to {master_path}")


def merge_rasters(input_paths: list[str], output_path: str) -> None:
    """
    Merge multiple GeoTIFFs into a single continuous raster, ensuring matching CRS, resolution, compression, and dtype (to minimize output size).

    Args:
        input_paths: List of file paths to GeoTIFFs to merge. They must share the same CRS and pixel resolution.
        output_path: Path for the merged output GeoTIFF.
    """
    import rasterio

    # Open all source datasets
    src_files = [rasterio.open(p) for p in input_paths]
    try:
        # Base metadata from the first file
        base_meta = src_files[0].meta.copy()
        # base_crs = base_meta["crs"]
        # base_res = (base_meta["transform"].a, -base_meta["transform"].e)

        # Inherit compression, predictor, and dtype
        out_compress = base_meta.get("compress")
        out_predictor = base_meta.get("predictor")
        out_dtype = base_meta.get("dtype")
        # If inputs are uncompressed, default to LZW for output to save space
        if not out_compress:
            out_compress = "lzw"
            out_predictor = 2  # good for float data

        # Validate CRS and resolution match across inputs

        mosaic, out_trans = merge(src_files)

        # Update metadata for output
        out_meta = base_meta
        out_meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": out_dtype,
            }
        )
        if out_compress:
            out_meta["compress"] = out_compress
        if out_predictor:
            out_meta["predictor"] = out_predictor

        # Write merged raster, casting to original dtype
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic.astype(out_dtype))
    finally:
        for src in src_files:
            src.close()


if __name__ == "__main__":
    merge_rasters(
        input_paths=[
            r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_201003240000000000-0000014848.tif",
            r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_201003240000000000-0000000000.tif",
        ],
        output_path=r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_20100324_merged.tif",
    )
