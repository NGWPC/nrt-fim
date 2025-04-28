from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from geocube.api.core import make_geocube
from matplotlib.colors import LinearSegmentedColormap


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
    so = dict(anon=True, default_fill_cache=False, default_cache_type="none")

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
    """A wrapper around xarray/geocube to convert a vector file or geopandas dataframe into a raster
    where the value comes from the vector's 'attriubte'

    Args:
        input_vector: A vector dataset as a file or geopandas geodataframe
        output_raster: Output path for raster file
        resolution: resolution of cells
        attribute: column name for value to rasterize
        crs: A spatial coordinate reference system either as a pyproj class, 'ESPG:####', or as int (e.g 4326)
        dtype: Specify a numpy datatype for the raster. 
        nodata_value: Value to use for no data in raste. Defaults to None.
    """
    ds = make_geocube(input_vector, measurements=[attribute], resolution=(-resolution, resolution), output_crs=crs)

    if nodata_value is not None:
        ds[attribute] = ds[attribute].rio.write_nodata(nodata_value)

    options = {"compress": "LZW", "tiled": "TRUE", "dtype": dtype}

    ds[attribute].rio.to_raster(output_raster, **options)


# if __name__ == "__main__":
#     start_time = datetime.strptime("20190520 000000", "%Y%m%d %H%M%S")
#     end_time = datetime.strptime("20190602 230000", "%Y%m%d %H%M%S")
#     read(start_time, end_time)
