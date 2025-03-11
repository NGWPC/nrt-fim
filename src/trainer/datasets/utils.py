from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr

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
        
    ds_r = xr.open_dataset(
        base_pattern_routing,    
        backend_kwargs={
            "storage_options": so
        }, 
        engine="zarr",
        chunked_array_type="cubed"
    ).sel(
        time=slice(start_time, end_time)
    ).qSfcLatRunoff
    
    ds_sm = xr.open_dataset(
        base_pattern_sm,    
        backend_kwargs={
            "storage_options": so
        }, 
        engine="zarr",
        chunked_array_type="cubed"
    ).sel(
        time=slice(start_time, end_time)
    ).SOIL_M

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
    obs_std = statistics.loc[3]   # std is at index 3
    pred_np = ((pred_np - 1e-8) * obs_std) + obs_mean
    
    vmin = np.min(pred_np)
    vmax = np.max(pred_np)
        
    # Create a figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.4, 0.65, 1), (0.2, 0.5, 1), (0, 0.3, 0.8)]
    flood_cmap = LinearSegmentedColormap.from_list('flood_cmap', colors)
    
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
    
    cbar1.set_label('% of pixels flooded')
    
    filename = f'pred_epoch_{epoch}' + (f'_batch_{batch}' if batch is not None else '') + '.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

# if __name__ == "__main__":
#     start_time = datetime.strptime("20190520 000000", "%Y%m%d %H%M%S")
#     end_time = datetime.strptime("20190602 230000", "%Y%m%d %H%M%S")
#     read(start_time, end_time)
