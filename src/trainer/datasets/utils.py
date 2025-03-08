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

def save_prediction_image(pred_tensor, epoch, save_dir, batch=None,):
    """
    Save prediction tensor as an image file
    
    Args:
        pred_tensor: The prediction tensor from your model
        epoch: Current epoch number
        batch: Optional batch number (if saving after each batch)
        save_dir: Directory to save prediction images
    """    
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Move tensor to CPU if needed and convert to numpy
    pred_np = pred_tensor.detach().cpu().squeeze().numpy()
    
    # Create a custom colormap for flood visualization
    # Blue gradient for water/flood
    cmap = LinearSegmentedColormap.from_list(
        'flood_cmap', [(0, 'white'), (1, 'blue')], N=256
    )
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(pred_np, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label='Flood Probability')
    plt.title(f'Prediction - Epoch {epoch}' + (f', Batch {batch}' if batch is not None else ''))
    plt.axis('off')
    
    # Save the image
    filename = f'pred_epoch_{epoch}' + (f'_batch_{batch}' if batch is not None else '') + '.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

# if __name__ == "__main__":
#     start_time = datetime.strptime("20190520 000000", "%Y%m%d %H%M%S")
#     end_time = datetime.strptime("20190602 230000", "%Y%m%d %H%M%S")
#     read(start_time, end_time)
