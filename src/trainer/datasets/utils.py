from datetime import datetime

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

# if __name__ == "__main__":
#     start_time = datetime.strptime("20190520 000000", "%Y%m%d %H%M%S")
#     end_time = datetime.strptime("20190602 230000", "%Y%m%d %H%M%S")
#     read(start_time, end_time)
