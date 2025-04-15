import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_statistics(cfg: DictConfig, inputs: dict[str, xr.Dataset | xr.DataArray | list[xr.Dataset]]) -> pd.DataFrame:
    """Creating the necessary statistics for normalizing atributes

    Parameters
    ----------
    cfg: Config
        The configuration object containing the path to the data sources.
    attributes: zarr.Group
        The zarr.Group object containing attributes.

    Returns
    -------
      pl.DataFrame: A polars DataFrame containing the statistics for normalizing attributes.
    """
    statistics_path = Path(cfg.data_sources.statistics)
    statistics_path.mkdir(exist_ok=True)
    stats_file = statistics_path / "sample_statistics_coffeyville.json"
    if stats_file.exists():
        log.info(f"Reading Statistics from file: {stats_file.name}")
        df = pd.read_csv(str(stats_file))
    else:
        json_ = {}
        for k, v in inputs.items():
            if isinstance(v, xr.Dataset):
                data = v[k].values.flatten()
            elif isinstance(v, xr.DataArray):
                data = v.values.flatten()
            else:
                msg = "Cannot determine input type"
                log.exception(msg)
                raise NotImplementedError(msg)
            json_[k] = {
                "min": np.nanmin(data),
                "max": np.nanmax(data),
                "mean": np.nanmean(data),
                "std": np.nanstd(data),
                "p10": np.nanpercentile(data, 10),
                "p90": np.nanpercentile(data, 90),
            }
        df = pd.DataFrame(json_)
        df.to_csv(str(stats_file))
    return df
