import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from rasterio import open as rio_open

log = logging.getLogger(__name__)


def get_statistics(
    cfg: DictConfig, inputs: dict[str, xr.Dataset | xr.DataArray | list[xr.Dataset]]
) -> pd.DataFrame:
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


# Streaming stats with percentile approximation via two-pass histogram
class OnlineStats:
    """Welford’s algorithm for streaming mean/var/min/max/p10/p90 over 1D data, ignoring NaN/Inf."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = np.inf
        self.max = -np.inf

        # (keep obs fields if you still need them separately)

    def update(self, data: np.ndarray):
        flat = data.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        nB = flat.size
        meanB = flat.mean()
        M2B = ((flat - meanB) ** 2).sum()
        minB, maxB = flat.min(), flat.max()

        # update min/max
        self.min = min(self.min, minB)
        self.max = max(self.max, maxB)

        # update mean/var via Welford
        if self.n == 0:
            self.n, self.mean, self.M2 = nB, meanB, M2B
        else:
            delta = meanB - self.mean
            nA = self.n
            n = nA + nB
            self.mean = (nA * self.mean + nB * meanB) / n
            self.M2 = self.M2 + M2B + delta**2 * nA * nB / n
            self.n = n

    def finalize(self):
        variance = (self.M2 / self.n) if self.n > 0 else np.nan
        return {
            "count": int(self.n),
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "std": float(np.sqrt(variance)),
        }

    def compute_input_stats_3d(
        self,
        ds_inputs: list[xr.DataArray],
        chunk_size_time: int = 24,
        chunk_size_y: int = 512,
        chunk_size_x: int = 512,
        time_range: tuple[np.datetime64, np.datetime64] | None = None,
        n_bins: int = 1000,
    ) -> dict:
        out: dict[str, dict] = {}
        for da in ds_inputs:
            name = da.name
            stats = OnlineStats()  # <-- fresh accumulator
            # 1) slice time if requested
            da_sliced = da.sel(time=slice(*time_range)) if time_range else da
            ntime, ny, nx = da_sliced.sizes["time"], da_sliced.sizes["y"], da_sliced.sizes["x"]

            # 2) first pass: mean/var/min/max
            for t0 in range(0, ntime, chunk_size_time):
                t1 = min(ntime, t0 + chunk_size_time)
                for y0 in range(0, ny, chunk_size_y):
                    y1 = min(ny, y0 + chunk_size_y)
                    for x0 in range(0, nx, chunk_size_x):
                        x1 = min(nx, x0 + chunk_size_x)
                        block = da_sliced.isel(
                            time=slice(t0, t1),
                            y=slice(y0, y1),
                            x=slice(x0, x1),
                        )
                        stats.update(block.values)

            base = stats.finalize()
            total = stats.n

            # 3) second pass: histogram for percentiles
            if total > 0:
                minv, maxv = stats.min, stats.max
                bins = np.linspace(minv, maxv, n_bins + 1)
                hist = np.zeros(n_bins, dtype=np.int64)

                for t0 in range(0, ntime, chunk_size_time):
                    t1 = min(ntime, t0 + chunk_size_time)
                    for y0 in range(0, ny, chunk_size_y):
                        y1 = min(ny, y0 + chunk_size_y)
                        for x0 in range(0, nx, chunk_size_x):
                            x1 = min(nx, x0 + chunk_size_x)
                            data = da_sliced.isel(
                                time=slice(t0, t1), y=slice(y0, y1), x=slice(x0, x1)
                            ).values.ravel()
                            data = data[np.isfinite(data)]
                            if data.size:
                                h, _ = np.histogram(data, bins=bins)
                                hist += h

                cdf = np.cumsum(hist)

                def pct(p):
                    idx = np.searchsorted(cdf, p * total)
                    return float(bins[min(idx, n_bins - 1)])

                base["p10"] = pct(0.10)
                base["p90"] = pct(0.90)
            else:
                base["p10"] = np.nan
                base["p90"] = np.nan

            out[name] = base

        return out

    def load_or_compute_input_stats_3d(
        self,
        ds_inputs: list[xr.DataArray],
        stats_path: str,
        time_range: tuple[np.datetime64, np.datetime64] | None = None,
        chunk_size_time: int = 24,
        chunk_size_y: int = 512,
        chunk_size_x: int = 512,
        n_bins: int = 1000,
    ) -> dict:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        names = [ds.name for ds in ds_inputs]

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            if all(n in stats for n in names):
                print(f"✔ Loaded input stats from {stats_path}")
                return stats
            print("⚠ Incomplete stats file, recomputing…")

        stats = self.compute_input_stats_3d(
            ds_inputs=ds_inputs,
            chunk_size_time=chunk_size_time,
            chunk_size_y=chunk_size_y,
            chunk_size_x=chunk_size_x,
            time_range=time_range,
            n_bins=n_bins,
        )
        # add timestamp, etc., if you like here
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"→ Saved input stats to {stats_path}")
        return stats

    def compute_target_stats(self, modis_paths: list[str]) -> dict:
        """
        Stream through a list of GeoTIFF files, one file at a time,
        and compute min, max, mean, std separately for each band.
        """
        # Open the first file just to get the band count
        with rio_open(modis_paths[0]) as src0:
            band_count = src0.count

        # Create one OnlineStats per band
        stats_per_band = [OnlineStats() for _ in range(band_count)]

        # Loop through files, updating each band's stats
        for p in modis_paths:
            with rio_open(p) as src:
                # src.read() returns array of shape (bands, height, width)
                arr = src.read().astype("float32")
            for i in range(band_count):
                stats_per_band[i].update(arr[i])

        # Finalize each band's stats into a dict
        out = {}
        for i, stats in enumerate(stats_per_band, start=1):
            out[f"band_{i}"] = stats.finalize()
        return out

    def load_or_compute_target_stats(self, modis_paths: list[str], stats_path: str) -> dict:
        """
        Load existing per-band target-stats JSON if valid (contains band_1…band_N),
        otherwise compute via compute_target_stats() and save to stats_path.
        """
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        # Try loading existing JSON
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            # Validate: must contain at least one 'band_1' key
            if any(key.startswith("band_") for key in stats):
                print(f"✔ Loaded target stats from {stats_path}")
                return stats
            else:
                print(f"⚠ Incomplete target stats in {stats_path}, recomputing...")

        # Compute and save
        stats = self.compute_target_stats(modis_paths)
        # Optionally stamp with timestamp
        # stats["_computed_at"] = np.datetime_as_string(np.datetime64("now"), unit="s")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"→ Saved target stats to {stats_path}")
        return stats
