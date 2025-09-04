from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple

import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window


# -----------------------------
# Online, streaming statistics
# -----------------------------
class OnlineStats:
    """
    Streaming mean/var/min/max + percentile estimation via 2-pass histogram.

    Ignores NaN/Inf and user-provided sentinel values.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = np.inf
        self.max = -np.inf

    def _sanitize_1d(
        self,
        arr1d: np.ndarray,
        ignore_values: Optional[Iterable[float]] = (255.0, 1e20),
        also_flag_large_as_nodata: bool = True,
        large_threshold: float = 1e19,
    ) -> np.ndarray:
        """Return finite values only, masking common nodata sentinels."""
        a = arr1d.astype("float64", copy=False)
        if ignore_values:
            for v in ignore_values:
                a[a == v] = np.nan
        if also_flag_large_as_nodata:
            a[np.abs(a) > large_threshold] = np.nan
        a = a[np.isfinite(a)]
        return a

    def update(
        self,
        data: np.ndarray,
        ignore_values: Optional[Iterable[float]] = (255.0, 1e20),
        also_flag_large_as_nodata: bool = True,
        large_threshold: float = 1e19,
    ):
        """Update running stats with one block."""
        flat = self._sanitize_1d(
            data.ravel(),
            ignore_values=ignore_values,
            also_flag_large_as_nodata=also_flag_large_as_nodata,
            large_threshold=large_threshold,
        )
        if flat.size == 0:
            return

        nB = flat.size
        meanB = float(flat.mean())
        M2B = float(((flat - meanB) ** 2).sum())
        minB, maxB = float(flat.min()), float(flat.max())

        # min/max
        self.min = min(self.min, minB)
        self.max = max(self.max, maxB)

        # Welford’s update
        if self.n == 0:
            self.n, self.mean, self.M2 = nB, meanB, M2B
        else:
            delta = meanB - self.mean
            nA = self.n
            n = nA + nB
            self.mean = (nA * self.mean + nB * meanB) / n
            self.M2 = self.M2 + M2B + (delta * delta) * nA * nB / n
            self.n = n

    def finalize(self) -> Dict[str, float]:
        var = (self.M2 / self.n) if self.n > 0 else np.nan
        return {
            "count": int(self.n),
            "min": float(self.min) if np.isfinite(self.min) else np.nan,
            "max": float(self.max) if np.isfinite(self.max) else np.nan,
            "mean": float(self.mean) if self.n > 0 else np.nan,
            "std": float(np.sqrt(var)) if np.isfinite(var) else np.nan,
        }


# ---------------------------------------
# Helpers to iterate blocks from inputs
# ---------------------------------------
def _iter_blocks_dynamic_3d(
    da: xr.DataArray,
    time_range: Optional[Tuple[np.datetime64, np.datetime64]],
    chunk_t: int,
    chunk_y: int,
    chunk_x: int,
):
    """Yield ndarray blocks (t,y,x) from a 3-D DataArray with dims ('time','y','x')."""
    if time_range is not None and "time" in da.dims:
        da = da.sel(time=slice(*time_range))

    # allow single-time arrays to pass (t=1)
    if "time" not in da.dims:
        # treat as 2D, but still honor chunking in space
        ny, nx = int(da.sizes["y"]), int(da.sizes["x"])
        for y0 in range(0, ny, chunk_y):
            y1 = min(ny, y0 + chunk_y)
            for x0 in range(0, nx, chunk_x):
                x1 = min(nx, x0 + chunk_x)
                yield da.isel(y=slice(y0, y1), x=slice(x0, x1)).values[np.newaxis, ...]  # (1,y,x)
        return

    nt = int(da.sizes["time"])
    ny = int(da.sizes["y"])
    nx = int(da.sizes["x"])

    for t0 in range(0, nt, chunk_t):
        t1 = min(nt, t0 + chunk_t)
        for y0 in range(0, ny, chunk_y):
            y1 = min(ny, y0 + chunk_y)
            for x0 in range(0, nx, chunk_x):
                x1 = min(nx, x0 + chunk_x)
                block = da.isel(time=slice(t0, t1), y=slice(y0, y1), x=slice(x0, x1)).values
                yield block  # (t,y,x)


def _iter_blocks_static_2d(
    path: str | Path,
    chunk_y: int,
    chunk_x: int,
    band: int = 1,
):
    """Yield ndarray blocks (y,x) from a GeoTIFF using raster windows."""
    with rasterio.open(path) as src:
        ny, nx = src.height, src.width
        for y0 in range(0, ny, chunk_y):
            h = min(chunk_y, ny - y0)
            for x0 in range(0, nx, chunk_x):
                w = min(chunk_x, nx - x0)
                arr = src.read(band, window=Window(x0, y0, w, h))
                yield arr  # (h,w)


# ----------------------------------------------------
# Core: compute (or load) stats from inputs_dict
# ----------------------------------------------------
class InputStatsComputer:
    """
    Computes per-variable stats for:
      - Dynamic vars in inputs_dict["dyn_vars"]: xr.DataArray with ('time','y','x')
      - Static rasters in inputs_dict["static_paths"]: GeoTIFFs

    Produces a dict: { var_name: {count,min,max,mean,std,p10,p90} }
    """

    def __init__(
        self,
        ignore_values: Optional[Iterable[float]] = (255.0, 1e20),
        also_flag_large_as_nodata: bool = True,
        large_threshold: float = 1e19,
    ):
        self.ignore_values = ignore_values
        self.also_flag_large_as_nodata = also_flag_large_as_nodata
        self.large_threshold = large_threshold

    def _percentiles_from_hist(self, hist: np.ndarray, bins: np.ndarray, total: int, p: float) -> float:
        cdf = np.cumsum(hist)
        idx = int(np.searchsorted(cdf, p * total))
        idx = max(0, min(idx, len(bins) - 2))  # safe in [0, n_bins-1]
        # return left bin edge as representative
        return float(bins[idx])

    def _two_pass_stats_dynamic(
        self,
        da: xr.DataArray,
        time_range: Optional[Tuple[np.datetime64, np.datetime64]],
        chunk_t: int,
        chunk_y: int,
        chunk_x: int,
        n_bins: int,
        calculate_p10_p90: bool = False,
    ) -> Dict[str, float]:
        # First pass: mean/var/min/max
        agg = OnlineStats()
        total = 0
        for block in _iter_blocks_dynamic_3d(da, time_range, chunk_t, chunk_y, chunk_x):
            # block is (t,y,x)
            agg.update(
                block,
                ignore_values=self.ignore_values,
                also_flag_large_as_nodata=self.also_flag_large_as_nodata,
                large_threshold=self.large_threshold,
            )
        base = agg.finalize()
        total = agg.n

        if calculate_p10_p90:
            # Second pass: histogram for p10/p90
            if total > 0 and np.isfinite(base["min"]) and np.isfinite(base["max"]):
                minv, maxv = base["min"], base["max"]
                if minv == maxv:
                    # degenerate
                    base["p10"] = minv
                    base["p90"] = maxv
                    return base

                bins = np.linspace(minv, maxv, n_bins + 1)
                hist = np.zeros(n_bins, dtype=np.int64)

                for block in _iter_blocks_dynamic_3d(da, time_range, chunk_t, chunk_y, chunk_x):
                    flat = block.ravel().astype("float64")
                    # sanitize
                    if self.ignore_values:
                        for v in self.ignore_values:
                            flat[flat == v] = np.nan
                    if self.also_flag_large_as_nodata:
                        flat[np.abs(flat) > self.large_threshold] = np.nan
                    flat = flat[np.isfinite(flat)]
                    if flat.size:
                        h, _ = np.histogram(flat, bins=bins)
                        hist += h

                base["p10"] = self._percentiles_from_hist(hist, bins, total, 0.10)
                base["p90"] = self._percentiles_from_hist(hist, bins, total, 0.90)
            else:
                base["p10"] = np.nan
                base["p90"] = np.nan
        return base

    def _two_pass_stats_static(
        self,
        path: str | Path,
        chunk_y: int,
        chunk_x: int,
        n_bins: int,
        calculate_p10_p90: bool = False,
    ) -> Dict[str, float]:
        # First pass
        agg = OnlineStats()
        for block in _iter_blocks_static_2d(path, chunk_y, chunk_x, band=1):
            agg.update(
                block,
                ignore_values=self.ignore_values,
                also_flag_large_as_nodata=self.also_flag_large_as_nodata,
                large_threshold=self.large_threshold,
            )
        base = agg.finalize()
        total = agg.n

        if calculate_p10_p90 == True:
            # Second pass
            if total > 0 and np.isfinite(base["min"]) and np.isfinite(base["max"]):
                minv, maxv = base["min"], base["max"]
                if minv == maxv:
                    base["p10"] = minv
                    base["p90"] = maxv
                    return base

                bins = np.linspace(minv, maxv, n_bins + 1)
                hist = np.zeros(n_bins, dtype=np.int64)

                for block in _iter_blocks_static_2d(path, chunk_y, chunk_x, band=1):
                    flat = block.ravel().astype("float64")
                    if self.ignore_values:
                        for v in self.ignore_values:
                            flat[flat == v] = np.nan
                    if self.also_flag_large_as_nodata:
                        flat[np.abs(flat) > self.large_threshold] = np.nan
                    flat = flat[np.isfinite(flat)]
                    if flat.size:
                        h, _ = np.histogram(flat, bins=bins)
                        hist += h

                base["p10"] = self._percentiles_from_hist(hist, bins, total, 0.10)
                base["p90"] = self._percentiles_from_hist(hist, bins, total, 0.90)
            else:
                base["p10"] = np.nan
                base["p90"] = np.nan
        return base

    def compute_from_inputs_dict(
        self,
        inputs_dict: Dict[str, Any],
        time_range: Optional[Tuple[np.datetime64, np.datetime64]] = None,
        chunk_size_time: int = 24,
        chunk_size_y: int = 512,
        chunk_size_x: int = 512,
        n_bins: int = 1000,
        calculate_p10_p90: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute stats for everything in inputs_dict the user selected:
          - inputs_dict["dyn_vars"]: {name -> xr.DataArray}
          - inputs_dict["static_paths"]: {name -> path}

        Returns: { name -> stats_dict }
        """
        out: Dict[str, Dict[str, float]] = {}

        # dynamic
        for name, da in (inputs_dict.get("dyn_vars") or {}).items():
            # make sure it has expected dims
            if not all(dim in da.dims for dim in ("y", "x")):
                raise ValueError(f"Dynamic var '{name}' missing required dims (y,x). Got dims={da.dims}")
            out[name] = self._two_pass_stats_dynamic(
                da=da,
                time_range=time_range,
                chunk_t=chunk_size_time,
                chunk_y=chunk_size_y,
                chunk_x=chunk_size_x,
                n_bins=n_bins,
                calculate_p10_p90=calculate_p10_p90
            )

        # statics
        for name, path in (inputs_dict.get("static_paths") or {}).items():
            out[name] = self._two_pass_stats_static(
                path=path,
                chunk_y=chunk_size_y,
                chunk_x=chunk_size_x,
                n_bins=n_bins,
                calculate_p10_p90=calculate_p10_p90
            )

        return out

    def load_or_compute_from_inputs_dict(
        self,
        inputs_dict: Dict[str, Any],
        stats_path: str | Path,
        time_range: Optional[Tuple[np.datetime64, np.datetime64]] = None,
        chunk_size_time: int = 24,
        chunk_size_y: int = 512,
        chunk_size_x: int = 512,
        n_bins: int = 1000,
        calculate_p10_p90: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # names we expect (dynamic + static that user actually selected)
        expected_names = set()
        expected_names |= set((inputs_dict.get("dyn_vars") or {}).keys())
        expected_names |= set((inputs_dict.get("static_paths") or {}).keys())

        if stats_path.exists():
            with open(stats_path) as f:
                saved = json.load(f)
            if expected_names and all(n in saved for n in expected_names):
                print(f"✔ Loaded input stats from {stats_path}")
                return saved
            else:
                print(f"⚠ Incomplete stats in {stats_path}. Recomputing…")

        stats = self.compute_from_inputs_dict(
            inputs_dict=inputs_dict,
            time_range=time_range,
            chunk_size_time=chunk_size_time,
            chunk_size_y=chunk_size_y,
            chunk_size_x=chunk_size_x,
            n_bins=n_bins,
            calculate_p10_p90=calculate_p10_p90
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"→ Saved input stats to {stats_path}")
        return stats

