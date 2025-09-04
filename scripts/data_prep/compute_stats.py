from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hydra
import numpy as np
import json

from omegaconf import DictConfig
import pandas as pd
from rasterio import open as rio_open
from trainer.utils.utils_stats import OnlineStats, InputStatsComputer
from trainer.data_prep.read_inputs import read_selected_inputs

def compute_input_stats(cfg: DictConfig) -> dict:
    inputs_dict = read_selected_inputs(cfg=cfg, compute_bounds=False)

    # time window for training
    train_start = datetime.strptime(str(cfg.train.start_time), "%Y%m%d")
    train_end   = datetime.strptime(str(cfg.train.end_time), "%Y%m%d")
    time_range = (np.datetime64(train_start), np.datetime64(train_end))

    stats_file_name = f"input_stats_{cfg.train.start_time}_{cfg.train.end_time}.json"
    stats_path = Path(cfg.project_root) / "statistics" / stats_file_name

    comp = InputStatsComputer(
        ignore_values=(255.0, 1e20),
        also_flag_large_as_nodata=True,
        large_threshold=1e19,
    )
    stats = comp.load_or_compute_from_inputs_dict(
        inputs_dict=inputs_dict,
        stats_path=stats_path,
        time_range=time_range,
        chunk_size_time=int(cfg.stats.chunk_size_time) if "stats" in cfg and "chunk_size_time" in cfg.stats else 48,
        chunk_size_y=int(cfg.stats.chunk_size_y) if "stats" in cfg and "chunk_size_y" in cfg.stats else 3840,
        chunk_size_x=int(cfg.stats.chunk_size_x) if "stats" in cfg and "chunk_size_x" in cfg.stats else 4608,
        n_bins=int(cfg.stats.n_bins) if "stats" in cfg and "n_bins" in cfg.stats else 1000,
    )
    return stats


def compute_target_stats(cfg: DictConfig, split_name: str = "train") -> dict:


    df = pd.read_csv(cfg.data_sources.index_csv)
    flood_img_paths = Path(cfg.data_sources.splits_dir, f"{split_name}.json")
    with open(flood_img_paths, "r") as f:
        flood_ids = json.load(f)
    use_ids = sorted(list(int(s) for s in flood_ids if s.strip()))
    modis_paths = df[df.flood_id.isin(use_ids)]["tif_path"].tolist()

    # Per-band streaming stats across all files, masking classic nodata
    # (kept close to your original)
    def _sanitize_1d(arr1d: np.ndarray) -> np.ndarray:
        a = arr1d.astype("float64", copy=False)
        a[a == 255.0] = np.nan
        a[a == 1e20] = np.nan
        a[np.abs(a) > 1e19] = np.nan
        a = a[np.isfinite(a)]
        return a

    # discover band count
    with rio_open(modis_paths[0]) as src0:
        band_count = src0.count

    per_band = [OnlineStats() for _ in range(band_count)]
    for p in modis_paths:
        with rio_open(p) as src:
            for b in range(1, band_count + 1):
                data = src.read(b)
                per_band[b - 1].update(_sanitize_1d(data).astype("float32"))

    out = {}
    for i, acc in enumerate(per_band, start=1):
        base = acc.finalize()
        # no histogram here; add if you want p10/p90 for targets too
        out[f"band_{i}"] = base

    stats_file_name = f"target_stats_{cfg.train.start_time}_{cfg.train.end_time}.json"
    stats_path = Path(cfg.project_root) / "statistics" / stats_file_name
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"→ Saved target stats to {stats_path}")
    return out


@hydra.main(version_base="1.3",
            config_path="../../config",
            config_name="training_config")
def main(cfg: DictConfig):
    compute_input_stats(cfg)
    compute_target_stats(cfg, "train")
    print("Stats are computed and saved.")


if __name__ == "__main__":
    main()
