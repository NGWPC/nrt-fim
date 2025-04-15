import logging
from pathlib import Path

import numpy as np
import rioxarray
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset
from tqdm import trange

from trainer.datasets.statistics import get_statistics

log = logging.getLogger(__name__)


class train_dataset(TorchDataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Load and process zarr datasets
        self.runoff = xr.open_dataset(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/router.zarr"),
            engine="zarr",
            chunked_array_type="cubed",
        )
        self.SOIL_M = xr.open_dataset(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/soil_moisture.zarr"),
            engine="zarr",
            chunked_array_type="cubed",
        )

        self.obs = rioxarray.open_rasterio(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/flood_percent_250m.tif"),
            chunked_array_type="cubed",
        )
        self.flow_acc = rioxarray.open_rasterio(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/flow_acc_250m.tif"),
            chunked_array_type="cubed",
        )
        self.flow_dir = rioxarray.open_rasterio(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/flow_dir_250m.tif"),
            chunked_array_type="cubed",
        )
        self.surface_extent = rioxarray.open_rasterio(
            Path("/Users/taddbindas/projects/NGWPC/f1_trainer/data/sample_dir/surface_extent_250m.tif"),
            chunked_array_type="cubed",
        )

        self.statistics = get_statistics(
            self.cfg,
            {
                "runoff": self.runoff,
                "SOIL_M": self.SOIL_M,
                "obs": self.obs,
                "flow_acc": self.flow_acc,
                "flow_dir": self.flow_dir,
                "surface_extent": self.surface_extent,
            },
        )

        static_inputs = self.process_static_inputs()
        runoff_inputs = self.process_dynamic_inputs(self.runoff, "runoff", self.statistics["runoff"])
        soil_moisture_inputs = self.process_dynamic_inputs(self.SOIL_M, "SOIL_M", self.statistics["SOIL_M"])
        self.observations = self.process_obs(self.statistics["obs"])
        self.inputs = torch.cat([static_inputs, runoff_inputs, soil_moisture_inputs], dim=0)

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def process_obs(self, statistics):
        # Squeeze to 2D space
        obs_array = self.obs.values.squeeze()

        obs_mean = statistics.loc[2]  # mean is at index 2
        obs_std = statistics.loc[3]  # std is at index 3
        obs_norm = (obs_array - obs_mean) / (obs_std + 1e-8)

        obs_norm = np.nan_to_num(obs_norm, nan=0.0)
        obs_np_array = np.array(obs_norm)
        return torch.tensor(obs_np_array, dtype=torch.float32, device=self.cfg.device)

    def process_dynamic_inputs(self, input_var, name, statistics):
        arr = []
        for i in trange(input_var.sizes["time"], desc="Normalizing dynamic inputs", ascii=True, ncols=80):
            data = input_var.isel(time=i)
            # data = data.rio.to_crs(self.obs.rio.crs)
            matched_data = data.rio.reproject_match(self.obs)
            input_data = matched_data[name].values.squeeze()
            input_data_mean = statistics.loc[2]  # mean is at index 2
            input_data_std = statistics.loc[3]  # std is at index 3
            input_data_norm = (input_data - input_data_mean) / (input_data_std + 1e-8)
            input_data_norm = np.nan_to_num(input_data_norm, nan=0.0)
            arr.append(input_data_norm)
        data_array = np.array(arr)
        return torch.tensor(data_array, dtype=torch.float32, device=self.cfg.device)

    def process_static_inputs(self):
        flow_acc_match = self.flow_acc.rio.reproject_match(self.obs)
        flow_dir_match = self.flow_dir.rio.reproject_match(self.obs)
        surface_extent_match = self.surface_extent.rio.reproject_match(self.obs)

        # Squeeze to 2D space
        flow_acc_array = flow_acc_match.values.squeeze()
        flow_dir_array = flow_dir_match.values.squeeze()
        surface_extent_array = surface_extent_match.values.squeeze()

        # Mean/std normalization for all features using self.statistics
        # Flow accumulation normalization
        flow_acc_mean = self.statistics.loc[2, "flow_acc"]  # mean is at index 2
        flow_acc_std = self.statistics.loc[3, "flow_acc"]  # std is at index 3
        flow_acc_norm = (flow_acc_array - flow_acc_mean) / (flow_acc_std + 1e-8)

        # Flow direction normalization
        flow_dir_mean = self.statistics.loc[2, "flow_dir"]
        flow_dir_std = self.statistics.loc[3, "flow_dir"]
        flow_dir_norm = (flow_dir_array - flow_dir_mean) / (flow_dir_std + 1e-8)

        # Surface extent normalization
        surface_extent_mean = self.statistics.loc[2, "surface_extent"]
        surface_extent_std = self.statistics.loc[3, "surface_extent"]
        surface_extent_norm = (surface_extent_array - surface_extent_mean) / (surface_extent_std + 1e-8)

        # Replace NaNs with zeros
        flow_acc_norm = np.nan_to_num(flow_acc_norm, nan=0.0)
        flow_dir_norm = np.nan_to_num(flow_dir_norm, nan=0.0)
        surface_extent_norm = np.nan_to_num(surface_extent_norm, nan=0.0)

        input_channels = []

        # Add static features
        input_channels.append(flow_acc_norm)
        input_channels.append(flow_dir_norm)
        input_channels.append(surface_extent_norm)
        input_array = np.array(input_channels)
        return torch.tensor(input_array, dtype=torch.float32, device=self.cfg.device)

    def collate_fn(self, _):
        """
        Process and normalize raster data for UNet input with proper dimensioning for 250m resolution data

        Returns
        -------
            tuple: (input_tensor, target_tensor)
        """
        input_tensor = self.inputs  # Shape: [149, 477, 240] (channels, height, width)

        # Get target tensor from observations (assuming this is the obs raster)
        # Extract data and ensure it's properly shaped
        target_tensor = self.observations.unsqueeze(0)

        # Make dimensions divisible by 32 (for 5 encoder/decoder blocks)
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            # Pad inputs and targets
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h))
            target_tensor = torch.nn.functional.pad(target_tensor, (0, pad_w, 0, pad_h))

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)  # [1, channels, H, W]
        target_tensor_resized = F.interpolate(
            target_tensor.unsqueeze(0),  # Add batch dim
            size=(240, 128),
            mode="bilinear",
            align_corners=True,
        )  # Remove extra batch dim

        return input_tensor, target_tensor_resized
