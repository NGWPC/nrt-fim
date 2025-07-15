import logging

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

log = logging.getLogger(__name__)


class train_dataset(TorchDataset):
    """training with NWM retrospective dataset"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Load and process zarr datasets
        so = {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}

        self.precip = xr.open_dataset(
            cfg["data_sources"]["base_pattern_precip"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        ).RAINRATE

        self.air_temp = xr.open_dataset(
            cfg["data_sources"]["base_pattern_air_temp"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        ).T2D

        self.solar_shortwave = xr.open_dataset(
            cfg["data_sources"]["base_pattern_solar_shortwave"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        ).SWDOWN

        self.ldas = xr.open_dataset(
            cfg["data_sources"]["base_pattern_ldas"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        )
        self.soil_sm_layer1 = self.ldas.SOIL_M.sel(soil_layers_stag=0)
        self.soil_sm_layer2 = self.ldas.SOIL_M.sel(soil_layers_stag=1)
        self.soil_sm_layer3 = self.ldas.SOIL_M.sel(soil_layers_stag=2)
        self.soil_sm_layer4 = self.ldas.SOIL_M.sel(soil_layers_stag=3)

        self.SNOWH = self.ldas.SNOWH

        self.ugd_runoff = self.ldas.UGDRNOFF

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return self.cfg["train"]["batch_size"]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        t_hourly = self.cfg["train"]["rho"]
        # t_3hourly = t_hourly // 3
        y_size, x_size = self.cfg["train"]["No_pixels_y"], self.cfg["train"]["No_pixels_x"]

        # Max valid end time (must be ≥ window size)
        max_end_hourly = self.precip.sizes["time"]
        max_end_3hourly = self.ldas.sizes["time"]
        max_end = min(max_end_hourly, max_end_3hourly * 3)

        min_end = t_hourly  # must allow slicing back `t_hourly` steps
        end_time_index = np.random.randint(min_end, max_end)

        max_y = self.precip.sizes["y"] - y_size
        max_x = self.precip.sizes["x"] - x_size

        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)

        return self.align_and_stack(end_time_index, y, x)

    def extract_patch(self, xr_array, time_index, y, x, t_len, y_size, x_size):
        """
        Clip xr dataset based on time and x, and y

        :param xr_array: dataset (can be input or target)
        :param time_index: time
        :param y: y
        :param x: x
        :param t_len: time winod of the dataset that we want to use
        :param y_size: number of pixels in y for clipping
        :param x_size: number of pixels in x for clipping
        :return: xarray clipped dataset
        """
        return xr_array.isel(
            time=slice(time_index, time_index + t_len),
            y=slice(y, y + y_size),
            x=slice(x, x + x_size),
        )

    def align_and_stack(self, end_time_index, y, x):
        """
        Aligns different inputs which are in different time steps, and concatenate them at the end.

        :param end_time_index: last time index when sampling
        :param y: y in the lower left corner of the sampled input image
        :param x: x in the lower left corner of the sampled input image
        :return: input and target tensors
        """
        t_hourly = self.cfg["train"]["rho"]
        t_3hourly = t_hourly // 3
        y_size, x_size = self.cfg["train"]["No_pixels_y"], self.cfg["train"]["No_pixels_x"]

        start_time_index = end_time_index - t_hourly
        # ldas_end = end_time_index // 3
        ldas_start = start_time_index // 3

        precip = self.extract_patch(self.precip, start_time_index, y, x, t_hourly, y_size, x_size)
        air_temp = self.extract_patch(self.air_temp, start_time_index, y, x, t_hourly, y_size, x_size)
        solar = self.extract_patch(self.solar_shortwave, start_time_index, y, x, t_hourly, y_size, x_size)

        sm1 = self.extract_patch(self.soil_sm_layer1, ldas_start, y, x, t_3hourly, y_size, x_size)
        sm2 = self.extract_patch(self.soil_sm_layer2, ldas_start, y, x, t_3hourly, y_size, x_size)
        sm3 = self.extract_patch(self.soil_sm_layer3, ldas_start, y, x, t_3hourly, y_size, x_size)
        sm4 = self.extract_patch(self.soil_sm_layer4, ldas_start, y, x, t_3hourly, y_size, x_size)
        snowh = self.extract_patch(self.SNOWH, ldas_start, y, x, t_3hourly, y_size, x_size)
        ugd = self.extract_patch(self.ugd_runoff, ldas_start, y, x, t_3hourly, y_size, x_size)

        input_vars = [precip, air_temp, solar, sm1, sm2, sm3, sm4, snowh]
        input_tensor = torch.cat([torch.tensor(var.values, dtype=torch.float32) for var in input_vars], dim=0)
        target_tensor = torch.tensor(ugd.values, dtype=torch.float32)

        # 🔽 Make H and W divisible by 32
        _, H, W = input_tensor.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            target_tensor = F.pad(target_tensor, (0, pad_w, 0, pad_h))

        return input_tensor, target_tensor[-1:, :, :]
