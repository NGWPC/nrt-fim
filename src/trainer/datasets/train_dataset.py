import glob
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError
from rasterio.warp import transform_bounds
from torch.utils.data import Dataset as TorchDataset

from src.trainer.datasets.stats_utils import OnlineStats
from src.trainer.datasets.utils import create_master_from_da
from src.trainer.preprocessing.MODIS import preprocess_modis

log = logging.getLogger(__name__)


class train_dataset(TorchDataset):
    """training with NWM retrospective dataset"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.OnlineStats = OnlineStats()  # to calculate statistics of the data
        # Load and process zarr datasets
        so = {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}

        # --- Load precipitation dataset and infer CRS ---
        ds_precip = xr.open_dataset(
            cfg.data_sources.base_pattern_precip,
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        )
        self.input_crs = self._infer_input_crs(ds_precip)
        self.precip = ds_precip["RAINRATE"]
        # self.precip.name = "precip"
        self.hourly_time = self.precip.time.values

        self.air_temp = xr.open_dataset(
            cfg["data_sources"]["base_pattern_air_temp"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        ).T2D
        # self.air_temp.name = "air_temp"

        self.solar_shortwave = xr.open_dataset(
            cfg["data_sources"]["base_pattern_solar_shortwave"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        ).SWDOWN
        # self.solar_shortwave.name = "solar_shortwave"

        self.ldas = xr.open_dataset(
            cfg["data_sources"]["base_pattern_ldas"],
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        )
        self.soil_sm_layer1 = self.ldas.SOIL_M.sel(soil_layers_stag=0)
        self.soil_sm_layer1.name = (
            "soil_sm_layer1"  # the default name for all soil layers was the same.changing it here
        )
        self.soil_sm_layer2 = self.ldas.SOIL_M.sel(soil_layers_stag=1)
        self.soil_sm_layer2.name = "soil_sm_layer2"
        self.soil_sm_layer3 = self.ldas.SOIL_M.sel(soil_layers_stag=2)
        self.soil_sm_layer3.name = "soil_sm_layer3"
        self.soil_sm_layer4 = self.ldas.SOIL_M.sel(soil_layers_stag=3)
        self.soil_sm_layer4.name = "soil_sm_layer4"
        self.three_hourly_time = self.soil_sm_layer1.time.values

        self.SNOWH = self.ldas.SNOWH
        # self.SNOWH.name = "SNOWH"

        self.ugd_runoff = self.ldas.UGDRNOFF
        # self.ugd_runoff.name = "ugd_runoff"

        # create a master grid from inputs
        # build the master‐grid path
        self.master_path = os.path.join(
            self.cfg["master_grids_dir"], f"master_{self.cfg['params']['resolution']}m.tif"
        )
        if not os.path.exists(self.master_path):
            create_master_from_da(
                da=self.precip, master_path=self.master_path, resolution=self.cfg["params"]["resolution"]
            )

        # --- Load MODIS flood extent TIFFs for each event ---
        modis_dir = cfg.data_sources.dfo_modis_dir
        pattern = os.path.join(modis_dir, "DFO_*", "*.tif")
        all_paths = sorted(glob.glob(pattern))
        if not all_paths:
            raise FileNotFoundError(f"No MODIS TIFF files found in {modis_dir}")

        # --- Compute and store bounding boxes keyed by floodID ---
        self.raw_modis_bboxes = self._compute_modis_bboxes(all_paths, self.input_crs)

        # --- Filter and assign modis_paths internally ---
        self.raw_modis_paths = self._filter_modis_paths(all_paths, self.raw_modis_bboxes)

        # pre-processing the satellite data with scripts/flood_percent_raster.py CLI, to get percentage and to regrid
        print("starting satellite images pre-prcoessing!")
        self.MODIS_paths = preprocess_modis(self.cfg, self.raw_modis_paths[:5], grid=self.master_path)
        print("satellite images pre-prcoessing done!")

        # --- Compute and store bounding boxes keyed by floodID ---
        self.modis_bboxes = self._compute_modis_bboxes(self.MODIS_paths, self.input_crs)
        self.flood_ids = list(self.modis_bboxes.keys())
        event_ending_time_str = [Path(p).name.split("_to_")[1][:8] for p in self.MODIS_paths]
        self.event_ending_time = [
            np.datetime64(datetime.strptime(event_str, "%Y%m%d")) for event_str in event_ending_time_str
        ]

        # Calculate Statistics:
        # 1) Input stats
        self.input_stats_path = Path(Path(__file__).parents[3], "Statistics", "input_stats.json")
        self.target_stats_path = Path(Path(__file__).parents[3], "Statistics", "target_stats.json")
        train_start_dt = datetime.strptime(str(self.cfg["train"]["start_time"]), "%Y%m%d")
        train_end_dt = datetime.strptime(str(self.cfg["train"]["end_time"]), "%Y%m%d")
        train_start = pd.to_datetime(train_start_dt).to_pydatetime()
        train_end = pd.to_datetime(train_end_dt).to_pydatetime()
        time_range = (train_start, train_end)

        ds_inputs = [
            self.precip,
            self.air_temp,
            self.solar_shortwave,
            self.soil_sm_layer1,
            self.soil_sm_layer2,
            self.soil_sm_layer3,
            self.soil_sm_layer4,
            self.SNOWH,
            self.ugd_runoff,
        ]
        # 1) Compute( or load) input stats
        self.input_stats = self.OnlineStats.load_or_compute_input_stats_3d(
            ds_inputs=ds_inputs,
            stats_path=self.input_stats_path,
            time_range=time_range,
            chunk_size_time=48,
            chunk_size_y=2048,
            chunk_size_x=2048,
            n_bins=1000,
        )

        # 2) Target stats
        self.target_stats = self.OnlineStats.load_or_compute_target_stats(
            self.MODIS_paths, stats_path=self.target_stats_path
        )

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return self.cfg["train"]["batch_size"]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        real_i = idx % len(self.MODIS_paths)
        sat_img_path = self.MODIS_paths[real_i]
        sat_img_bbox = self.modis_bboxes[self.flood_ids[real_i]]
        sat_img = rioxarray.open_rasterio(sat_img_path)
        sat_img = sat_img[0]  # first band is the flooded area band

        y_size, x_size = self.cfg["train"]["No_pixels_y"], self.cfg["train"]["No_pixels_x"]
        # maximum valid offsets so you don't run off the edge
        max_y = sat_img.sizes["y"] - y_size
        max_x = sat_img.sizes["x"] - x_size

        y0 = int(np.random.randint(0, max_y + 1))
        x0 = int(np.random.randint(0, max_x + 1))

        flood_patch = sat_img.isel(y=slice(y0, y0 + y_size), x=slice(x0, x0 + x_size))

        event_ending_time = self.event_ending_time[real_i]
        end_hourly_time_index = np.argmin(np.abs(self.hourly_time - event_ending_time))
        end_3hourly_time_index = np.argmin(np.abs(self.three_hourly_time - event_ending_time))
        t_hourly = self.cfg["train"]["rho"]
        t_3hourly = t_hourly // 3
        start_hourly_time_index = end_hourly_time_index - t_hourly
        start_3hourly_time_index = end_3hourly_time_index - t_3hourly

        inputs_vars = []
        # inputs with hourly time step
        for arr in (self.precip, self.air_temp, self.solar_shortwave):
            input_patch = self.extract_patch(
                arr,
                flood_patch,
                start_hourly_time_index,
                end_hourly_time_index,
            )
            inputs_vars.append(input_patch)

        # inputs with 3-hourly time step
        for arr in (
            self.soil_sm_layer1,
            self.soil_sm_layer2,
            self.soil_sm_layer3,
            self.soil_sm_layer4,
            self.SNOWH,
        ):
            input_patch = self.extract_patch(
                arr,
                flood_patch,
                start_3hourly_time_index,
                end_3hourly_time_index,
            )
            inputs_vars.append(input_patch)


        # 4) Stack into input tensor
        input_tensor = torch.tensor(
            np.concatenate(inputs_vars, axis=0),  # (channels, y_size, x_size)
            dtype=torch.float32,
        )
        # 5) Build target tensor
        target_tensor = torch.tensor(
            flood_patch.values[np.newaxis, ...],  # (1, y_size, x_size)
            dtype=torch.float32,
        )

        input_tensor, target_tensor = self.collate_fn(input_tensor, target_tensor)

        return input_tensor, target_tensor

    def extract_patch(
        self,
        input_array: xr.DataArray,
        master_patch: xr.DataArray,
        start_time_index: int,
        end_time_index: int,
        # edge_and_size: tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract a spatio‐temporal patch:

          - time slice from start_time_index to end_time_index inclusive
          - spatial window defined by (x0,y0,x_size,y_size), aligned to master_patch

        Returns
        -------
            matched.values: a numpy array of shape (time, y_size, x_size)
        """
        # x0, y0, x_size, y_size = edge_and_size

        # 1) Temporal slice
        temp = input_array.isel(time=slice(start_time_index, end_time_index))
        temp = temp.rio.write_crs(self.input_crs, inplace=False)

        # 2) get master_patch bounds & CRS
        mp_crs = master_patch.rio.crs
        minx, miny, maxx, maxy = master_patch.rio.bounds()
        # minx, maxy = master_patch.rio.transform() * (x0, y0)
        # maxx, miny = master_patch.rio.transform() * (x0 + x_size, y0 + y_size)

        bx = transform_bounds(mp_crs, self.input_crs, minx, miny, maxx, maxy)
        bx_minx, bx_miny, bx_maxx, bx_maxy = bx

        # 4) clip + reproject to match satellite grid
        clipped = temp.rio.clip_box(
            minx=bx_minx, miny=bx_miny, maxx=bx_maxx, maxy=bx_maxy, crs=self.input_crs
        )

        matched = clipped.rio.reproject_match(master_patch, resampling=Resampling.bilinear)

        return matched.values

    def collate_fn(self, input_tensor, target_tensor):
        """
        Aligns different inputs which are in different time steps, and concatenate them at the end.

        :param end_time_index: last time index when sampling
        :param y: y in the lower left corner of the sampled input image
        :param x: x in the lower left corner of the sampled input image
        :return: input and target tensors
        """
        # 🔽 Make H and W divisible by 32
        _, H, W = input_tensor.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            target_tensor = F.pad(target_tensor, (0, pad_w, 0, pad_h))

        return input_tensor, target_tensor[-1:, :, :]

    def _filter_modis_paths(self, paths, bboxes) -> list:
        """Filter flood events outside CONUS using computed bboxes, log excluded IDs, and return sorted list of valid paths."""
        filtered = []
        excluded = []
        for path in paths:
            floodid = self._extract_floodid(path)
            if floodid in bboxes:
                filtered.append(path)
            else:
                excluded.append(floodid)
        if excluded:
            log.info(f"Excluding flood events outside CONUS: {sorted(set(excluded))}")
        if not filtered:
            raise RuntimeError("No MODIS flood events remain within CONUS bounds.")
        return sorted(filtered)

    def _compute_modis_bboxes(self, paths, input_crs: str) -> dict:
        """Read each MODIS TIFF, transform its bounds to the input CRS, and return a dict mapping floodID to (left, bottom, right, top) for CONUS events. Accept any partial overlap with the USA CONUS area."""
        # CONUS bounding box in WGS84 (lon/lat)
        conus_wgs84 = (-124.848974, 24.396308, -66.885444, 49.384358)
        geog_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_string(input_crs)
        # Transform CONUS bounds to target CRS if needed
        if geog_crs != target_crs:
            conus_bounds = transform_bounds(geog_crs, target_crs, *conus_wgs84)
        else:
            conus_bounds = conus_wgs84
        bboxes = {}
        for path in paths:
            # extract floodID from directory name
            floodid = self._extract_floodid(path)
            with rasterio.open(path) as src:
                left, bottom, right, top = src.bounds
                src_crs = src.crs
            dst_bounds = (left, bottom, right, top)
            if src_crs != target_crs:
                # reproject bounds if needed
                dst_bounds = transform_bounds(src_crs, target_crs, left, bottom, right, top)
            # Accept any partial intersection with CONUS
            if not (
                dst_bounds[2] < conus_bounds[0]
                or dst_bounds[0] > conus_bounds[2]
                or dst_bounds[3] < conus_bounds[1]
                or dst_bounds[1] > conus_bounds[3]
            ):
                bboxes[floodid] = dst_bounds
        return bboxes

    def _extract_floodid(self, path: str) -> str:
        """Parse the floodID from a path of form .../DFO_<floodID>_<start>_<end>/file.tif"""
        dir_name = os.path.basename(os.path.dirname(path))
        parts = dir_name.split("_")
        return parts[1] if len(parts) > 1 else dir_name

    def _infer_input_crs(self, ds_precip: xr.Dataset) -> str:
        """
        Infer the input CRS from the precipitation dataset's 'crs' variable metadata.

        Returns a PROJ string or defaults to 'EPSG:4326'.
        """
        spatial_ref = None
        if "crs" in ds_precip:
            crs_da = ds_precip["crs"]
            spatial_ref = crs_da.attrs.get("spatial_ref")
        if spatial_ref:
            try:
                parsed_crs = CRS.from_wkt(spatial_ref)
                return parsed_crs.to_string()
            except (CRSError, ValueError) as e:
                log.warning(f"Failed to parse CRS WKT due to {e.__class__.__name__}; defaulting to EPSG:4326")
        return "EPSG:4326"
