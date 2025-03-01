import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

log = logging.getLogger(__name__)


class train_dataset(TorchDataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        pass

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs):
        pass
        
        # return 
