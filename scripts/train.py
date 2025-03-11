import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from trainer._version import __version__
from trainer import FModel
from trainer.datasets.train_dataset import train_dataset
from trainer.datasets.utils import save_prediction_image

log = logging.getLogger(__name__)

def _set_seed(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.np_seed)
    random.seed(cfg.seed)
    
def training_loop(cfg, nn):
    dataset = train_dataset(cfg=cfg)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    
    lr = cfg.train.lr
    
    optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    
    for epoch in range(0, cfg.train.epochs + 1):
        for i, mini_batch in enumerate(dataloader, start=0):
            inputs, target = mini_batch
            pred = nn(inputs)
            
            loss = mse_loss(
                input=pred,
                target=target,
            )

            log.info("Running backpropagation")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            save_prediction_image(pred, epoch, save_dir=cfg.params.save_path / "plots", statistics=dataset.statistics["obs"], batch=i, )
            
            log.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")



@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="training_config",
)
def main(cfg: DictConfig) -> None:
    _set_seed(cfg=cfg)
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    try:
        start_time = time.perf_counter()
        nn = FModel(num_classes=1, in_channels=149, device=cfg.device)  # Dynamic = (73 * 2); Static = 3; Total = 149
        training_loop(cfg=cfg, nn=nn)
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    
    finally:
        print("Cleaning up...")
    
        total_time = time.perf_counter() - start_time
        log.info(
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
        
if __name__ == "__main__":
    print(f"Training F-Model with version: {__version__}")
    main()
