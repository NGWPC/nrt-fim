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

from trainer import FModel
from trainer._version import __version__
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


def evaluate(cfg, nn):
    """
    Main loop for training nn

    :param cfg: Configuration file
    :param nn: neural network defined in main
    :return: None
    """
    dataset = train_dataset(cfg=cfg, mode="test")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
    )

    if cfg.experiment.checkpoint:
        file_path = Path(cfg.experiment.checkpoint)
        device = torch.device(cfg.device)
        log.info(f"Loading nn from checkpoint: {file_path.stem}")
        state = torch.load(file_path, map_location=device)
        state_dict = state["model_state_dict"]
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        nn.load_state_dict(state_dict)

    else:
        log.warning("Creating new model for evaluation.")

    with torch.no_grad():  # Disable gradient calculations during evaluation
        nn.eval()

        for i, mini_batch in enumerate(dataloader, start=0):
            inputs, target = mini_batch
            inputs, target = inputs.to(cfg.device), target.to(cfg.device)
            pred = nn(inputs)

            save_prediction_image(
                pred,
                epoch=0,
                save_dir=cfg.params.save_path / f"output_mb_{i}",
                statistics=dataset.statistics["obs"],
                batch=i,
            )


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="training_config",
)
def main(cfg: DictConfig) -> None:
    """
    Using configuration file, setting up randomseeds, building nn, and looping through it.

    :param cfg: Configuration file
    :return: None
    """
    _set_seed(cfg=cfg)
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    start_time = time.perf_counter()
    try:
        nn = FModel(
            num_classes=1, in_channels=336, device=cfg.device
        )  # Dynamic = (73 * 2); Static = 3; Total = 149
        evaluate(cfg=cfg, nn=nn)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    finally:
        print("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Evaluating F-Model with version: {__version__}")
    main()
