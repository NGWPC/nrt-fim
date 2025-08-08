import logging
import os
import shutil
from pathlib import Path

from scripts.flood_percent_raster import generate_flood_percent

log = logging.getLogger(__name__)


def preprocess_modis(cfg, raw_MODIS_paths, grid=None):
    """For each raw MODIS TIFF in self.modis_paths, generate the flood‐percent raster by calling the `generate_flood_percent` click command. Skip any that already exist."""
    resolution = cfg["params"]["resolution"]
    overwrite_flag = cfg["params"].get("overwrite_click", False)
    MODIS_paths = []
    for input_path in raw_MODIS_paths:
        # Build output path in the same directory, e.g. ".../DFO_1818_..._pct.tif"
        file_name = Path(input_path).name
        dir0 = Path(input_path).parents[2]
        folder_name0 = Path(input_path).parents[0].name
        folder_name1 = Path(input_path).parents[1].name
        output_path = dir0.joinpath(folder_name1 + "_perc_regrid", folder_name0, file_name)

        if os.path.exists(output_path):
            log.info(f"Output already exists, skipping: {output_path}")
            MODIS_paths.append(str(output_path))
            continue

        # Ensure the directory exists (though it should, since input lives there)
        os.makedirs(output_path.parents[0], exist_ok=True)

        # Build the CLI args list
        args = [input_path, str(output_path), "--resolution", resolution]
        if overwrite_flag:
            args.append("--overwrite")

        # (Optional) if you also need to pass grid or CRS flags:
        if grid:
            args += ["--grid", grid]
        crs = None  # cfg["data_sources"].get("crs")
        if crs:
            args += ["--crs", str(crs)]

        log.info(f"Running flood-percent on {input_path}")
        try:
            # standalone_mode=False prevents Click from sys.exit(0) on success
            generate_flood_percent.main(args, standalone_mode=False)
            if os.path.exists(output_path):
                MODIS_paths.append(str(output_path))
            # if there is no overlap between input_path (satellite image) and master_grid. the dir will be deleted
            else:
                out_dir = Path(output_path).parent
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)

        except SystemExit as e:
            if e.code != 0:
                # non-zero exit → error
                raise RuntimeError(f"generate_flood_percent failed on {input_path}") from None

        log.info(f"→ Written: {output_path}")
    return MODIS_paths
