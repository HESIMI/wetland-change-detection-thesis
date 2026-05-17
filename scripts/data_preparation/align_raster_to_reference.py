from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
    "cubic": Resampling.cubic,
}


def align_raster(
    src_path: Path,
    ref_path: Path,
    output_path: Path,
    resampling: Resampling,
    backup_path: Path | None,
) -> None:
    if backup_path is not None and output_path.exists() and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, backup_path)

    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        ref_profile = ref.profile.copy()
        destination = np.zeros((src.count, ref.height, ref.width), dtype=src.dtypes[0])
        reproject(
            source=src.read(),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=ref_profile.get("nodata", 0),
            resampling=resampling,
        )

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff",
        count=destination.shape[0],
        dtype=str(destination.dtype),
        compress="lzw",
        nodata=ref_profile.get("nodata", 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a raster to a reference raster grid.")
    parser.add_argument("--src", type=Path, required=True, help="Source raster to reproject/resample.")
    parser.add_argument("--ref", type=Path, required=True, help="Reference raster providing CRS, transform, height, and width.")
    parser.add_argument("--output", type=Path, required=True, help="Output aligned raster.")
    parser.add_argument("--backup", type=Path, default=None, help="Optional backup path when overwriting an existing output.")
    parser.add_argument("--resampling", choices=sorted(RESAMPLING), default="bilinear")
    args = parser.parse_args()

    align_raster(
        src_path=args.src,
        ref_path=args.ref,
        output_path=args.output,
        resampling=RESAMPLING[args.resampling],
        backup_path=args.backup,
    )


if __name__ == "__main__":
    main()
