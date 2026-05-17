from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        compress="lzw",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)


def top_counts(array: np.ndarray, topk: int = 12) -> list[dict[str, int]]:
    values, counts = np.unique(array, return_counts=True)
    order = np.argsort(counts)[::-1][:topk]
    return [{"value": int(values[i]), "count": int(counts[i])} for i in order]


def build_area(area_dir: Path, output_root: Path, t1_year: int, t2_year: int) -> dict:
    area = area_dir.name
    lc_t1_src = area_dir / f"glc_{t1_year}.tif"
    lc_t2_src = area_dir / f"glc_{t2_year}.tif"
    if not lc_t1_src.exists() or not lc_t2_src.exists():
        raise FileNotFoundError(f"Missing land-cover pair for {area}: {lc_t1_src}, {lc_t2_src}")

    area_out = output_root / area
    area_out.mkdir(parents=True, exist_ok=True)

    lc_t1_out = area_out / f"lc_t1_{t1_year}.tif"
    lc_t2_out = area_out / f"lc_t2_{t2_year}.tif"
    shutil.copy2(lc_t1_src, lc_t1_out)
    shutil.copy2(lc_t2_src, lc_t2_out)

    with rasterio.open(lc_t1_src) as src1, rasterio.open(lc_t2_src) as src2:
        lc_t1 = src1.read(1)
        lc_t2 = src2.read(1)
        profile = src1.profile.copy()

        if lc_t1.shape != lc_t2.shape:
            raise ValueError(f"Shape mismatch for {area}: {lc_t1.shape} vs {lc_t2.shape}")
        if src1.transform != src2.transform:
            raise ValueError(f"Transform mismatch for {area}")
        if src1.crs != src2.crs:
            raise ValueError(f"CRS mismatch for {area}")

    valid_mask = (lc_t1 != 0) & (lc_t2 != 0)
    initial_change = ((lc_t1 != lc_t2) & valid_mask).astype(np.uint8)
    write_raster(area_out / "initial_change.tif", initial_change, profile)

    changed_pixels = int(initial_change.sum())
    valid_pixels = int(valid_mask.sum())
    total_pixels = int(initial_change.size)

    return {
        "area": area,
        "t1_year": t1_year,
        "t2_year": t2_year,
        "shape": [int(lc_t1.shape[0]), int(lc_t1.shape[1])],
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "changed_pixels": changed_pixels,
        "initial_change_ratio_total": float(changed_pixels / total_pixels) if total_pixels else 0.0,
        "initial_change_ratio_valid": float(changed_pixels / valid_pixels) if valid_pixels else 0.0,
        "lc_t1": str(lc_t1_out),
        "lc_t2": str(lc_t2_out),
        "initial_change": str(area_out / "initial_change.tif"),
        "top_classes_t1": top_counts(lc_t1),
        "top_classes_t2": top_counts(lc_t2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build initial weak labels from two land-cover maps.")
    parser.add_argument("--glc-root", type=Path, required=True, help="Directory containing area/glc_YEAR.tif files.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--t1-year", type=int, default=2018)
    parser.add_argument("--t2-year", type=int, default=2022)
    args = parser.parse_args()

    areas = sorted(path for path in args.glc_root.iterdir() if path.is_dir())
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = [build_area(area, args.output_root, args.t1_year, args.t2_year) for area in areas]
    summary_path = args.output_root / "initial_weak_label_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    for item in summary:
        print(
            f"{item['area']}: changed={item['changed_pixels']} "
            f"ratio_valid={item['initial_change_ratio_valid']:.4f}"
        )


if __name__ == "__main__":
    main()

