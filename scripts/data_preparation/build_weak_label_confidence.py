from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


PSEUDO_CHANGE_FILES = ["pseudo_change_mask.tif"]


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


def read_resampled(path: Path, dst_profile: dict, indexes: list[int]) -> np.ndarray:
    bands = []
    with rasterio.open(path) as src:
        for index in indexes:
            dst = np.zeros((dst_profile["height"], dst_profile["width"]), dtype=np.float32)
            reproject(
                source=rasterio.band(src, index),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                resampling=Resampling.average,
            )
            bands.append(dst)
    return np.stack(bands, axis=0)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def robust_scale(score: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = score[valid_mask]
    if valid_values.size == 0:
        return np.zeros_like(score, dtype=np.float32)
    low, high = np.percentile(valid_values, [2, 98])
    if high <= low:
        return np.zeros_like(score, dtype=np.float32)
    scaled = (score - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def spectral_change_score(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    # Expected Sentinel-2 band order from the current pipeline: B02, B03, B04, B08.
    blue1, green1, red1, nir1 = t1
    blue2, green2, red2, nir2 = t2

    ndvi1 = normalized_difference(nir1, red1)
    ndvi2 = normalized_difference(nir2, red2)
    ndwi1 = normalized_difference(green1, nir1)
    ndwi2 = normalized_difference(green2, nir2)
    brightness1 = (blue1 + green1 + red1 + nir1) / 4.0
    brightness2 = (blue2 + green2 + red2 + nir2) / 4.0

    return np.sqrt(
        (ndvi2 - ndvi1) ** 2
        + (ndwi2 - ndwi1) ** 2
        + ((brightness2 - brightness1) / 10000.0) ** 2
    ).astype(np.float32)


def find_pseudo_mask(change_label_root: Path, area: str, shape: tuple[int, int]) -> np.ndarray:
    area_dir = change_label_root / area
    for filename in PSEUDO_CHANGE_FILES:
        path = area_dir / filename
        if path.exists():
            with rasterio.open(path) as src:
                return (src.read(1) > 0).astype(np.uint8)
    return np.zeros(shape, dtype=np.uint8)


def build_area(
    area_dir: Path,
    raw_root: Path,
    change_label_root: Path,
    output_root: Path,
    spectral_high_quantile: float,
    spectral_low_quantile: float,
) -> dict:
    area = area_dir.name
    area_out = output_root / area
    area_out.mkdir(parents=True, exist_ok=True)

    lc_t1_path = area_dir / "lc_t1_2018.tif"
    lc_t2_path = area_dir / "lc_t2_2022.tif"
    initial_change_path = area_dir / "initial_change.tif"
    s2_t1_path = raw_root / area / "sentinel2_2018.tif"
    s2_t2_path = raw_root / area / "sentinel2_2022.tif"

    with rasterio.open(lc_t1_path) as src:
        lc_t1 = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(lc_t2_path) as src:
        lc_t2 = src.read(1)
    with rasterio.open(initial_change_path) as src:
        initial_change = (src.read(1) > 0)

    valid_mask = (lc_t1 != 0) & (lc_t2 != 0)
    pseudo_change = find_pseudo_mask(change_label_root, area, initial_change.shape).astype(bool)

    dst_profile = {
        "height": profile["height"],
        "width": profile["width"],
        "transform": profile["transform"],
        "crs": profile["crs"],
    }
    s2_t1 = read_resampled(s2_t1_path, dst_profile, [1, 2, 3, 4])
    s2_t2 = read_resampled(s2_t2_path, dst_profile, [1, 2, 3, 4])
    spectral_raw = spectral_change_score(s2_t1, s2_t2)
    spectral_scaled = robust_scale(spectral_raw, valid_mask)

    valid_scores = spectral_scaled[valid_mask]
    low_threshold = float(np.quantile(valid_scores, spectral_low_quantile)) if valid_scores.size else 0.25
    high_threshold = float(np.quantile(valid_scores, spectral_high_quantile)) if valid_scores.size else 0.65

    spectral_low = spectral_scaled <= low_threshold
    spectral_high = spectral_scaled >= high_threshold

    landcover_change = initial_change & valid_mask
    landcover_unchanged = (~initial_change) & valid_mask

    multi_source_consistency = (
        (landcover_change & spectral_high)
        | (landcover_unchanged & spectral_low)
    )

    # With two available years, this is a temporal proxy based on same-season spectral stability/change evidence.
    temporal_consistency = (
        (landcover_change & (~pseudo_change) & spectral_high)
        | (landcover_unchanged & spectral_low)
    )

    high_confidence_change = landcover_change & (~pseudo_change) & spectral_high
    high_confidence_unchanged = landcover_unchanged & spectral_low
    high_confidence_mask = high_confidence_change | high_confidence_unchanged

    low_confidence_mask = (
        pseudo_change
        | (landcover_change & spectral_low)
        | (landcover_unchanged & spectral_high)
    ) & valid_mask

    confidence_score = np.zeros(initial_change.shape, dtype=np.uint8)
    confidence_score[high_confidence_mask] = 2
    confidence_score[low_confidence_mask] = 1

    outputs = {
        "spectral_change_score.tif": (spectral_scaled * 10000).astype(np.uint16),
        "multi_source_consistency.tif": multi_source_consistency.astype(np.uint8),
        "temporal_consistency.tif": temporal_consistency.astype(np.uint8),
        "high_confidence_mask.tif": high_confidence_mask.astype(np.uint8),
        "high_confidence_change.tif": high_confidence_change.astype(np.uint8),
        "high_confidence_unchanged.tif": high_confidence_unchanged.astype(np.uint8),
        "low_confidence_mask.tif": low_confidence_mask.astype(np.uint8),
        "confidence_score.tif": confidence_score,
    }
    for filename, array in outputs.items():
        write_raster(area_out / filename, array, profile)

    valid_pixels = int(valid_mask.sum())
    return {
        "area": area,
        "valid_pixels": valid_pixels,
        "initial_change_pixels": int(landcover_change.sum()),
        "pseudo_change_pixels": int((pseudo_change & valid_mask).sum()),
        "spectral_low_threshold": low_threshold,
        "spectral_high_threshold": high_threshold,
        "multi_source_consistent_pixels": int(multi_source_consistency.sum()),
        "temporal_consistent_pixels": int(temporal_consistency.sum()),
        "high_confidence_pixels": int(high_confidence_mask.sum()),
        "low_confidence_pixels": int(low_confidence_mask.sum()),
        "high_confidence_change_pixels": int(high_confidence_change.sum()),
        "high_confidence_unchanged_pixels": int(high_confidence_unchanged.sum()),
        "high_confidence_ratio": float(high_confidence_mask.sum() / valid_pixels) if valid_pixels else 0.0,
        "low_confidence_ratio": float(low_confidence_mask.sum() / valid_pixels) if valid_pixels else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak-label confidence masks using land-cover and Sentinel-2 evidence.")
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--change-label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spectral-high-quantile", type=float, default=0.75)
    parser.add_argument("--spectral-low-quantile", type=float, default=0.40)
    args = parser.parse_args()

    areas = sorted(path for path in args.initial_root.iterdir() if path.is_dir())
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = [
        build_area(
            area_dir=area,
            raw_root=args.raw_root,
            change_label_root=args.change_label_root,
            output_root=args.output_root,
            spectral_high_quantile=args.spectral_high_quantile,
            spectral_low_quantile=args.spectral_low_quantile,
        )
        for area in areas
    ]

    summary_path = args.output_root / "weak_label_confidence_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    for item in summary:
        print(
            f"{item['area']}: high={item['high_confidence_pixels']} "
            f"low={item['low_confidence_pixels']} "
            f"pseudo={item['pseudo_change_pixels']}"
        )


if __name__ == "__main__":
    main()
