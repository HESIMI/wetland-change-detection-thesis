from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ESA_S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

ESA_TO_UNIFIED = {
    10: 20,   # Tree cover -> forest
    20: 30,   # Shrubland -> shrubland
    30: 40,   # Grassland -> grassland
    40: 10,   # Cropland -> cropland
    50: 100,  # Built-up -> built_up
    60: 110,  # Bare/sparse vegetation -> bare_land
    70: 120,  # Snow and ice -> snow_ice
    80: 50,   # Permanent water bodies -> water
    90: 90,   # Herbaceous wetland -> wetland
    95: 90,   # Mangroves -> wetland
    100: 40,  # Moss and lichen -> grassland/low vegetation
}

GLC_TO_UNIFIED = {
    10: 10,
    11: 10,
    12: 10,
    20: 20,
    30: 30,
    40: 40,
    50: 50,
    51: 50,
    52: 50,
    53: 50,
    60: 110,
    61: 110,
    62: 110,
    70: 120,
    71: 120,
    72: 120,
    180: 90,
    181: 90,
    182: 90,
    183: 90,
    184: 90,
    185: 90,
    186: 90,
    187: 90,
    190: 100,
    210: 90,
}

UNIFIED_CLASS_NAMES = {
    0: "nodata",
    10: "cropland",
    20: "forest",
    30: "shrubland",
    40: "grassland",
    50: "water",
    90: "wetland",
    100: "built_up",
    110: "bare_land",
    120: "snow_ice",
}


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        compress="lzw",
        nodata=0,
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)


def recode(array: np.ndarray, rules: dict[int, int]) -> np.ndarray:
    out = np.zeros(array.shape, dtype=np.uint8)
    for src_value, dst_value in rules.items():
        out[array == src_value] = dst_value
    return out


def tile_name(lon: int, lat: int) -> str:
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"


def worldcover_tiles(bounds: rasterio.coords.BoundingBox) -> list[str]:
    min_lon = math.floor(bounds.left / 3.0) * 3
    max_lon = math.floor((bounds.right - 1e-10) / 3.0) * 3
    min_lat = math.floor(bounds.bottom / 3.0) * 3
    max_lat = math.floor((bounds.top - 1e-10) / 3.0) * 3

    tiles = []
    for lat in range(min_lat, max_lat + 1, 3):
        for lon in range(min_lon, max_lon + 1, 3):
            tiles.append(tile_name(lon, lat))
    return sorted(tiles)


def download_tile(tile: str, tile_dir: Path, force: bool = False, retries: int = 3) -> Path:
    filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    output_path = tile_dir / filename
    if output_path.exists() and not force:
        return output_path

    tile_dir.mkdir(parents=True, exist_ok=True)
    url = f"{ESA_S3_PREFIX}/{filename}"
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Downloading {tile} (attempt {attempt}/{retries}): {url}")
            urllib.request.urlretrieve(url, tmp_path)
            break
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    tmp_path.replace(output_path)
    return output_path


def align_tiles_to_reference(tile_paths: list[Path], reference_profile: dict) -> np.ndarray:
    dst = np.zeros((reference_profile["height"], reference_profile["width"]), dtype=np.uint8)
    dst_grid = {
        "height": reference_profile["height"],
        "width": reference_profile["width"],
        "transform": reference_profile["transform"],
        "crs": reference_profile["crs"],
    }

    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            tmp = np.zeros(dst.shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=dst_grid["transform"],
                dst_crs=dst_grid["crs"],
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            dst[tmp != 0] = tmp[tmp != 0]
    return dst


def top_counts(array: np.ndarray, topk: int = 12) -> list[dict[str, int | str]]:
    values, counts = np.unique(array, return_counts=True)
    order = np.argsort(counts)[::-1][:topk]
    return [
        {
            "value": int(values[i]),
            "name": UNIFIED_CLASS_NAMES.get(int(values[i]), "unknown"),
            "count": int(counts[i]),
        }
        for i in order
    ]


def build_area(area_dir: Path, tile_dir: Path, output_root: Path, force_download: bool) -> dict:
    area = area_dir.name
    glc_t1_path = area_dir / "glc_2018.tif"
    glc_t2_path = area_dir / "glc_2022.tif"
    if not glc_t1_path.exists() or not glc_t2_path.exists():
        raise FileNotFoundError(f"Missing GLC pair for {area}: {glc_t1_path}, {glc_t2_path}")

    with rasterio.open(glc_t1_path) as ref:
        profile = ref.profile.copy()
        bounds = ref.bounds
        glc_t1 = ref.read(1)
    with rasterio.open(glc_t2_path) as src:
        glc_t2 = src.read(1)

    tiles = worldcover_tiles(bounds)
    tile_paths = [download_tile(tile, tile_dir, force=force_download) for tile in tiles]
    esa_raw = align_tiles_to_reference(tile_paths, profile)
    esa_semantic = recode(esa_raw, ESA_TO_UNIFIED)
    glc_t1_semantic = recode(glc_t1, GLC_TO_UNIFIED)
    glc_t2_semantic = recode(glc_t2, GLC_TO_UNIFIED)

    valid_t1 = (esa_semantic != 0) & (glc_t1_semantic != 0)
    valid_t2 = (esa_semantic != 0) & (glc_t2_semantic != 0)
    esa_glc_t1_consistency = (esa_semantic == glc_t1_semantic) & valid_t1
    esa_glc_t2_consistency = (esa_semantic == glc_t2_semantic) & valid_t2
    esa_glc_any_consistency = esa_glc_t1_consistency | esa_glc_t2_consistency
    esa_consistency_score = esa_glc_t1_consistency.astype(np.uint8) + esa_glc_t2_consistency.astype(np.uint8)

    area_out = output_root / area
    write_raster(area_out / "esa_worldcover_2021_raw_aligned.tif", esa_raw, profile)
    write_raster(area_out / "esa_worldcover_2021_semantic.tif", esa_semantic, profile)
    write_raster(area_out / "glc_2018_semantic_for_esa.tif", glc_t1_semantic, profile)
    write_raster(area_out / "glc_2022_semantic_for_esa.tif", glc_t2_semantic, profile)
    write_raster(area_out / "esa_glc_t1_consistency.tif", esa_glc_t1_consistency.astype(np.uint8), profile)
    write_raster(area_out / "esa_glc_t2_consistency.tif", esa_glc_t2_consistency.astype(np.uint8), profile)
    write_raster(area_out / "esa_glc_any_consistency.tif", esa_glc_any_consistency.astype(np.uint8), profile)
    write_raster(area_out / "esa_consistency_score.tif", esa_consistency_score.astype(np.uint8), profile)

    total_pixels = int(esa_semantic.size)
    valid_pixels = int((esa_semantic != 0).sum())
    return {
        "area": area,
        "tiles": tiles,
        "shape": [int(esa_semantic.shape[0]), int(esa_semantic.shape[1])],
        "total_pixels": total_pixels,
        "esa_valid_pixels": valid_pixels,
        "esa_valid_ratio": float(valid_pixels / total_pixels) if total_pixels else 0.0,
        "esa_glc_t1_consistent_pixels": int(esa_glc_t1_consistency.sum()),
        "esa_glc_t2_consistent_pixels": int(esa_glc_t2_consistency.sum()),
        "esa_glc_any_consistent_pixels": int(esa_glc_any_consistency.sum()),
        "esa_glc_t1_consistency_ratio": float(esa_glc_t1_consistency.sum() / valid_t1.sum()) if valid_t1.sum() else 0.0,
        "esa_glc_t2_consistency_ratio": float(esa_glc_t2_consistency.sum() / valid_t2.sum()) if valid_t2.sum() else 0.0,
        "top_esa_semantic_classes": top_counts(esa_semantic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ESA WorldCover 2021 tiles, align them to wetland GLC grids, and build consistency layers."
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing area/glc_2018.tif and glc_2022.tif.")
    parser.add_argument("--tile-dir", type=Path, required=True, help="Directory used to cache ESA WorldCover COG tiles.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for aligned ESA products and consistency maps.")
    parser.add_argument("--force-download", action="store_true", help="Redownload ESA tiles even if they already exist.")
    args = parser.parse_args()

    areas = sorted(path for path in args.raw_root.iterdir() if path.is_dir())
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = [build_area(area, args.tile_dir, args.output_root, args.force_download) for area in areas]

    summary_path = args.output_root / "esa_worldcover_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    for item in summary:
        print(
            f"{item['area']}: ESA-GLC T1={item['esa_glc_t1_consistency_ratio']:.4f} "
            f"T2={item['esa_glc_t2_consistency_ratio']:.4f} tiles={','.join(item['tiles'])}"
        )


if __name__ == "__main__":
    main()
