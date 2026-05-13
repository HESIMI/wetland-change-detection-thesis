import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


YEAR_START = 2000


AREAS = [
    {
        "name": "hangzhou_xixi",
        "source_tile": "GLC_FCS30D_20002022_E120N35_Annual.tif",
        "bbox": [120.0000, 30.1800, 120.1800, 30.3200],
        "years": [2018, 2022],
    },
    {
        "name": "qiantang_estuary",
        "source_tile": "GLC_FCS30D_20002022_E120N35_Annual.tif",
        "bbox": [120.9000, 30.2500, 121.2500, 30.5500],
        "years": [2018, 2022],
    },
    {
        "name": "poyang_lake",
        "source_tile": "GLC_FCS30D_20002022_E115N30_Annual.tif",
        "bbox": [116.0000, 28.8000, 116.4500, 29.2000],
        "years": [2018, 2022],
    },
    {
        "name": "dongting_lake",
        "source_tile": "GLC_FCS30D_20002022_E110N30_Annual.tif",
        "bbox": [112.2000, 28.7000, 113.4000, 29.5500],
        "years": [2018, 2022],
    },
    {
        "name": "yellow_river_delta",
        "source_tile": "GLC_FCS30D_20002022_E115N40_Annual.tif",
        "bbox": [118.5500, 37.3500, 119.4500, 38.2000],
        "years": [2018, 2022],
    },
    {
        "name": "chongming_dongtan",
        "source_tile": "GLC_FCS30D_20002022_E120N35_Annual.tif",
        "bbox": [121.3000, 31.3500, 122.1500, 31.9500],
        "years": [2018, 2022],
    },
]


def band_index_for_year(year: int) -> int:
    return year - YEAR_START + 1


def ensure_inside(bounds, bbox, area_name: str) -> None:
    left, bottom, right, top = bbox
    if not (bounds.left <= left < right <= bounds.right and bounds.bottom <= bottom < top <= bounds.top):
        raise ValueError(f"{area_name} bbox {bbox} is outside source bounds {bounds}")


def save_single_band(dst_path: Path, array: np.ndarray, src, window) -> dict:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        height=array.shape[0],
        width=array.shape[1],
        transform=rasterio.windows.transform(window, src.transform),
        compress="lzw",
    )
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(array, 1)
    values, counts = np.unique(array, return_counts=True)
    pairs = sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:10]
    return {
        "path": str(dst_path),
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "top_values": [{"value": int(v), "count": int(c)} for c, v in pairs],
    }


def main() -> None:
    root = Path.cwd()
    raw_dir = root / "项目" / "data" / "raw_glc_fcs30d"
    out_root = root / "项目" / "data" / "glc_subsets"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for area in AREAS:
        src_path = raw_dir / area["source_tile"]
        area_dir = out_root / area["name"]
        area_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(src_path) as src:
            ensure_inside(src.bounds, area["bbox"], area["name"])
            window = from_bounds(*area["bbox"], transform=src.transform)
            window = window.round_offsets().round_lengths()

            area_summary = {
                "area": area["name"],
                "source_tile": area["source_tile"],
                "bbox": area["bbox"],
                "years": area["years"],
                "outputs": [],
            }

            for year in area["years"]:
                band = band_index_for_year(year)
                array = src.read(band, window=window)
                dst_path = area_dir / f"glc_{year}.tif"
                band_summary = save_single_band(dst_path, array, src, window)
                band_summary["year"] = year
                band_summary["band_index"] = band
                area_summary["outputs"].append(band_summary)

        summary.append(area_summary)

    summary_path = out_root / "clip_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    for item in summary:
        print(item["area"])
        for output in item["outputs"]:
            print(
                f"  year={output['year']} band={output['band_index']} "
                f"shape={tuple(output['shape'])} path={output['path']}"
            )


if __name__ == "__main__":
    main()
