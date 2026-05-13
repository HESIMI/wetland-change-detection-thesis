import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from odc.stac import load
import planetary_computer
from pystac_client import Client
from shapely.geometry import box


AREAS = [
    {
        "name": "hangzhou_xixi",
        "bbox": [120.0000, 30.1800, 120.1800, 30.3200],
        "years": [2018, 2022],
    },
    {
        "name": "qiantang_estuary",
        "bbox": [120.9000, 30.2500, 121.2500, 30.5500],
        "years": [2018, 2022],
    },
    {
        "name": "poyang_lake",
        "bbox": [116.0000, 28.8000, 116.4500, 29.2000],
        "years": [2018, 2022],
    },
    {
        "name": "dongting_lake",
        "bbox": [112.2000, 28.7000, 113.4000, 29.5500],
        "years": [2018, 2022],
    },
    {
        "name": "yellow_river_delta",
        "bbox": [118.5500, 37.3500, 119.4500, 38.2000],
        "years": [2018, 2022],
    },
    {
        "name": "chongming_dongtan",
        "bbox": [121.3000, 31.3500, 122.1500, 31.9500],
        "years": [2018, 2022],
    },
]

BANDS = ["B02", "B03", "B04", "B08"]
MAX_ITEMS = 6
CLOUD_LT = 20


def xr_to_profile(ds) -> dict:
    gt = ds["spatial_ref"].attrs.get("GeoTransform")
    if gt:
        x0, xres, xrot, y0, yrot, yres = [float(v) for v in gt.split()]
        transform = Affine(xres, xrot, x0, yrot, yres, y0)
    else:
        x = ds.x.values
        y = ds.y.values
        xres = float(x[1] - x[0])
        yres = float(y[1] - y[0])
        transform = Affine.translation(float(x[0] - xres / 2), float(y[0] - yres / 2)) * Affine.scale(xres, yres)
    return {
        "driver": "GTiff",
        "height": ds.sizes["y"],
        "width": ds.sizes["x"],
        "count": len(BANDS),
        "dtype": "uint16",
        "crs": ds["spatial_ref"].attrs.get("crs_wkt"),
        "transform": transform,
        "compress": "lzw",
    }


def select_items(catalog, bbox, year):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-04-01/{year}-10-31",
        query={"eo:cloud_cover": {"lt": CLOUD_LT}},
    )
    items = sorted(
        list(search.items()),
        key=lambda item: item.properties.get("eo:cloud_cover", 100),
    )
    return items[:MAX_ITEMS]


def save_composite(ds, out_path: Path):
    profile = xr_to_profile(ds)
    arr = np.stack([np.asarray(ds[band].values) for band in BANDS], axis=0)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(np.rint(arr), 0, 10000).astype(np.uint16)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = tuple(BANDS)


def main():
    root = Path.cwd()
    out_root = root / "项目" / "data" / "raw"
    out_root.mkdir(parents=True, exist_ok=True)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    summary = []

    for area in AREAS:
        area_dir = out_root / area["name"]
        area_dir.mkdir(parents=True, exist_ok=True)
        area_summary = {"area": area["name"], "bbox": area["bbox"], "years": []}

        for year in area["years"]:
            items = select_items(catalog, area["bbox"], year)
            if not items:
                raise RuntimeError(f"No Sentinel-2 items found for {area['name']} {year}")

            crs = items[0].properties["proj:code"]
            ds = load(
                items,
                bands=BANDS,
                geopolygon=box(*area["bbox"]),
                resolution=10,
                crs=crs,
                groupby="solar_day",
                chunks={},
            )
            composite = ds.median(dim="time").compute()
            out_path = area_dir / f"sentinel2_{year}.tif"
            save_composite(composite, out_path)

            area_summary["years"].append(
                {
                    "year": year,
                    "output": str(out_path),
                    "selected_items": [
                        {
                            "id": item.id,
                            "datetime": item.datetime.isoformat() if item.datetime else None,
                            "cloud_cover": item.properties.get("eo:cloud_cover"),
                            "tile": item.properties.get("s2:mgrs_tile"),
                        }
                        for item in items
                    ],
                }
            )
            print(f"Saved {area['name']} {year} composite to {out_path}")

        summary.append(area_summary)

    summary_path = out_root / "sentinel_download_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
