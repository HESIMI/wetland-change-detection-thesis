import json
from pathlib import Path

import numpy as np
import rasterio


# First-level semantic recoding for later CLIP prompt construction.
SEMANTIC_RULES = {
    "cropland": [10, 11, 12],
    "forest": [20],
    "shrubland": [30],
    "grassland": [40],
    "water": [50, 51, 52, 53],
    "wetland": [180, 181, 182, 183, 184, 185, 186, 187],
    "built_up": [190],
    "bare_land": [60, 61, 62],
    "snow_ice": [70, 71, 72],
    "tidal_flat_special": [210],
}

SEMANTIC_IDS = {
    "background": 0,
    "cropland": 10,
    "forest": 20,
    "shrubland": 30,
    "grassland": 40,
    "water": 50,
    "wetland": 90,
    "built_up": 100,
    "bare_land": 110,
    "snow_ice": 120,
    "tidal_flat_special": 130,
}

PSEUDO_CHANGE_TRANSITIONS = {
    (50, 90),   # water -> wetland
    (90, 50),   # wetland -> water
    (90, 130),  # wetland -> tidal flat
    (130, 90),  # tidal flat -> wetland
    (50, 130),  # water -> tidal flat
    (130, 50),  # tidal flat -> water
}


def recode_semantic(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint16)
    for semantic_name, raw_values in SEMANTIC_RULES.items():
        semantic_id = SEMANTIC_IDS[semantic_name]
        out[np.isin(arr, raw_values)] = semantic_id
    return out


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        compress="lzw",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)


def top_counts(arr: np.ndarray, topk: int = 12) -> list[dict]:
    values, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1][:topk]
    return [
        {"value": int(values[i]), "count": int(counts[i])}
        for i in order
    ]


def main() -> None:
    root = Path.cwd()
    subset_root = root / "项目" / "data" / "glc_subsets"
    out_root = root / "项目" / "data" / "change_labels"
    out_root.mkdir(parents=True, exist_ok=True)
    areas = sorted([p.name for p in subset_root.iterdir() if p.is_dir()])

    summary = []

    for area in areas:
        area_in = subset_root / area
        area_out = out_root / area
        area_out.mkdir(parents=True, exist_ok=True)

        t1_path = area_in / "glc_2018.tif"
        t2_path = area_in / "glc_2022.tif"

        with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
            t1 = src1.read(1)
            t2 = src2.read(1)
            profile = src1.profile.copy()

        s1 = recode_semantic(t1)
        s2 = recode_semantic(t2)

        raw_change = (t1 != t2).astype(np.uint8)
        semantic_change = np.where(t1 != t2, s1.astype(np.uint32) * 1000 + s2.astype(np.uint32), 0).astype(np.uint32)

        binary_change = raw_change.copy()
        pseudo_change = np.zeros_like(raw_change, dtype=np.uint8)
        for src_id, dst_id in PSEUDO_CHANGE_TRANSITIONS:
            mask = (s1 == src_id) & (s2 == dst_id)
            pseudo_change[mask] = 1
            binary_change[mask] = 0

        wetland_mask_t1 = (s1 == SEMANTIC_IDS["wetland"]).astype(np.uint8)
        wetland_mask_t2 = (s2 == SEMANTIC_IDS["wetland"]).astype(np.uint8)

        outputs = {
            "glc_2018_semantic.tif": s1.astype(np.uint16),
            "glc_2022_semantic.tif": s2.astype(np.uint16),
            "wetland_mask_2018.tif": wetland_mask_t1,
            "wetland_mask_2022.tif": wetland_mask_t2,
            "binary_change_raw.tif": raw_change,
            "binary_change_final.tif": binary_change,
            "pseudo_change_mask.tif": pseudo_change,
            "semantic_change.tif": semantic_change,
        }

        for name, arr in outputs.items():
            write_raster(area_out / name, arr, profile)

        area_summary = {
            "area": area,
            "shape": [int(t1.shape[0]), int(t1.shape[1])],
            "raw_changed_pixels": int(raw_change.sum()),
            "final_changed_pixels": int(binary_change.sum()),
            "pseudo_change_pixels": int(pseudo_change.sum()),
            "raw_change_ratio": float(raw_change.mean()),
            "final_change_ratio": float(binary_change.mean()),
            "wetland_pixels_2018": int(wetland_mask_t1.sum()),
            "wetland_pixels_2022": int(wetland_mask_t2.sum()),
            "top_raw_classes_2018": top_counts(t1),
            "top_raw_classes_2022": top_counts(t2),
            "top_semantic_transitions": top_counts(semantic_change[semantic_change > 0]) if np.any(semantic_change > 0) else [],
        }
        summary.append(area_summary)

    summary_path = out_root / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    for item in summary:
        print(
            f"{item['area']}: raw={item['raw_changed_pixels']} "
            f"final={item['final_changed_pixels']} pseudo={item['pseudo_change_pixels']}"
        )


if __name__ == "__main__":
    main()
