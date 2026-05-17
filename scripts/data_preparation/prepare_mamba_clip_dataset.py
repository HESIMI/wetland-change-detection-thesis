import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject


@dataclass
class RasterBundle:
    data: np.ndarray
    profile: dict


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_raster(path: Path) -> RasterBundle:
    with rasterio.open(path) as src:
      data = src.read()
      profile = src.profile.copy()
    return RasterBundle(data=data, profile=profile)


def align_to_reference(src_path: Path, ref_profile: dict, resampling: Resampling) -> RasterBundle:
    with rasterio.open(src_path) as src:
        destination = np.zeros(
            (src.count, ref_profile["height"], ref_profile["width"]),
            dtype=src.dtypes[0],
        )
        reproject(
            source=src.read(),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=ref_profile.get("nodata", 0),
        )
    aligned_profile = ref_profile.copy()
    aligned_profile.update(count=destination.shape[0], dtype=str(destination.dtype))
    return RasterBundle(data=destination, profile=aligned_profile)


def recode_semantic_classes(
    label: np.ndarray,
    semantic_map: Dict[str, str],
    wetland_codes: Iterable[int],
    semantic_rules: Dict[str, List[int]] | None = None,
) -> np.ndarray:
    recoded = np.zeros_like(label, dtype=np.uint8)
    wetland_codes = set(int(code) for code in wetland_codes)
    if semantic_rules:
        for class_id_str, _ in semantic_map.items():
            class_id = int(class_id_str)
            if class_id == 90:
                for code in wetland_codes:
                    recoded[label == code] = 90
                continue

            raw_codes = semantic_rules.get(class_id_str, [class_id])
            for code in raw_codes:
                recoded[label == int(code)] = class_id
    else:
        for class_id_str in semantic_map:
            class_id = int(class_id_str)
            if class_id == 90:
                for code in wetland_codes:
                    recoded[label == code] = 90
            else:
                recoded[label == class_id] = class_id
    return recoded


def build_change_labels(
    label_t1: np.ndarray,
    label_t2: np.ndarray,
    wetland_codes: Iterable[int],
    treat_internal_shift_as_change: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    label_t1 = label_t1.astype(np.int32, copy=False)
    label_t2 = label_t2.astype(np.int32, copy=False)
    wetland_codes = set(int(code) for code in wetland_codes)
    wetland_t1 = np.isin(label_t1, list(wetland_codes))
    wetland_t2 = np.isin(label_t2, list(wetland_codes))

    binary_change = (wetland_t1 != wetland_t2).astype(np.uint8)
    if treat_internal_shift_as_change:
        binary_change[(wetland_t1 & wetland_t2) & (label_t1 != label_t2)] = 1

    semantic_change = np.zeros(label_t1.shape, dtype=np.int32)
    transition_mask = label_t1 != label_t2
    semantic_change[transition_mask] = label_t1[transition_mask] * 1000 + label_t2[transition_mask]
    return binary_change, semantic_change


def sliding_windows(width: int, height: int, chip_size: int, overlap: int) -> Iterable[Tuple[int, int, Window]]:
    stride = chip_size - overlap
    cols = max(1, math.ceil((width - overlap) / stride))
    rows = max(1, math.ceil((height - overlap) / stride))
    for row in range(rows):
        for col in range(cols):
            x = min(col * stride, max(0, width - chip_size))
            y = min(row * stride, max(0, height - chip_size))
            yield row, col, Window(x, y, chip_size, chip_size)


def chip_valid_ratio(mask: np.ndarray, nodata_value: int) -> float:
    valid = np.count_nonzero(mask != nodata_value)
    return valid / mask.size


def summarize_transition(semantic_chip: np.ndarray, class_names: Dict[str, str]) -> str:
    transitions = semantic_chip[semantic_chip > 0]
    if transitions.size == 0:
        return "no_permanent_change"

    values, counts = np.unique(transitions, return_counts=True)
    major = int(values[np.argmax(counts)])
    src = major // 1000
    dst = major % 1000
    src_name = class_names.get(str(src), f"class_{src}")
    dst_name = class_names.get(str(dst), f"class_{dst}")
    return f"{src_name}_to_{dst_name}"


def build_prompts(
    binary_chip: np.ndarray,
    semantic_chip: np.ndarray,
    class_names: Dict[str, str],
    scene_template: str,
    transition_template: str,
) -> List[str]:
    prompts: List[str] = []
    if binary_chip.max() == 0:
        prompts.append("A satellite image pair with seasonal fluctuation but no permanent wetland change.")
        return prompts

    transitions = semantic_chip[semantic_chip > 0]
    if transitions.size == 0:
        prompts.append("A remote sensing change patch showing wetland conversion.")
        return prompts

    unique_transitions = np.unique(transitions)
    for encoded in unique_transitions[:3]:
        src = int(encoded) // 1000
        dst = int(encoded) % 1000
        prompts.append(
            transition_template.format(
                src=class_names.get(str(src), f"class_{src}"),
                dst=class_names.get(str(dst), f"class_{dst}"),
            )
        )

    covered = {int(code) % 1000 for code in unique_transitions}
    for dst in list(covered)[:2]:
        prompts.append(scene_template.format(label=class_names.get(str(dst), f"class_{dst}")))
    return prompts


def save_chip(path: Path, array: np.ndarray, profile: dict) -> None:
    chip_profile = profile.copy()
    chip_profile.update(
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=str(array.dtype),
    )
    with rasterio.open(path, "w", **chip_profile) as dst:
        dst.write(array)


def prepare_area(area: dict, config: dict) -> List[dict]:
    input_root = Path(config["input_root"])
    output_root = Path(config["output_root"])
    area_name = area["name"]
    split = area["split"]
    year_t1, year_t2 = area["years"]

    area_input = input_root / area_name
    s2_t1_path = area_input / f"sentinel2_{year_t1}.tif"
    s2_t2_path = area_input / f"sentinel2_{year_t2}.tif"
    glc_t1_path = area_input / f"glc_{year_t1}.tif"
    glc_t2_path = area_input / f"glc_{year_t2}.tif"

    for path in [s2_t1_path, s2_t2_path, glc_t1_path, glc_t2_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input raster: {path}")

    s2_t1 = open_raster(s2_t1_path)
    s2_t2 = align_to_reference(s2_t2_path, s2_t1.profile, Resampling.bilinear)
    glc_t1 = align_to_reference(glc_t1_path, s2_t1.profile, Resampling.nearest)
    glc_t2 = align_to_reference(glc_t2_path, s2_t1.profile, Resampling.nearest)

    label_t1 = glc_t1.data[0]
    label_t2 = glc_t2.data[0]
    wetland_codes = config["wetland_codes"]
    semantic_map = config["semantic_classes"]

    semantic_rules = config.get("semantic_rules")
    semantic_t1 = recode_semantic_classes(label_t1, semantic_map, wetland_codes, semantic_rules)
    semantic_t2 = recode_semantic_classes(label_t2, semantic_map, wetland_codes, semantic_rules)
    binary_change, semantic_change = build_change_labels(
        semantic_t1,
        semantic_t2,
        wetland_codes=[90],
        treat_internal_shift_as_change=config["binary_change_rules"]["treat_wetland_internal_shift_as_change"],
    )

    out_area = output_root / split / area_name
    img_dir = out_area / "images"
    label_dir = out_area / "labels"
    prompt_dir = out_area / "prompts"
    for directory in [img_dir, label_dir, prompt_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    chip_size = int(config["chip_size"])
    overlap = int(config["overlap"])
    nodata_value = int(config["nodata_value"])

    for row, col, window in sliding_windows(s2_t1.profile["width"], s2_t1.profile["height"], chip_size, overlap):
        x0, y0 = int(window.col_off), int(window.row_off)
        x1, y1 = x0 + chip_size, y0 + chip_size

        img_t1_chip = s2_t1.data[:, y0:y1, x0:x1]
        img_t2_chip = s2_t2.data[:, y0:y1, x0:x1]
        binary_chip = binary_change[y0:y1, x0:x1]
        semantic_chip = semantic_change[y0:y1, x0:x1]

        if img_t1_chip.shape[1] != chip_size or img_t1_chip.shape[2] != chip_size:
            continue
        if chip_valid_ratio(img_t1_chip[0], nodata_value) < float(config["min_valid_ratio"]):
            continue

        sample_id = f"{area_name}_{year_t1}_{year_t2}_r{row:03d}_c{col:03d}"
        img_stack = np.concatenate([img_t1_chip, img_t2_chip], axis=0)
        label_stack = np.stack([binary_chip.astype(np.uint8), semantic_chip.astype(np.int32)])

        chip_profile = s2_t1.profile.copy()
        chip_profile.update(transform=rasterio.windows.transform(window, s2_t1.profile["transform"]))

        save_chip(img_dir / f"{sample_id}.tif", img_stack, chip_profile)
        save_chip(label_dir / f"{sample_id}.tif", label_stack, chip_profile)

        prompts = build_prompts(
            binary_chip=binary_chip,
            semantic_chip=semantic_chip,
            class_names=semantic_map,
            scene_template=config["prompt_template"],
            transition_template=config["transition_prompt_template"],
        )
        prompt_payload = {
            "sample_id": sample_id,
            "area": area_name,
            "split": split,
            "years": [year_t1, year_t2],
            "dominant_transition": summarize_transition(semantic_chip, semantic_map),
            "prompts": prompts,
        }
        (prompt_dir / f"{sample_id}.json").write_text(
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "area": area_name,
                "image_path": str((img_dir / f"{sample_id}.tif").resolve()),
                "label_path": str((label_dir / f"{sample_id}.tif").resolve()),
                "prompt_path": str((prompt_dir / f"{sample_id}.json").resolve()),
                "binary_change_ratio": float(binary_chip.mean()),
                "dominant_transition": summarize_transition(semantic_chip, semantic_map),
            }
        )

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare wetland change detection data for Mamba-CLIP.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_dataset_config.json"),
        help="Path to the dataset configuration JSON file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    all_rows: List[dict] = []
    for area in config["study_areas"]:
        all_rows.extend(prepare_area(area, config))

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "dataset_manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Prepared {len(all_rows)} chips.")


if __name__ == "__main__":
    main()
