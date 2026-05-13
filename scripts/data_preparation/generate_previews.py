from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def percentile_stretch(arr: np.ndarray, p2: float = 2, p98: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.nanpercentile(arr, p2)
    hi = np.nanpercentile(arr, p98)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def save_rgb_preview(tif_path: Path, png_path: Path, bands: tuple[int, int, int]) -> None:
    with rasterio.open(tif_path) as src:
        arr = src.read(list(bands))
    rgb = np.stack([percentile_stretch(arr[i]) for i in range(3)], axis=-1)
    Image.fromarray(rgb).save(png_path)


def colorize_label(arr: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        rgb[arr == value] = color
    return rgb


def save_label_preview(tif_path: Path, png_path: Path, color_map: dict[int, tuple[int, int, int]]) -> None:
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
    rgb = colorize_label(arr, color_map)
    Image.fromarray(rgb).save(png_path)


def main() -> None:
    root = Path.cwd()
    out_dir = root / "项目" / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_colors = {
        10: (230, 210, 80),
        11: (235, 200, 60),
        20: (34, 139, 34),
        52: (65, 105, 225),
        61: (194, 178, 128),
        62: (170, 150, 120),
        72: (220, 240, 255),
        130: (188, 170, 110),
        181: (0, 153, 102),
        182: (46, 139, 87),
        183: (102, 205, 170),
        186: (72, 179, 151),
        187: (95, 158, 160),
        190: (220, 20, 60),
        210: (176, 196, 222),
    }
    binary_colors = {
        0: (25, 25, 25),
        1: (255, 80, 80),
    }
    pseudo_colors = {
        0: (25, 25, 25),
        1: (255, 200, 0),
    }

    tasks = [
        ("hangzhou_xixi", "raw", "sentinel2_2018.tif", "hangzhou_xixi_s2_2018.png", (3, 2, 1), None),
        ("hangzhou_xixi", "raw", "sentinel2_2022.tif", "hangzhou_xixi_s2_2022.png", (3, 2, 1), None),
        ("poyang_lake", "raw", "sentinel2_2018.tif", "poyang_lake_s2_2018.png", (3, 2, 1), None),
        ("poyang_lake", "raw", "sentinel2_2022.tif", "poyang_lake_s2_2022.png", (3, 2, 1), None),
        ("hangzhou_xixi", "raw", "glc_2018.tif", "hangzhou_xixi_glc_2018.png", None, semantic_colors),
        ("hangzhou_xixi", "raw", "glc_2022.tif", "hangzhou_xixi_glc_2022.png", None, semantic_colors),
        ("hangzhou_xixi", "change_labels", "binary_change_final.tif", "hangzhou_xixi_change_final.png", None, binary_colors),
        ("hangzhou_xixi", "change_labels", "pseudo_change_mask.tif", "hangzhou_xixi_pseudo_change.png", None, pseudo_colors),
        ("poyang_lake", "change_labels", "binary_change_final.tif", "poyang_lake_change_final.png", None, binary_colors),
        ("poyang_lake", "change_labels", "pseudo_change_mask.tif", "poyang_lake_pseudo_change.png", None, pseudo_colors),
    ]

    for area, category, tif_name, png_name, bands, cmap in tasks:
        tif_path = root / "项目" / "data" / category / area / tif_name
        png_path = out_dir / png_name
        if bands:
            save_rgb_preview(tif_path, png_path, bands)
        else:
            save_label_preview(tif_path, png_path, cmap)
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
