from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def stable_value(name: str) -> float:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def image_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def find_two_period_root(raw_root: Path) -> Path:
    candidates = [p for p in raw_root.rglob("*") if p.is_dir() and p.name == "1. The two-period image data"]
    if not candidates:
        raise FileNotFoundError(f"Cannot find WHU two-period image directory under {raw_root}")
    return candidates[0]


def ensure_dirs(out_root: Path) -> None:
    for split in SPLITS:
        for subdir in ("A", "B", "label"):
            (out_root / split / subdir).mkdir(parents=True, exist_ok=True)


def save_mask(mask: np.ndarray, dst: Path) -> dict[str, int | float]:
    out = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(out).save(dst)
    changed = int((out > 0).sum())
    total = int(out.size)
    return {
        "changed_pixels": changed,
        "total_pixels": total,
        "change_ratio": changed / total if total else 0.0,
        "is_empty": int(changed == 0),
    }


def export_patch_sample(
    sample_id: str,
    split: str,
    a_path: Path,
    b_path: Path,
    label_a_path: Path,
    label_b_path: Path,
    out_root: Path,
    patch_size: int,
) -> list[dict[str, str | int | float]]:
    a_img = Image.open(a_path).convert("RGB")
    b_img = Image.open(b_path).convert("RGB")
    label_a = np.asarray(Image.open(label_a_path))
    label_b = np.asarray(Image.open(label_b_path))
    if label_a.ndim == 3:
        label_a = label_a[..., 0]
    if label_b.ndim == 3:
        label_b = label_b[..., 0]
    change = (label_a > 0) != (label_b > 0)

    width, height = a_img.size
    rows: list[dict[str, str | int | float]] = []
    for y0 in range(0, height, patch_size):
        for x0 in range(0, width, patch_size):
            x1 = min(x0 + patch_size, width)
            y1 = min(y0 + patch_size, height)
            if x1 - x0 != patch_size or y1 - y0 != patch_size:
                continue

            patch_id = f"{sample_id}_{y0:04d}_{x0:04d}"
            a_img.crop((x0, y0, x1, y1)).save(out_root / split / "A" / f"{patch_id}.png")
            b_img.crop((x0, y0, x1, y1)).save(out_root / split / "B" / f"{patch_id}.png")
            stats = save_mask(change[y0:y1, x0:x1], out_root / split / "label" / f"{patch_id}.png")
            rows.append(
                {
                    "dataset": "WHU-CD",
                    "split": split,
                    "sample_id": patch_id,
                    "source_id": sample_id,
                    **stats,
                }
            )
    return rows


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary = []
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        total_pixels = sum(int(row["total_pixels"]) for row in split_rows)
        changed_pixels = sum(int(row["changed_pixels"]) for row in split_rows)
        summary.append(
            {
                "split": split,
                "patches": len(split_rows),
                "changed_patches": sum(1 for row in split_rows if int(row["changed_pixels"]) > 0),
                "empty_label_patches": sum(int(row["is_empty"]) for row in split_rows),
                "mean_change_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
            }
        )
    return summary


def export_split(two_period_root: Path, official_split: str, out_root: Path, val_ratio: float, patch_size: int) -> list:
    a_dir = two_period_root / "2012" / "splited_images" / official_split / "image"
    b_dir = two_period_root / "2016" / "splited_images" / official_split / "image"
    label_a_dir = two_period_root / "2012" / "splited_images" / official_split / "label"
    label_b_dir = two_period_root / "2016" / "splited_images" / official_split / "label"

    a_files = image_files(a_dir)
    b_files = image_files(b_dir)
    label_a_files = image_files(label_a_dir)
    label_b_files = image_files(label_b_dir)
    stems = sorted(set(a_files) & set(b_files) & set(label_a_files) & set(label_b_files))
    if not stems:
        raise FileNotFoundError(f"No matched WHU samples found for split={official_split}")

    rows = []
    for index, stem in enumerate(stems, start=1):
        split = "test" if official_split == "test" else ("val" if stable_value(stem) < val_ratio else "train")
        rows.extend(
            export_patch_sample(
                stem,
                split,
                a_files[stem],
                b_files[stem],
                label_a_files[stem],
                label_b_files[stem],
                out_root,
                patch_size,
            )
        )
        if index % 100 == 0:
            print(f"{official_split}: exported {index}/{len(stems)} source tiles", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the official WHU-CD archive to train/val/test A/B/label folders.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_root} already exists; pass --overwrite to replace it.")
        shutil.rmtree(args.out_root)
    ensure_dirs(args.out_root)

    two_period_root = find_two_period_root(args.raw_root)
    rows = []
    rows.extend(export_split(two_period_root, "train", args.out_root, args.val_ratio, args.patch_size))
    rows.extend(export_split(two_period_root, "test", args.out_root, args.val_ratio, args.patch_size))

    summary = summarize(rows)
    with (args.out_root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "WHU-CD",
                "raw_root": str(args.raw_root),
                "output_root": str(args.out_root),
                "label_rule": "XOR of official 2012 and 2016 building masks",
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
