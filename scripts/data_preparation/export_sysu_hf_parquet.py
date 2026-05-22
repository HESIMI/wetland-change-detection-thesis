from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SPLITS = ("train", "val", "test")


def image_bytes(value) -> bytes:
    if isinstance(value, dict) and "bytes" in value:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def save_rgb(value, path: Path) -> None:
    image = Image.open(io.BytesIO(image_bytes(value))).convert("RGB")
    image.save(path)


def save_label(value, path: Path) -> dict[str, int | float]:
    image = Image.open(io.BytesIO(image_bytes(value)))
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)

    changed = int((mask > 0).sum())
    total = int(mask.size)
    return {
        "changed_pixels": changed,
        "total_pixels": total,
        "change_ratio": changed / total if total else 0.0,
        "is_empty": int(changed == 0),
    }


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


def export_split(parquet_root: Path, out_root: Path, split: str) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    split_files = sorted(parquet_root.glob(f"{split}-*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"No parquet files found for split={split} under {parquet_root}")

    for subdir in ("A", "B", "label"):
        (out_root / split / subdir).mkdir(parents=True, exist_ok=True)

    sample_index = 0
    for parquet_path in split_files:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        for row_index, row in df.iterrows():
            sample_id = f"{split}_{sample_index:06d}"
            save_rgb(row["imageA"], out_root / split / "A" / f"{sample_id}.png")
            save_rgb(row["imageB"], out_root / split / "B" / f"{sample_id}.png")
            stats = save_label(row["label"], out_root / split / "label" / f"{sample_id}.png")
            rows.append(
                {
                    "dataset": "SYSU-CD",
                    "split": split,
                    "sample_id": sample_id,
                    "source_file": parquet_path.name,
                    "source_row": int(row_index),
                    **stats,
                }
            )
            sample_index += 1
        print(f"{split}: exported {sample_index} samples", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HuggingFace SYSU_CD parquet files to A/B/label folders.")
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_root} already exists; pass --overwrite to replace it.")
        import shutil

        shutil.rmtree(args.out_root)

    rows: list[dict[str, str | int | float]] = []
    for split in SPLITS:
        rows.extend(export_split(args.parquet_root, args.out_root, split))

    summary = summarize(rows)
    with (args.out_root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "SYSU-CD",
                "parquet_root": str(args.parquet_root),
                "output_root": str(args.out_root),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
