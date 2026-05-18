from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


SUBDIRS = ["images1", "images2", "labels", "labels_map", "landcovers1", "landcovers2"]


@dataclass(frozen=True)
class SampleInfo:
    source_split: str
    sample_id: str
    change_ratio: float

    @property
    def has_change(self) -> bool:
        return self.change_ratio > 0


def read_change_ratio(label_path: Path) -> float:
    with rasterio.open(label_path) as dataset:
        label = dataset.read(1)
    return float((label > 0).mean())


def list_complete_samples(source_root: Path, source_splits: list[str]) -> list[SampleInfo]:
    samples: list[SampleInfo] = []
    for split in source_splits:
        label_dir = source_root / split / "labels"
        if not label_dir.exists():
            continue

        for label_path in tqdm(sorted(label_dir.glob("*.tif")), desc=f"Scan HRSCD {split}", unit="label"):
            stem = label_path.stem
            if all((source_root / split / subdir / f"{stem}.tif").exists() for subdir in SUBDIRS):
                samples.append(
                    SampleInfo(
                        source_split=split,
                        sample_id=stem,
                        change_ratio=read_change_ratio(label_path),
                    )
                )
    return samples


def select_balanced(
    samples: list[SampleInfo],
    split_counts: dict[str, int],
    changed_fraction: float,
    seed: int,
    min_change_ratio: float,
) -> dict[str, list[SampleInfo]]:
    rng = random.Random(seed)
    changed = [sample for sample in samples if sample.change_ratio >= min_change_ratio]
    empty = [sample for sample in samples if sample.change_ratio < min_change_ratio]
    rng.shuffle(changed)
    rng.shuffle(empty)

    selected_by_split: dict[str, list[SampleInfo]] = {}
    for split, count in split_counts.items():
        changed_count = min(round(count * changed_fraction), len(changed))
        empty_count = min(count - changed_count, len(empty))
        selected = changed[:changed_count] + empty[:empty_count]
        del changed[:changed_count]
        del empty[:empty_count]

        while len(selected) < count and (changed or empty):
            pool = changed if changed else empty
            selected.append(pool.pop(0))

        if len(selected) < count:
            raise RuntimeError(f"Not enough complete HRSCD samples for {split}: {len(selected)}/{count}")

        rng.shuffle(selected)
        selected_by_split[split] = selected
    return selected_by_split


def copy_selected(source_root: Path, output_root: Path, selected_by_split: dict[str, list[SampleInfo]]) -> None:
    for split, samples in selected_by_split.items():
        for sample in tqdm(samples, desc=f"Copy HRSCD {split}", unit="sample"):
            for subdir in SUBDIRS:
                src = source_root / sample.source_split / subdir / f"{sample.sample_id}.tif"
                dst = output_root / split / subdir / f"{sample.sample_id}.tif"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def write_manifest(output_root: Path, selected_by_split: dict[str, list[SampleInfo]]) -> None:
    manifest = output_root / "sample_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_id", "images1", "images2", "labels", "labels_map", "landcovers1", "landcovers2"])
        for split, samples in selected_by_split.items():
            for sample in samples:
                writer.writerow(
                    [
                        split,
                        sample.sample_id,
                        f"{split}/images1/{sample.sample_id}.tif",
                        f"{split}/images2/{sample.sample_id}.tif",
                        f"{split}/labels/{sample.sample_id}.tif",
                        f"{split}/labels_map/{sample.sample_id}.tif",
                        f"{split}/landcovers1/{sample.sample_id}.tif",
                        f"{split}/landcovers2/{sample.sample_id}.tif",
                    ]
                )


def write_change_records(output_root: Path, selected_by_split: dict[str, list[SampleInfo]], min_change_ratio: float) -> list[dict]:
    records: list[dict] = []
    for split, samples in selected_by_split.items():
        for sample in samples:
            records.append(
                {
                    "split": split,
                    "sample_id": sample.sample_id,
                    "source_split": sample.source_split,
                    "change_ratio": sample.change_ratio,
                    "has_change": sample.change_ratio >= min_change_ratio,
                }
            )

    records_path = output_root / "sample_change_records.csv"
    with records_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_id", "source_split", "change_ratio", "has_change"],
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def write_summary(output_root: Path, records: list[dict]) -> list[dict]:
    summary = []
    for split in ["train", "val", "test"]:
        split_records = [record for record in records if record["split"] == split]
        ratios = [float(record["change_ratio"]) for record in split_records]
        changed_count = sum(bool(record["has_change"]) for record in split_records)
        summary.append(
            {
                "split": split,
                "patch_count": len(split_records),
                "changed_patch_count": changed_count,
                "mean_change_ratio": float(np.mean(ratios)) if ratios else 0.0,
                "empty_label_patch_count": len(split_records) - changed_count,
                "median_change_ratio": float(np.median(ratios)) if ratios else 0.0,
                "max_change_ratio": float(np.max(ratios)) if ratios else 0.0,
            }
        )

    csv_path = output_root / "sample_quality_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "patch_count",
                "changed_patch_count",
                "mean_change_ratio",
                "empty_label_patch_count",
                "median_change_ratio",
                "max_change_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    json_path = output_root / "sample_quality_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebalance an extracted HRSCD sample into stratified train/val/test splits.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--train", type=int, default=220)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--changed-fraction", type=float, default=0.5)
    parser.add_argument("--min-change-ratio", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = list_complete_samples(args.source_root, args.source_splits)
    selected_by_split = select_balanced(
        samples=samples,
        split_counts={"train": args.train, "val": args.val, "test": args.test},
        changed_fraction=args.changed_fraction,
        seed=args.seed,
        min_change_ratio=args.min_change_ratio,
    )
    copy_selected(args.source_root, args.output_root, selected_by_split)
    write_manifest(args.output_root, selected_by_split)
    records = write_change_records(args.output_root, selected_by_split, args.min_change_ratio)
    summary = write_summary(args.output_root, records)

    for item in summary:
        print(
            f"{item['split']}: patches={item['patch_count']} "
            f"changed={item['changed_patch_count']} "
            f"empty={item['empty_label_patch_count']} "
            f"mean_change={item['mean_change_ratio']:.6f}"
        )
    print(args.output_root)


if __name__ == "__main__":
    main()
