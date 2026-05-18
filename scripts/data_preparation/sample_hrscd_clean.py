from __future__ import annotations

import argparse
import csv
import random
import time
import zipfile
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from remotezip import RemoteZip
from tqdm import tqdm


HRSCD_URL = "https://huggingface.co/datasets/EPFL-ECEO/HRSCD_clean/resolve/main/HRSCD_Clean.zip"
SUBDIRS = ["images1", "images2", "labels", "labels_map", "landcovers1", "landcovers2"]


def list_stems(remote_zip: zipfile.ZipFile | RemoteZip, split: str) -> list[str]:
    prefix = f"HRSCD_D35/{split}/images1/"
    stems = []
    for info in remote_zip.infolist():
        name = info.filename
        if name.startswith(prefix) and name.lower().endswith(".tif"):
            stems.append(Path(name).stem)
    return sorted(stems)


def extract_member(remote_zip: zipfile.ZipFile | RemoteZip, member: str, output_path: Path, retries: int = 5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with remote_zip.open(member) as src:
                payload = src.read()
            with output_path.open("wb") as dst:
                dst.write(payload)
            return
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            if attempt == retries:
                raise RuntimeError(f"Failed to extract {member}") from exc
            time.sleep(min(30, 2 * attempt))


def read_label_ratio(path: Path) -> float:
    with rasterio.open(path) as dataset:
        label = dataset.read(1)
    return float((label > 0).mean())


def list_label_infos(remote_zip: zipfile.ZipFile | RemoteZip, split: str) -> list[zipfile.ZipInfo]:
    prefix = f"HRSCD_D35/{split}/labels/"
    return [
        info
        for info in remote_zip.infolist()
        if info.filename.startswith(prefix) and info.filename.lower().endswith(".tif")
    ]


def sample_split_balanced(
    remote_zip: zipfile.ZipFile | RemoteZip,
    split: str,
    count: int,
    output_root: Path,
    seed: int,
    changed_fraction: float,
    changed_label_compress_size: int,
) -> list[str]:
    label_infos = list_label_infos(remote_zip, split)
    rng = random.Random(f"{seed}-{split}-balanced")

    target_changed = int(math.ceil(count * changed_fraction))
    target_empty = max(0, count - target_changed)

    changed = [
        Path(info.filename).stem
        for info in label_infos
        if info.compress_size > changed_label_compress_size
    ]
    empty = [
        Path(info.filename).stem
        for info in label_infos
        if info.compress_size <= changed_label_compress_size
    ]
    rng.shuffle(changed)
    rng.shuffle(empty)

    if len(changed) < target_changed:
        print(
            f"Warning: {split} changed candidates are insufficient: "
            f"{len(changed)}/{target_changed}. Filling with available samples."
        )
    if len(empty) < target_empty:
        print(
            f"Warning: {split} empty candidates are insufficient: "
            f"{len(empty)}/{target_empty}. Filling with changed samples."
        )

    selected = changed[:target_changed] + empty[:target_empty]
    selected_stems = set(selected)

    if len(selected) < count:
        for stem in changed + empty:
            if stem in selected_stems:
                continue
            selected.append(stem)
            selected_stems.add(stem)
            if len(selected) >= count:
                break

    selected = sorted(selected[:count])

    print(
        f"{split}: zip-index candidates changed={len(changed)} empty={len(empty)}; "
        f"selected={len(selected)} target_changed={target_changed}"
    )
    extract_selected(remote_zip, split, selected, output_root)
    return selected


def extract_selected(
    remote_zip: zipfile.ZipFile | RemoteZip,
    split: str,
    selected: list[str],
    output_root: Path,
) -> None:
    expected_members = [
        f"HRSCD_D35/{split}/{subdir}/{stem}.tif"
        for stem in selected
        for subdir in SUBDIRS
    ]
    existing = {info.filename for info in remote_zip.infolist()}
    missing = [member for member in expected_members if member not in existing]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} members, first: {missing[0]}")

    for stem in tqdm(selected, desc=f"Extract HRSCD {split}", unit="sample"):
        for subdir in SUBDIRS:
            member = f"HRSCD_D35/{split}/{subdir}/{stem}.tif"
            out = output_root / split / subdir / f"{stem}.tif"
            if out.exists() and out.stat().st_size > 0:
                continue
            extract_member(remote_zip, member, out)


def sample_split(remote_zip: zipfile.ZipFile | RemoteZip, split: str, count: int, output_root: Path, seed: int) -> list[str]:
    stems = list_stems(remote_zip, split)
    rng = random.Random(f"{seed}-{split}")
    rng.shuffle(stems)
    selected = sorted(stems[: min(count, len(stems))])

    extract_selected(remote_zip, split, selected, output_root)
    return selected


def write_manifest(output_root: Path, selected_by_split: dict[str, list[str]]) -> None:
    manifest = output_root / "sample_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        f.write("split,sample_id,images1,images2,labels,labels_map,landcovers1,landcovers2\n")
        for split, stems in selected_by_split.items():
            for stem in stems:
                values = [
                    split,
                    stem,
                    f"{split}/images1/{stem}.tif",
                    f"{split}/images2/{stem}.tif",
                    f"{split}/labels/{stem}.tif",
                    f"{split}/labels_map/{stem}.tif",
                    f"{split}/landcovers1/{stem}.tif",
                    f"{split}/landcovers2/{stem}.tif",
                ]
                f.write(",".join(values) + "\n")


def write_change_records(output_root: Path, records: list[dict]) -> None:
    path = output_root / "sample_change_records.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "sample_id", "change_ratio", "has_change"])
        writer.writeheader()
        writer.writerows(records)


def build_change_records(
    output_root: Path,
    selected_by_split: dict[str, list[str]],
    min_change_ratio: float,
) -> list[dict]:
    records = []
    for split, stems in selected_by_split.items():
        for stem in tqdm(stems, desc=f"Summarize HRSCD {split}", unit="label"):
            label_path = output_root / split / "labels" / f"{stem}.tif"
            ratio = read_label_ratio(label_path)
            records.append(
                {
                    "split": split,
                    "sample_id": stem,
                    "change_ratio": ratio,
                    "has_change": ratio >= min_change_ratio,
                }
            )
    return records


def summarize_records(output_root: Path, records: list[dict]) -> list[dict]:
    summary = []
    for split in ["train", "val", "test"]:
        split_records = [record for record in records if record["split"] == split]
        ratios = [float(record["change_ratio"]) for record in split_records]
        changed = [record for record in split_records if record["has_change"]]
        item = {
            "split": split,
            "patch_count": len(split_records),
            "changed_patch_count": len(changed),
            "empty_label_patch_count": len(split_records) - len(changed),
            "mean_change_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "median_change_ratio": float(np.median(ratios)) if ratios else 0.0,
            "max_change_ratio": float(np.max(ratios)) if ratios else 0.0,
        }
        summary.append(item)

    json_path = output_root / "sample_quality_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
    return summary


def open_remote_zip_with_retries(url: str, retries: int) -> RemoteZip:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Opening remote HRSCD archive (attempt {attempt}/{retries})")
            return RemoteZip(url)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Failed to open remote archive after {retries} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sampled subset from HRSCD-Clean.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None, help="Optional local HRSCD_Clean.zip. Faster and recommended for large samples.")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--balanced", action="store_true", help="Sample a fixed fraction of changed patches per split.")
    parser.add_argument("--changed-fraction", type=float, default=0.5)
    parser.add_argument("--min-change-ratio", type=float, default=1e-5)
    parser.add_argument(
        "--changed-label-compress-size",
        type=int,
        default=300,
        help="HRSCD labels with ZIP compressed size greater than this value are used as changed candidates.",
    )
    parser.add_argument("--remote-open-retries", type=int, default=5)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_by_split: dict[str, list[str]] = {}

    if args.archive:
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        zip_context = zipfile.ZipFile(args.archive)
    else:
        zip_context = open_remote_zip_with_retries(HRSCD_URL, args.remote_open_retries)

    with zip_context as dataset_zip:
        for split, count in [("train", args.train), ("val", args.val), ("test", args.test)]:
            if args.balanced:
                selected = sample_split_balanced(
                    remote_zip=dataset_zip,
                    split=split,
                    count=count,
                    output_root=args.output_root,
                    seed=args.seed,
                    changed_fraction=args.changed_fraction,
                    changed_label_compress_size=args.changed_label_compress_size,
                )
                selected_by_split[split] = selected
            else:
                selected_by_split[split] = sample_split(dataset_zip, split, count, args.output_root, args.seed)

    write_manifest(args.output_root, selected_by_split)
    change_records = build_change_records(args.output_root, selected_by_split, args.min_change_ratio)
    write_change_records(args.output_root, change_records)
    summary = summarize_records(args.output_root, change_records)
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
