from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    a: Path
    b: Path
    label: Path


def stable_split(name: str, val_ratio: float, test_ratio: float) -> str:
    value = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def find_dirs(root: Path, names: list[str]) -> list[Path]:
    lowered = {name.lower() for name in names}
    return [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in lowered]


def image_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def choose_triplet_dirs(root: Path) -> tuple[Path, Path, Path]:
    a_dirs = find_dirs(root, ["A", "a", "T1", "t1", "time1", "before", "image1"])
    b_dirs = find_dirs(root, ["B", "b", "T2", "t2", "time2", "after", "image2"])
    label_dirs = find_dirs(root, ["label", "labels", "mask", "masks", "change", "change_label", "OUT"])
    best: tuple[int, Path, Path, Path] | None = None
    for a_dir in a_dirs:
        a_files = image_files(a_dir)
        for b_dir in b_dirs:
            b_files = image_files(b_dir)
            common_ab = set(a_files) & set(b_files)
            if not common_ab:
                continue
            for label_dir in label_dirs:
                label_files = image_files(label_dir)
                score = len(common_ab & set(label_files))
                if score and (best is None or score > best[0]):
                    best = (score, a_dir, b_dir, label_dir)
    if best is None:
        raise FileNotFoundError(f"Cannot find matching A/B/label directories under {root}")
    return best[1], best[2], best[3]


def collect_samples(root: Path) -> list[Sample]:
    a_dir, b_dir, label_dir = choose_triplet_dirs(root)
    a_files = image_files(a_dir)
    b_files = image_files(b_dir)
    label_files = image_files(label_dir)
    stems = sorted(set(a_files) & set(b_files) & set(label_files))
    return [Sample(stem, a_files[stem], b_files[stem], label_files[stem]) for stem in stems]


def normalize_label(src: Path, dst: Path) -> dict[str, float | int]:
    arr = np.asarray(Image.open(src))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(dst)
    changed = int((mask > 0).sum())
    total = int(mask.size)
    return {
        "changed_pixels": changed,
        "total_pixels": total,
        "change_ratio": changed / total if total else 0.0,
        "is_empty": int(changed == 0),
    }


def copy_sample(sample: Sample, split_dir: Path) -> dict[str, float | int]:
    a_dst = split_dir / "A" / f"{sample.sample_id}.png"
    b_dst = split_dir / "B" / f"{sample.sample_id}.png"
    label_dst = split_dir / "label" / f"{sample.sample_id}.png"
    a_dst.parent.mkdir(parents=True, exist_ok=True)
    b_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.a, a_dst)
    shutil.copy2(sample.b, b_dst)
    return normalize_label(sample.label, label_dst)


def ensure_split_dirs(dataset_out: Path) -> None:
    for split in SPLITS:
        for subdir in ("A", "B", "label"):
            (dataset_out / split / subdir).mkdir(parents=True, exist_ok=True)


def extract_archives(raw_root: Path) -> None:
    for archive in raw_root.rglob("*.zip"):
        target = archive.with_suffix("")
        marker = target / ".extracted"
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        marker.write_text("ok\n", encoding="utf-8")


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    result = []
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        total_pixels = sum(int(row["total_pixels"]) for row in split_rows)
        changed_pixels = sum(int(row["changed_pixels"]) for row in split_rows)
        result.append(
            {
                "split": split,
                "patches": len(split_rows),
                "changed_patches": sum(1 for row in split_rows if int(row["changed_pixels"]) > 0),
                "empty_label_patches": sum(int(row["is_empty"]) for row in split_rows),
                "mean_change_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
            }
        )
    return result


def organize_dataset(
    name: str,
    raw_root: Path,
    out_root: Path,
    val_ratio: float,
    test_ratio: float,
    split_from_dirs: bool,
) -> dict[str, object]:
    dataset_out = out_root / name
    if dataset_out.exists():
        shutil.rmtree(dataset_out)
    ensure_split_dirs(dataset_out)

    samples_by_split: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    if split_from_dirs and any((raw_root / split).exists() for split in SPLITS):
        for split in SPLITS:
            split_root = raw_root / split
            if split_root.exists():
                samples_by_split[split] = collect_samples(split_root)
    else:
        for sample in collect_samples(raw_root):
            samples_by_split[stable_split(sample.sample_id, val_ratio, test_ratio)].append(sample)

    rows: list[dict[str, str | int | float]] = []
    for split, samples in samples_by_split.items():
        for sample in samples:
            stats = copy_sample(sample, dataset_out / split)
            rows.append({"dataset": name, "split": split, "sample_id": sample.sample_id, **stats})

    summary = summarize(rows)
    with (dataset_out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset", "split", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    with (dataset_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset": name, "raw_root": str(raw_root), "summary": summary}, f, ensure_ascii=False, indent=2)
    return {"dataset": name, "raw_root": str(raw_root), "output_root": str(dataset_out), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize binary CD datasets into split/A,B,label layout.")
    parser.add_argument("--public-root", type=Path, default=Path(r"D:\桌面\文献\论文\公开数据集"))
    parser.add_argument("--dataset", choices=["LEVIR-CD", "WHU-CD", "SYSU-CD", "all"], default="all")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--extract-zip", action="store_true")
    args = parser.parse_args()

    out_root = args.public_root / "datasets"
    datasets = ["LEVIR-CD", "WHU-CD", "SYSU-CD"] if args.dataset == "all" else [args.dataset]
    results = []
    for dataset in datasets:
        raw_root = args.public_root / dataset
        if args.extract_zip:
            extract_archives(raw_root)
        results.append(
            organize_dataset(
                dataset,
                raw_root,
                out_root,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_from_dirs=True,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
