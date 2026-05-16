from __future__ import annotations

import argparse
import random
import time
import zipfile
from pathlib import Path

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


def sample_split(remote_zip: zipfile.ZipFile | RemoteZip, split: str, count: int, output_root: Path, seed: int) -> list[str]:
    stems = list_stems(remote_zip, split)
    rng = random.Random(f"{seed}-{split}")
    rng.shuffle(stems)
    selected = sorted(stems[: min(count, len(stems))])

    expected_members = [
        f"HRSCD_D35/{split}/{subdir}/{stem}.tif"
        for stem in selected
        for subdir in SUBDIRS
    ]
    existing = {info.filename for info in remote_zip.infolist()}
    missing = [member for member in expected_members if member not in existing]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} members, first: {missing[0]}")

    for stem in tqdm(selected, desc=f"HRSCD {split}", unit="sample"):
        for subdir in SUBDIRS:
            member = f"HRSCD_D35/{split}/{subdir}/{stem}.tif"
            out = output_root / split / subdir / f"{stem}.tif"
            if out.exists() and out.stat().st_size > 0:
                continue
            extract_member(remote_zip, member, out)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sampled subset from HRSCD-Clean.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None, help="Optional local HRSCD_Clean.zip. Faster and recommended for large samples.")
    parser.add_argument("--train", type=int, default=2500)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--test", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_by_split: dict[str, list[str]] = {}

    if args.archive:
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        zip_context = zipfile.ZipFile(args.archive)
    else:
        zip_context = RemoteZip(HRSCD_URL)

    with zip_context as dataset_zip:
        for split, count in [("train", args.train), ("val", args.val), ("test", args.test)]:
            selected_by_split[split] = sample_split(dataset_zip, split, count, args.output_root, args.seed)

    write_manifest(args.output_root, selected_by_split)
    print(args.output_root)


if __name__ == "__main__":
    main()
