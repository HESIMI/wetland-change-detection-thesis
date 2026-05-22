from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
SUBDIRS = ("A", "B", "label")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def export_dataset(src_root: Path, out_root: Path, mode: str) -> dict[str, object]:
    if out_root.exists():
        shutil.rmtree(out_root)
    for subdir in (*SUBDIRS, "list"):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    summary = []
    for split in SPLITS:
        split_names = []
        src_dirs = {subdir: src_root / split / subdir for subdir in SUBDIRS}
        missing = [str(path) for path in src_dirs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing source directories: {missing}")

        stems = sorted(p.stem for p in src_dirs["A"].glob("*") if p.is_file())
        for stem in stems:
            src_paths = {subdir: next(src_dirs[subdir].glob(f"{stem}.*"), None) for subdir in SUBDIRS}
            if any(path is None for path in src_paths.values()):
                continue
            name = f"{split}_{stem}.png"
            for subdir, src_path in src_paths.items():
                assert src_path is not None
                link_or_copy(src_path, out_root / subdir / name, mode)
            split_names.append(name)
            rows.append({"split": split, "name": name})

        (out_root / "list" / f"{split}.txt").write_text("\n".join(split_names) + "\n", encoding="utf-8")
        summary.append({"split": split, "samples": len(split_names)})

    with (out_root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "name"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_root": str(src_root),
                "output_root": str(out_root),
                "layout": "CDMamba official A/B/label/list",
                "mode": mode,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"source_root": str(src_root), "output_root": str(out_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert split/A,B,label data to CDMamba official A/B/label/list layout.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    result = export_dataset(args.src_root, args.out_root, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
