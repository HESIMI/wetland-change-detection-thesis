from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


SECOND_GOOGLE_DRIVE_ID = "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u"
HRSCD_HF_REPO = "EPFL-ECEO/HRSCD_clean"
HRSCD_FILENAME = "HRSCD_Clean.zip"


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def download_second(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "second_dataset.zip"
    if not archive.exists():
        run(
            [
                "python",
                "-m",
                "gdown",
                f"https://drive.google.com/uc?id={SECOND_GOOGLE_DRIVE_ID}",
                "-O",
                str(archive),
            ]
        )
    print(f"SECOND archive: {archive}")


def download_hrscd(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            "-c",
            (
                "from huggingface_hub import hf_hub_download;"
                f"hf_hub_download(repo_id='{HRSCD_HF_REPO}', repo_type='dataset', "
                f"filename='{HRSCD_FILENAME}', local_dir=r'{root}')"
            ),
        ]
    )
    print(f"HRSCD-Clean archive: {root / HRSCD_FILENAME}")


def extract_second(root: Path) -> None:
    archive_contents = root / "archive_contents"
    extracted = root / "extracted_7z"
    archive_contents.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    second_archive = root / "second_dataset.zip"
    if not second_archive.exists():
        raise FileNotFoundError(second_archive)

    run(["powershell", "-NoProfile", "-Command", f"Expand-Archive -LiteralPath '{second_archive}' -DestinationPath '{archive_contents}' -Force"])

    seven_zip = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
    train_rar = archive_contents / "SECOND_train_set.rar"
    test_zip = archive_contents / "SECOND_total_test.zip"
    if not Path(seven_zip).exists() and shutil.which("7z") is None:
        raise FileNotFoundError("7-Zip is required to extract SECOND_train_set.rar.")

    run([seven_zip, "x", str(train_rar), f"-o{extracted}", "-y"])
    run([seven_zip, "x", str(test_zip), f"-o{extracted}", "-y"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by the thesis experiments.")
    parser.add_argument("--dataset", choices=["second", "hrscd", "all"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/public"))
    parser.add_argument("--extract-second", action="store_true")
    args = parser.parse_args()

    if args.dataset in {"second", "all"}:
        second_root = args.root / "SECOND"
        download_second(second_root)
        if args.extract_second:
            extract_second(second_root)

    if args.dataset in {"hrscd", "all"}:
        download_hrscd(args.root / "HRSCD_clean")


if __name__ == "__main__":
    main()

