from pathlib import Path

try:
    from .dataset import build_datasets
except ImportError:
    from dataset import build_datasets


DEFAULT_MANIFEST = Path("D:/桌面/毕业论文/项目/data/processed/dataset_manifest.csv")


def main() -> None:
    datasets = build_datasets(DEFAULT_MANIFEST, include_prompt=True)
    for split, ds in datasets.items():
        sample = ds[0]
        print(
            split,
            len(ds),
            sample["sample_id"],
            tuple(sample["t1"].shape),
            tuple(sample["binary_mask"].shape),
            sample.get("dominant_transition", ""),
        )


if __name__ == "__main__":
    main()
