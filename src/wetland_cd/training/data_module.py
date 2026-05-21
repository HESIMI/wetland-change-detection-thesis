from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


SPLITS = ["train", "val", "test"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def build_datasets_from_config(config: dict[str, Any]) -> dict[str, object]:
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"].lower()
    image_size = data_cfg.get("image_size")

    if dataset_name == "wetland":
        try:
            from .dataset import WetlandChangeDataset
        except ImportError:
            from dataset import WetlandChangeDataset

        manifest = resolve_path(data_cfg["manifest"])
        return {
            split: WetlandChangeDataset(
                manifest_path=manifest,
                split=split,
                include_prompt=bool(data_cfg.get("include_prompt", False)),
                output_size=image_size,
            )
            for split in SPLITS
        }

    if dataset_name in {"second", "hrscd", "levir-cd", "whu-cd", "sysu-cd"}:
        try:
            from .public_datasets import PublicSemanticChangeDataset
        except ImportError:
            from public_datasets import PublicSemanticChangeDataset

        root = resolve_path(data_cfg["root"])
        return {
            split: PublicSemanticChangeDataset(
                dataset=dataset_name,
                root=root,
                split=split,
                output_size=image_size,
            )
            for split in SPLITS
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_loaders_from_config(config: dict[str, Any]) -> tuple[dict[str, object], dict[str, DataLoader]]:
    train_cfg = config["training"]
    datasets = build_datasets_from_config(config)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            drop_last=(split == "train" and bool(train_cfg.get("drop_last", False))),
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders
