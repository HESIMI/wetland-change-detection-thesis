from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

try:
    from .public_datasets import PublicSemanticChangeDataset
except ImportError:
    from public_datasets import PublicSemanticChangeDataset


def inspect_dataset(dataset: str, root: Path, split: str, batch_size: int) -> None:
    ds = PublicSemanticChangeDataset(dataset=dataset, root=root, split=split)
    sample = ds[0]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"dataset={dataset}")
    print(f"root={root}")
    print(f"split={split}")
    print(f"samples={len(ds)}")
    print(f"sample_id={sample['sample_id']}")
    print(f"t1={tuple(sample['t1'].shape)}")
    print(f"t2={tuple(sample['t2'].shape)}")
    print(f"image={tuple(sample['image'].shape)}")
    print(f"binary_mask={tuple(sample['binary_mask'].shape)}")
    print(f"semantic_t1={tuple(sample['semantic_t1'].shape)}")
    print(f"semantic_t2={tuple(sample['semantic_t2'].shape)}")
    print(f"batch_image={tuple(batch['image'].shape)}")
    print(f"batch_binary_mask={tuple(batch['binary_mask'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect public semantic change detection datasets.")
    parser.add_argument("--dataset", choices=["second", "hrscd"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    inspect_dataset(args.dataset, args.root, args.split, args.batch_size)


if __name__ == "__main__":
    main()

