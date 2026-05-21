from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


def default_rgb_transform(image: torch.Tensor) -> torch.Tensor:
    return image.float() / 255.0


def default_binary_mask_transform(mask: torch.Tensor) -> torch.Tensor:
    return mask.float()


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _resize_chw(image: torch.Tensor, output_size: int | None, mode: str) -> torch.Tensor:
    if output_size is None or image.shape[-2:] == (output_size, output_size):
        return image
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(image.unsqueeze(0), size=(output_size, output_size), **kwargs).squeeze(0)


def _rgb_to_ids(label: np.ndarray) -> torch.Tensor:
    """Convert an RGB semantic map to stable integer ids within the sample."""
    flat = label.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Sort colors lexicographically so ids are deterministic for the same palette.
    order = np.lexsort((colors[:, 2], colors[:, 1], colors[:, 0]))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    ids = remap[inverse].reshape(label.shape[:2]).astype(np.int64)
    return torch.from_numpy(ids)


def _deterministic_partition(name: str, val_ratio: float = 0.15) -> Split:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


@dataclass(frozen=True)
class PublicChangeSample:
    sample_id: str
    t1_path: Path
    t2_path: Path
    label1_path: Path | None
    label2_path: Path | None
    binary_path: Path | None = None
    split: str = ""


class PublicSemanticChangeDataset(Dataset):
    """Unified reader for public semantic change detection datasets.

    Returned fields follow the same convention as the in-house wetland dataset:
    t1, t2, image, binary_mask, semantic_t1, semantic_t2, sample_id and dataset.
    """

    def __init__(
        self,
    dataset: Literal["second", "hrscd", "levir-cd", "whu-cd", "sysu-cd"],
        root: str | Path,
        split: Split,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        output_size: int | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform or default_rgb_transform
        self.mask_transform = mask_transform or default_binary_mask_transform
        self.output_size = output_size

        if self.dataset == "second":
            self.samples = _build_second_samples(self.root, split)
        elif self.dataset == "hrscd":
            self.samples = _build_hrscd_samples(self.root, split)
        elif self.dataset in {"levir-cd", "whu-cd", "sysu-cd"}:
            self.samples = _build_binary_cd_samples(self.root, self.dataset, split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if not self.samples:
            raise FileNotFoundError(f"No samples found for dataset={dataset}, split={split}, root={self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        t1_rgb = _read_rgb(sample.t1_path)
        t2_rgb = _read_rgb(sample.t2_path)

        t1 = self.image_transform(_to_chw_tensor(t1_rgb))
        t2 = self.image_transform(_to_chw_tensor(t2_rgb))

        semantic_t1 = _load_semantic(sample.label1_path, t1_rgb.shape[:2])
        semantic_t2 = _load_semantic(sample.label2_path, t2_rgb.shape[:2])
        binary_mask = _load_binary(sample.binary_path, sample.label1_path, sample.label2_path, t1_rgb.shape[:2])
        binary_mask = self.mask_transform(binary_mask.unsqueeze(0))
        t1 = _resize_chw(t1, self.output_size, "bilinear")
        t2 = _resize_chw(t2, self.output_size, "bilinear")
        binary_mask = _resize_chw(binary_mask, self.output_size, "nearest")
        semantic_t1 = _resize_chw(semantic_t1.unsqueeze(0).float(), self.output_size, "nearest").squeeze(0).long()
        semantic_t2 = _resize_chw(semantic_t2.unsqueeze(0).float(), self.output_size, "nearest").squeeze(0).long()

        return {
            "dataset": self.dataset,
            "sample_id": sample.sample_id,
            "split": sample.split or self.split,
            "t1": t1,
            "t2": t2,
            "image": torch.cat([t1, t2], dim=0),
            "binary_mask": binary_mask,
            "semantic_t1": semantic_t1,
            "semantic_t2": semantic_t2,
        }


def _load_semantic(path: Path | None, shape: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.zeros(shape, dtype=torch.long)

    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        return torch.from_numpy(image.astype(np.int64))
    return _rgb_to_ids(image[..., :3])


def _load_binary(
    binary_path: Path | None,
    label1_path: Path | None,
    label2_path: Path | None,
    shape: tuple[int, int],
) -> torch.Tensor:
    if binary_path is not None and binary_path.exists():
        mask = np.asarray(Image.open(binary_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return torch.from_numpy((mask > 0).astype(np.float32))

    if label1_path is not None and label2_path is not None:
        label1 = np.asarray(Image.open(label1_path))
        label2 = np.asarray(Image.open(label2_path))
        if label1.ndim == 2 and label2.ndim == 2:
            changed = label1 != label2
        else:
            changed = np.any(label1[..., :3] != label2[..., :3], axis=2)
        return torch.from_numpy(changed.astype(np.float32))

    return torch.zeros(shape, dtype=torch.float32)


def _build_second_samples(root: Path, split: Split) -> list[PublicChangeSample]:
    base = _resolve_second_base(root)
    if split == "test":
        sample_root = base / "test"
        ids = _common_stems(sample_root / "im1", sample_root / "im2", sample_root / "label1", sample_root / "label2")
        return [
            PublicChangeSample(
                sample_id=stem,
                t1_path=sample_root / "im1" / f"{stem}.png",
                t2_path=sample_root / "im2" / f"{stem}.png",
                label1_path=sample_root / "label1" / f"{stem}.png",
                label2_path=sample_root / "label2" / f"{stem}.png",
                split="test",
            )
            for stem in ids
        ]

    ids = _common_stems(base / "im1", base / "im2", base / "label1", base / "label2")
    selected = [stem for stem in ids if _deterministic_partition(stem) == split]
    return [
        PublicChangeSample(
            sample_id=stem,
            t1_path=base / "im1" / f"{stem}.png",
            t2_path=base / "im2" / f"{stem}.png",
            label1_path=base / "label1" / f"{stem}.png",
            label2_path=base / "label2" / f"{stem}.png",
            split=split,
        )
        for stem in selected
    ]


def _resolve_second_base(root: Path) -> Path:
    candidates = [
        root,
        root / "extracted_7z",
        root / "extracted",
        root / "SECOND" / "extracted_7z",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in ["im1", "im2", "label1", "label2"]):
            return candidate
    raise FileNotFoundError(f"Cannot find SECOND im1/im2/label1/label2 under {root}")


def _build_hrscd_samples(root: Path, split: Split) -> list[PublicChangeSample]:
    base = _resolve_hrscd_base(root)
    split_base = _find_split_base(base, split)

    t1_dir = _find_first_dir(split_base, ["im1", "image1", "images1", "A", "t1", "T1", "imgs_1"])
    t2_dir = _find_first_dir(split_base, ["im2", "image2", "images2", "B", "t2", "T2", "imgs_2"])
    label1_dir = _find_optional_dir(split_base, ["label1", "labels1", "map1", "lcm1", "landcovers1", "seg1", "masks_1"])
    label2_dir = _find_optional_dir(split_base, ["label2", "labels2", "map2", "lcm2", "landcovers2", "seg2", "masks_2"])
    binary_dir = _find_optional_dir(split_base, ["change", "change_label", "cm", "binary", "change_masks", "mask", "labels"])

    required_dirs = [t1_dir, t2_dir]
    optional_dirs = [d for d in [label1_dir, label2_dir, binary_dir] if d is not None]
    stems = _common_stems(*required_dirs, *optional_dirs)

    return [
        PublicChangeSample(
            sample_id=stem,
            t1_path=_find_file(t1_dir, stem),
            t2_path=_find_file(t2_dir, stem),
            label1_path=_find_file(label1_dir, stem) if label1_dir else None,
            label2_path=_find_file(label2_dir, stem) if label2_dir else None,
            binary_path=_find_file(binary_dir, stem) if binary_dir else None,
            split=split,
        )
        for stem in stems
    ]


def _resolve_hrscd_base(root: Path) -> Path:
    candidates = [root, root / "HRSCD_Clean", root / "HRSCD_clean", root / "extracted"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find HRSCD root under {root}")


def _build_binary_cd_samples(root: Path, dataset: str, split: Split) -> list[PublicChangeSample]:
    base = _resolve_binary_cd_base(root, dataset)
    split_base = base / split
    if not split_base.exists():
        raise FileNotFoundError(f"Cannot find split={split} under {base}")
    t1_dir = _find_first_dir(split_base, ["A", "a", "im1", "image1", "T1", "t1"])
    t2_dir = _find_first_dir(split_base, ["B", "b", "im2", "image2", "T2", "t2"])
    binary_dir = _find_first_dir(split_base, ["label", "labels", "mask", "masks", "change", "change_label"])
    stems = _common_stems(t1_dir, t2_dir, binary_dir)
    return [
        PublicChangeSample(
            sample_id=stem,
            t1_path=_find_file(t1_dir, stem),
            t2_path=_find_file(t2_dir, stem),
            label1_path=None,
            label2_path=None,
            binary_path=_find_file(binary_dir, stem),
            split=split,
        )
        for stem in stems
    ]


def _resolve_binary_cd_base(root: Path, dataset: str) -> Path:
    candidates = [
        root,
        root / dataset,
        root / dataset.upper(),
        root / dataset.replace("-", "_"),
        root / "datasets" / dataset,
        root / "datasets" / dataset.upper(),
    ]
    for candidate in candidates:
        if all((candidate / split).exists() for split in ["train", "val", "test"]):
            return candidate
    raise FileNotFoundError(f"Cannot find normalized {dataset} train/val/test under {root}")


def _find_split_base(root: Path, split: Split) -> Path:
    direct = root / split
    if direct.exists():
        return direct
    for candidate in root.rglob(split):
        if candidate.is_dir():
            return candidate
    return root


def _find_first_dir(root: Path, names: list[str]) -> Path:
    result = _find_optional_dir(root, names)
    if result is None:
        raise FileNotFoundError(f"Cannot find any of {names} under {root}")
    return result


def _find_optional_dir(root: Path, names: list[str]) -> Path | None:
    lower_names = {name.lower() for name in names}
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and path.name.lower() in lower_names:
            return path
    return None


def _common_stems(*dirs: Path) -> list[str]:
    stem_sets = []
    for directory in dirs:
        files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
        stem_sets.append({p.stem for p in files})
    return sorted(set.intersection(*stem_sets)) if stem_sets else []


def _find_file(directory: Path | None, stem: str) -> Path:
    if directory is None:
        raise FileNotFoundError(stem)
    for suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")
