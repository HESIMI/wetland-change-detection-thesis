import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_manifest_path(raw_path: str | Path) -> Path:
    path_str = str(raw_path).strip()
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    normalized = path_str.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        remapped = project_root() / "data" / Path(relative)
        if remapped.exists():
            return remapped

    training_marker = "/training/"
    if training_marker in normalized:
        relative = normalized.split(training_marker, 1)[1]
        remapped = project_root() / "training" / Path(relative)
        if remapped.exists():
            return remapped

    return candidate


def default_image_transform(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    image = torch.clamp(image / 10000.0, 0.0, 1.0)
    return image


def default_mask_transform(mask: torch.Tensor) -> torch.Tensor:
    return mask.float()


class WetlandChangeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        include_prompt: bool = False,
    ) -> None:
        manifest_path = Path(manifest_path)
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
        self.rows = [row for row in rows if row["split"] == split]
        self.image_transform = image_transform or default_image_transform
        self.mask_transform = mask_transform or default_mask_transform
        self.include_prompt = include_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        row = self.rows[index]

        image_path = _normalize_manifest_path(row["image_path"])
        label_path = _normalize_manifest_path(row["label_path"])

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            label = src.read()

        image = torch.from_numpy(image)
        t1 = image[: image.shape[0] // 2]
        t2 = image[image.shape[0] // 2 :]
        binary_mask = torch.from_numpy(label[0].astype(np.float32)).unsqueeze(0)
        semantic_mask = torch.from_numpy(label[1].astype(np.int64))

        t1 = self.image_transform(t1)
        t2 = self.image_transform(t2)
        binary_mask = self.mask_transform(binary_mask)

        sample = {
            "sample_id": row["sample_id"],
            "area": row["area"],
            "t1": t1,
            "t2": t2,
            "image": torch.cat([t1, t2], dim=0),
            "binary_mask": binary_mask,
            "semantic_mask": semantic_mask,
            "binary_change_ratio": torch.tensor(float(row["binary_change_ratio"]), dtype=torch.float32),
        }

        if self.include_prompt:
            prompt_path = _normalize_manifest_path(row["prompt_path"])
            prompt_payload = json.loads(prompt_path.read_text(encoding="utf-8"))
            sample["prompts"] = prompt_payload.get("prompts", [])
            sample["dominant_transition"] = prompt_payload.get("dominant_transition", "")

        return sample


def build_datasets(manifest_path: str | Path, include_prompt: bool = False) -> dict[str, WetlandChangeDataset]:
    return {
        split: WetlandChangeDataset(manifest_path=manifest_path, split=split, include_prompt=include_prompt)
        for split in ["train", "val", "test"]
    }
