from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wetland_cd.training.data_module import build_datasets_from_config  # noqa: E402
from wetland_cd.training.models import build_model  # noqa: E402


def tensor_to_rgb(tensor: torch.Tensor, dataset: str) -> Image.Image:
    array = tensor.detach().cpu().float().numpy()
    if array.shape[0] >= 3:
        if dataset == "wetland":
            rgb = np.stack([array[2], array[1], array[0]], axis=-1)
        else:
            rgb = np.stack([array[0], array[1], array[2]], axis=-1)
    else:
        rgb = np.repeat(array[0][..., None], 3, axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def mask_to_image(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask > 0] = color
    return Image.fromarray(rgb)


def overlay_mask(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.asarray(overlay).copy()
    overlay_arr[mask > 0] = (*color, int(alpha * 255))
    return Image.alpha_composite(base, Image.fromarray(overlay_arr)).convert("RGB")


def add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), (255, 255, 255))
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(20, 20, 20))
    return canvas


def make_panel(sample: dict, logits: torch.Tensor, dataset: str, threshold: float) -> Image.Image:
    t1 = tensor_to_rgb(sample["t1"], dataset)
    t2 = tensor_to_rgb(sample["t2"], dataset)
    gt = sample["binary_mask"].squeeze(0).detach().cpu().numpy() > 0.5
    pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy() > threshold

    gt_img = mask_to_image(gt, (0, 180, 80))
    pred_img = mask_to_image(pred, (230, 55, 45))
    overlay = overlay_mask(overlay_mask(t2, gt, (0, 180, 80), 0.35), pred, (230, 55, 45), 0.35)

    panels = [
        add_title(t1, "T1"),
        add_title(t2, "T2"),
        add_title(gt_img, "GT change"),
        add_title(pred_img, "Prediction"),
        add_title(overlay, "Overlay: GT green, pred red"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize predictions from a unified training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.run_dir / "config_resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = build_datasets_from_config(config)
    dataset = datasets[args.split]

    sample = dataset[0]
    in_channels = int(sample["t1"].shape[0])
    model_cfg = config["model"]
    model = build_model(
        model_cfg["name"],
        in_channels=int(model_cfg.get("in_channels", in_channels)),
        **dict(model_cfg.get("params", {})),
    )
    checkpoint_path = args.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = args.output_dir or (args.run_dir / "visualizations" / args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = config["data"]["dataset"]

    with torch.no_grad():
        for index in range(min(args.num_samples, len(dataset))):
            sample = dataset[index]
            logits = model(sample["t1"].unsqueeze(0), sample["t2"].unsqueeze(0))[0]
            panel = make_panel(sample, logits, dataset_name, args.threshold)
            sample_id = safe_name(str(sample["sample_id"]))
            output_path = output_dir / f"{index:03d}_{sample_id}.png"
            panel.save(output_path)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
