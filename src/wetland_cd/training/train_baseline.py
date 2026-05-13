import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from models import SiameseUNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "runs" / "siamese_unet"


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1 - (numerator + eps) / denominator


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    max_batches: int = 0,
    pos_weight: float = 1.0,
) -> dict[str, float]:
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    model.train(train)
    losses = []
    metrics = []

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        mask = batch["binary_mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(t1, t2)
            loss = bce(logits, mask) + dice_loss_from_logits(logits, mask).mean()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        metrics.append(compute_metrics(logits.detach(), mask.detach()))

    summary = {"loss": sum(losses) / max(len(losses), 1)}
    for key in ["precision", "recall", "f1", "iou"]:
        summary[key] = sum(item[key] for item in metrics) / max(len(metrics), 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese UNet baseline on wetland change patches.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--pos-weight", type=float, default=6.0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = build_datasets(args.manifest)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }

    model = SiameseUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            train=True,
            max_batches=args.limit_train_batches,
            pos_weight=args.pos_weight,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            optimizer,
            device,
            train=False,
            max_batches=args.limit_val_batches,
            pos_weight=args.pos_weight,
        )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.outdir / "best_model.pt",
            )

    test_metrics = run_epoch(
        model,
        loaders["test"],
        optimizer,
        device,
        train=False,
        max_batches=args.limit_test_batches,
        pos_weight=args.pos_weight,
    )
    result = {"history": history, "test": test_metrics}
    (args.outdir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
