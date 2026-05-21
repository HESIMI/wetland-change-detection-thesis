from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .data_module import build_loaders_from_config, repo_root, resolve_path
    from .losses import BinaryChangeLoss
    from .metrics import BinaryChangeMetrics
    from .models import build_model
except ImportError:
    from data_module import build_loaders_from_config, repo_root, resolve_path
    from losses import BinaryChangeLoss
    from metrics import BinaryChangeMetrics
    from models import build_model


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if getattr(args, "dataset", None) is not None:
        updates.setdefault("data", {})["dataset"] = args.dataset
    if getattr(args, "manifest", None) is not None:
        updates.setdefault("data", {})["manifest"] = str(args.manifest)
    if getattr(args, "root", None) is not None:
        updates.setdefault("data", {})["root"] = str(args.root)
    if getattr(args, "image_size", None) is not None:
        updates.setdefault("data", {})["image_size"] = args.image_size
    if getattr(args, "model", None) is not None:
        updates.setdefault("model", {})["name"] = args.model
    if getattr(args, "epochs", None) is not None:
        updates.setdefault("training", {})["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        updates.setdefault("training", {})["batch_size"] = args.batch_size
    if getattr(args, "lr", None) is not None:
        updates.setdefault("training", {})["lr"] = args.lr
    if getattr(args, "num_workers", None) is not None:
        updates.setdefault("training", {})["num_workers"] = args.num_workers
    if getattr(args, "outdir", None) is not None:
        updates.setdefault("results", {})["out_dir"] = str(args.outdir)
    if getattr(args, "limit_train_batches", None) is not None:
        updates.setdefault("training", {}).setdefault("limit_batches", {})["train"] = args.limit_train_batches
    if getattr(args, "limit_val_batches", None) is not None:
        updates.setdefault("training", {}).setdefault("limit_batches", {})["val"] = args.limit_val_batches
    if getattr(args, "limit_test_batches", None) is not None:
        updates.setdefault("training", {}).setdefault("limit_batches", {})["test"] = args.limit_test_batches
    return deep_update(config, updates)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(config: dict[str, Any]) -> torch.device:
    requested = config["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    train_cfg = config["training"]
    scheduler_cfg = train_cfg.get("scheduler", {"name": "none"})
    name = scheduler_cfg.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    threshold: float,
    max_batches: int = 0,
) -> dict[str, float]:
    model.train(train)
    metrics = BinaryChangeMetrics(threshold=threshold)
    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        targets = batch["binary_mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(t1, t2)
            loss = criterion(logits, targets)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(t1.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        metrics.update(logits.detach(), targets.detach())

    summary = {"loss": total_loss / max(total_samples, 1)}
    summary.update(metrics.compute())
    return summary


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    rows = []
    for item in history:
        row: dict[str, Any] = {"epoch": item["epoch"], "lr": item["lr"]}
        row.update(flatten_metrics("train", item["train"]))
        row.update(flatten_metrics("val", item["val"]))
        rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    return value > best


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    device = select_device(config)
    datasets, loaders = build_loaders_from_config(config)

    sample = datasets["train"][0]
    inferred_channels = int(sample["t1"].shape[0])
    model_cfg = config["model"]
    model_params = dict(model_cfg.get("params", {}))
    in_channels = int(model_cfg.get("in_channels", inferred_channels))
    model = build_model(model_cfg["name"], in_channels=in_channels, **model_params).to(device)

    train_cfg = config["training"]
    criterion = BinaryChangeLoss(
        pos_weight=float(train_cfg.get("pos_weight", 1.0)),
        dice_weight=float(train_cfg.get("dice_weight", 1.0)),
    )
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("experiment_name") or f"{config['data']['dataset']}_{model_cfg['name']}_{timestamp}"
    out_dir = resolve_path(config.get("results", {}).get("out_dir", f"runs/{run_name}"))
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deepcopy(config)
    resolved_config["run_name"] = run_name
    resolved_config["model"]["in_channels"] = in_channels
    resolved_config["results"]["out_dir"] = str(out_dir)
    (out_dir / "config_resolved.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monitor = train_cfg.get("monitor", "f1")
    monitor_mode = train_cfg.get("monitor_mode", "max")
    threshold = float(train_cfg.get("threshold", 0.5))
    limits = train_cfg.get("limit_batches", {})
    best_value: float | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    history_jsonl = out_dir / "history.jsonl"
    if history_jsonl.exists():
        history_jsonl.unlink()

    print(f"run_name={run_name}")
    print(f"device={device}")
    print(f"dataset={config['data']['dataset']} model={model_cfg['name']} in_channels={in_channels}")
    print(f"results={out_dir}")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            train=True,
            threshold=threshold,
            max_batches=int(limits.get("train", 0)),
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            None,
            device,
            train=False,
            threshold=threshold,
            max_batches=int(limits.get("val", 0)),
        )
        row = {
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        append_jsonl(history_jsonl, row)
        print(json.dumps(row, ensure_ascii=False))

        monitor_value = float(val_metrics[monitor])
        if is_better(monitor_value, best_value, monitor_mode):
            best_value = monitor_value
            best_epoch = epoch
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                resolved_config,
                {"train": train_metrics, "val": val_metrics},
            )

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            resolved_config,
            {"train": train_metrics, "val": val_metrics},
        )
        if scheduler is not None:
            scheduler.step()

    best_checkpoint = checkpoints_dir / "best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        None,
        device,
        train=False,
        threshold=threshold,
        max_batches=int(limits.get("test", 0)),
    )
    write_history_csv(out_dir / "history.csv", history)

    result = {
        "run_name": run_name,
        "dataset": config["data"]["dataset"],
        "model": model_cfg["name"],
        "input_size": config["data"].get("image_size"),
        "epochs": int(train_cfg["epochs"]),
        "optimizer": train_cfg.get("optimizer", "adamw"),
        "lr": float(train_cfg["lr"]),
        "scheduler": train_cfg.get("scheduler", {"name": "none"}),
        "monitor": monitor,
        "best_epoch": best_epoch,
        "best_val": best_value,
        "history": history,
        "test": test_metrics,
        "paths": {
            "out_dir": str(out_dir),
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(checkpoints_dir / "last.pt"),
            "history_csv": str(out_dir / "history.csv"),
            "history_jsonl": str(history_jsonl),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "best_epoch": best_epoch}, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified binary change detection training entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/training/wetland_siamese_unet.json"))
    parser.add_argument(
        "--dataset",
        choices=["wetland", "second", "hrscd", "levir-cd", "whu-cd", "sysu-cd"],
        default=None,
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Wetland dataset manifest override.")
    parser.add_argument("--root", type=Path, default=None, help="Public dataset root override.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    config = apply_cli_overrides(load_config(config_path), args)
    config.setdefault("results", {})
    train_from_config(config)


if __name__ == "__main__":
    main()
