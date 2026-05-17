from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train import apply_cli_overrides, load_config, resolve_path, train_from_config
except ImportError:
    from train import apply_cli_overrides, load_config, resolve_path, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backward-compatible Siamese U-Net baseline entrypoint. Prefer train.py for new experiments."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training/wetland_siamese_unet.json"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    config = apply_cli_overrides(config, args)
    if args.pos_weight is not None:
        config.setdefault("training", {})["pos_weight"] = args.pos_weight
    config.setdefault("results", {})
    train_from_config(config)


if __name__ == "__main__":
    main()
