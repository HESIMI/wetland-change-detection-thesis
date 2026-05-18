from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_name",
    "dataset",
    "model",
    "input_size",
    "epochs",
    "optimizer",
    "lr",
    "scheduler",
    "best_epoch",
    "best_val",
    "test_loss",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_iou",
    "test_accuracy",
    "out_dir",
]


def collect_run(metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = payload.get("test", {})
    scheduler = payload.get("scheduler", {})
    return {
        "run_name": payload.get("run_name", metrics_path.parent.name),
        "dataset": payload.get("dataset", ""),
        "model": payload.get("model", ""),
        "input_size": payload.get("input_size", ""),
        "epochs": payload.get("epochs", ""),
        "optimizer": payload.get("optimizer", ""),
        "lr": payload.get("lr", ""),
        "scheduler": scheduler.get("name", scheduler) if isinstance(scheduler, dict) else scheduler,
        "best_epoch": payload.get("best_epoch", ""),
        "best_val": payload.get("best_val", ""),
        "test_loss": test.get("loss", ""),
        "test_precision": test.get("precision", ""),
        "test_recall": test.get("recall", ""),
        "test_f1": test.get("f1", ""),
        "test_iou": test.get("iou", ""),
        "test_accuracy": test.get("accuracy", ""),
        "out_dir": str(metrics_path.parent),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Result Summary",
        "",
        "| Run | Dataset | Model | Epochs | Best Val F1 | Test F1 | Test IoU | Test Precision | Test Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['dataset']} | {row['model']} | {row['epochs']} | "
            f"{_fmt(row['best_val'])} | {_fmt(row['test_f1'])} | {_fmt(row['test_iou'])} | "
            f"{_fmt(row['test_precision'])} | {_fmt(row['test_recall'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return ""
    if value == "":
        return ""
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize unified training runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/baseline_results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/baseline_results.md"))
    args = parser.parse_args()

    rows = [collect_run(path) for path in sorted(args.runs_root.glob("*/metrics.json"))]
    rows = [row for row in rows if row["run_name"]]
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"Collected {len(rows)} runs")
    print(f"Saved CSV to: {args.output_csv}")
    print(f"Saved Markdown to: {args.output_md}")


if __name__ == "__main__":
    main()
