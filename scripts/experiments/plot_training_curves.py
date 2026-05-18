from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def load_history(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "history.csv"
    if csv_path.exists():
        rows = []
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append({key: _to_float(value) for key, value in row.items()})
        return rows

    metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("history", []):
        row = {"epoch": float(item["epoch"]), "lr": float(item.get("lr", 0.0))}
        for split in ["train", "val"]:
            for key, value in item.get(split, {}).items():
                row[f"{split}_{key}"] = float(value)
        rows.append(row)
    return rows


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def draw_chart(
    draw: ImageDraw.ImageDraw,
    rows: list[dict],
    keys: list[str],
    colors: list[tuple[int, int, int]],
    box: tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    pad_left, pad_top, pad_right, pad_bottom = 54, 30, 18, 38
    plot = (x0 + pad_left, y0 + pad_top, x1 - pad_right, y1 - pad_bottom)
    px0, py0, px1, py1 = plot

    draw.rectangle(box, outline=(210, 210, 210), width=1)
    draw.text((x0 + 12, y0 + 8), title, fill=(20, 20, 20))
    draw.line((px0, py1, px1, py1), fill=(70, 70, 70), width=1)
    draw.line((px0, py0, px0, py1), fill=(70, 70, 70), width=1)

    values = [row[key] for row in rows for key in keys if key in row]
    if not values:
        draw.text((px0 + 8, py0 + 8), "No data", fill=(140, 0, 0))
        return
    ymin, ymax = min(values), max(values)
    if abs(ymax - ymin) < 1e-8:
        ymax = ymin + 1.0

    epochs = [row.get("epoch", i + 1) for i, row in enumerate(rows)]
    xmin, xmax = min(epochs), max(epochs)
    if abs(xmax - xmin) < 1e-8:
        xmax = xmin + 1.0

    draw.text((x0 + 8, py0 - 6), f"{ymax:.3f}", fill=(90, 90, 90))
    draw.text((x0 + 8, py1 - 8), f"{ymin:.3f}", fill=(90, 90, 90))
    draw.text((px0, py1 + 10), f"epoch {int(xmin)}", fill=(90, 90, 90))
    draw.text((px1 - 70, py1 + 10), f"epoch {int(max(epochs))}", fill=(90, 90, 90))

    for key, color in zip(keys, colors):
        points = []
        for index, row in enumerate(rows):
            if key not in row:
                continue
            x = px0 + (row.get("epoch", index + 1) - xmin) / (xmax - xmin) * (px1 - px0)
            y = py1 - (row[key] - ymin) / (ymax - ymin) * (py1 - py0)
            points.append((x, y))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        elif len(points) > 1:
            draw.line(points, fill=color, width=2)

    legend_x = px0 + 8
    for key, color in zip(keys, colors):
        draw.rectangle((legend_x, y1 - 24, legend_x + 12, y1 - 12), fill=color)
        draw.text((legend_x + 16, y1 - 27), key, fill=(40, 40, 40))
        legend_x += 130


def plot_training_curves(run_dir: Path, output: Path) -> None:
    rows = load_history(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1200, 520), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw_chart(
        draw,
        rows,
        ["train_loss", "val_loss"],
        [(31, 119, 180), (214, 39, 40)],
        (20, 20, 590, 500),
        "Loss",
    )
    draw_chart(
        draw,
        rows,
        ["train_f1", "val_f1"],
        [(44, 160, 44), (255, 127, 14)],
        (610, 20, 1180, 500),
        "F1",
    )
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a unified run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.run_dir / "training_curves.png")
    plot_training_curves(args.run_dir, output)
    print(f"Saved training curves to: {output}")


if __name__ == "__main__":
    main()
