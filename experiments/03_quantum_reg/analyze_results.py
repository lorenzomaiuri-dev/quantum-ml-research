"""Generate thesis figures from a complete experiment 03 campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VANILLA = "#6f7782"
BOUNDED = "#2864a0"
QUANTUM = "#b23a48"
GRID = "#d8dde5"
TEXT = "#20242a"
MUTED = "#6f7782"
DATASET_LABELS = {
    "pathmnist": "PathMNIST",
    "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180), optimize=True)


def gap_by_dataset(data: dict, output: Path) -> None:
    width, height = 1700, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 170, 175, 1610, 820
    variants = (
        ("vanilla", "Vanilla", VANILLA),
        ("bounded_mlp", "MLP bounded", BOUNDED),
        ("quantum_reg", "Quantistica", QUANTUM),
    )
    datasets = data["datasets"]
    values = {
        (dataset, variant): [
            run["generalization_gap"] for run in data["runs"][dataset][variant]
        ]
        for dataset in datasets
        for variant, _, _ in variants
    }
    upper = max(
        statistics.mean(series) + statistics.stdev(series)
        for series in values.values()
    )
    y_max = max(0.7, math.ceil((upper + 0.03) * 10) / 10)

    def y(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    for index in range(int(round(y_max / 0.1)) + 1):
        tick = index * 0.1
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text(
            (left - 25, yy), f"{tick:.1f}", fill=TEXT, font=_font(26), anchor="rm"
        )

    group_width = (right - left) / len(datasets)
    bar_width = 105
    offsets = (-125, 0, 125)
    for dataset_index, dataset in enumerate(datasets):
        center = left + group_width * (dataset_index + 0.5)
        for offset, (variant, _, color) in zip(offsets, variants):
            series = values[(dataset, variant)]
            mean = statistics.mean(series)
            sd = statistics.stdev(series)
            xx = center + offset
            draw.rectangle(
                (xx - bar_width / 2, y(mean), xx + bar_width / 2, bottom), fill=color
            )
            draw.line((xx, y(mean - sd), xx, y(mean + sd)), fill=TEXT, width=5)
            draw.line(
                (xx - 18, y(mean - sd), xx + 18, y(mean - sd)), fill=TEXT, width=5
            )
            draw.line(
                (xx - 18, y(mean + sd), xx + 18, y(mean + sd)), fill=TEXT, width=5
            )
            draw.text(
                (xx, y(mean + sd) - 12),
                f"{mean:.3f}",
                fill=TEXT,
                font=_font(23, bold=True),
                anchor="mb",
            )
        draw.text(
            (center, bottom + 28),
            DATASET_LABELS[dataset],
            fill=TEXT,
            font=_font(29, bold=True),
            anchor="ma",
        )

    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    draw.text(
        (width // 2, 42),
        "Esperimento 03: gap di generalizzazione al checkpoint selezionato",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text(
        (width // 2, 940),
        "Gap = accuracy train - accuracy test; valori inferiori indicano "
        "migliore generalizzazione",
        fill=MUTED,
        font=_font(26),
        anchor="ma",
    )
    draw.text(
        (65, (top + bottom) // 2),
        "Gap",
        fill=TEXT,
        font=_font(31),
        anchor="mm",
    )
    legend_x = 875
    for index, (_, label, color) in enumerate(variants):
        xx = legend_x + index * 245
        draw.rectangle((xx, 116, xx + 35, 145), fill=color)
        draw.text((xx + 48, 130), label, fill=TEXT, font=_font(25), anchor="lm")
    draw.text(
        (left, 875),
        f"Media su {len(data['seeds'])} seed; barre di errore = +/- 1 "
        "deviazione standard.",
        fill=MUTED,
        font=_font(24),
    )
    _save(image, output)


def paired_gap_effects(data: dict, output: Path) -> None:
    width, height = 1700, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 315, 185, 1370, 790
    effects = {}
    bounds = [0.0]
    for dataset in data["datasets"]:
        bounded = {
            run["seed"]: run for run in data["runs"][dataset]["bounded_mlp"]
        }
        quantum = {
            run["seed"]: run for run in data["runs"][dataset]["quantum_reg"]
        }
        differences = [
            quantum[seed]["generalization_gap"]
            - bounded[seed]["generalization_gap"]
            for seed in data["seeds"]
        ]
        stats = data["paired_statistics"][dataset]["comparisons"][
            "quantum_reg_vs_bounded_mlp"
        ]["generalization_gap"]
        effects[dataset] = differences, stats
        bounds.extend(differences)
        bounds.extend(stats["mean_difference_ci"])
    x_min = math.floor((min(bounds) - 0.03) * 10) / 10
    x_max = math.ceil((max(bounds) + 0.03) * 10) / 10

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    tick = x_min
    while tick <= x_max + 1e-9:
        xx = x(tick)
        draw.line(
            (xx, top, xx, bottom),
            fill=TEXT if abs(tick) < 1e-9 else GRID,
            width=4 if abs(tick) < 1e-9 else 2,
        )
        draw.text(
            (xx, bottom + 24), f"{tick:+.1f}", fill=TEXT, font=_font(25), anchor="ma"
        )
        tick += 0.1

    row_height = (bottom - top) / len(data["datasets"])
    jitter = (-38, -19, 0, 19, 38)
    for row, dataset in enumerate(data["datasets"]):
        yy = top + row_height * (row + 0.5)
        differences, stats = effects[dataset]
        mean = stats["differences"]["mean"]
        ci_low, ci_high = stats["mean_difference_ci"]
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        for offset, difference in zip(jitter, differences):
            color = QUANTUM if difference > 0 else BOUNDED
            draw.ellipse(
                (
                    x(difference) - 9,
                    yy + offset - 9,
                    x(difference) + 9,
                    yy + offset + 9,
                ),
                fill=color + "b8",
            )
        draw.line((x(ci_low), yy, x(ci_high), yy), fill=TEXT, width=9)
        draw.line((x(ci_low), yy - 17, x(ci_low), yy + 17), fill=TEXT, width=5)
        draw.line((x(ci_high), yy - 17, x(ci_high), yy + 17), fill=TEXT, width=5)
        draw.ellipse((x(mean) - 14, yy - 14, x(mean) + 14, yy + 14), fill=TEXT)
        draw.text(
            (left - 28, yy),
            DATASET_LABELS[dataset],
            fill=TEXT,
            font=_font(28, bold=True),
            anchor="rm",
        )
        p_value = stats["paired_t_test"]["p_value"]
        draw.text(
            (right + 24, yy),
            f"{mean:+.3f}\n[{ci_low:+.3f}, {ci_high:+.3f}]\np={p_value:.3f}",
            fill=TEXT,
            font=_font(23),
            anchor="lm",
            spacing=6,
        )

    draw.text(
        (width // 2, 42),
        "Esperimento 03: effetto appaiato sul gap di generalizzazione",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text(
        (width // 2, 930),
        "Gap quantistico - gap MLP bounded  (valori positivi sfavoriscono "
        "la variante quantistica)",
        fill=TEXT,
        font=_font(26),
        anchor="ma",
    )
    draw.text(
        (left, 855),
        "Punti = differenze per seed; punto nero = media; barra = IC t al 95% (n=5).",
        fill=MUTED,
        font=_font(24),
    )
    _save(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if data.get("experiment") != "03_quantum_reg" or data.get("status") != "complete":
        raise ValueError("expected a complete experiment 03 aggregate")
    for dataset in data["datasets"]:
        for variant in ("vanilla", "bounded_mlp", "quantum_reg"):
            runs = data["runs"][dataset][variant]
            if [run["seed"] for run in runs] != data["seeds"]:
                raise ValueError(f"incomplete {dataset}/{variant} run list")

    gap_by_dataset(data, args.output_dir / "experiment_03_gap_by_dataset.png")
    paired_gap_effects(data, args.output_dir / "experiment_03_paired_gap_effects.png")


if __name__ == "__main__":
    main()
