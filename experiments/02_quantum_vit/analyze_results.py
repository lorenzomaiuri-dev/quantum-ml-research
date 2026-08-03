"""Generate thesis figures from a complete experiment 02 aggregate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSICAL = "#2864a0"
QUANTUM = "#b23a48"
GRID = "#d8dde5"
TEXT = "#20242a"
MUTED = "#6f7782"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180), optimize=True)


def _draw_vertical_label(
    image: Image.Image, text: str, *, x: int, center_y: int, size: int
) -> None:
    label = Image.new("RGBA", (300, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (150, 35), text, fill=TEXT, font=_font(size), anchor="mm"
    )
    label = label.rotate(90, expand=True)
    image.paste(label, (x, center_y - label.height // 2), label)


def learning_curves(data: dict, output: Path) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 150, 155, 1520, 790

    runs = data["runs"]
    epochs = [point["epoch"] for point in runs["classical"][0]["history"]]
    series: dict[str, list[tuple[float, float]]] = {}
    for variant in ("classical", "quantum"):
        series[variant] = [
            _mean_sd([run["history"][index]["val_acc"] for run in runs[variant]])
            for index in range(len(epochs))
        ]

    extrema = [
        mean + sign * sd
        for variant in series.values()
        for mean, sd in variant
        for sign in (-1, 1)
    ]
    y_min = max(0.0, math.floor((min(extrema) - 0.03) * 20) / 20)
    y_max = min(1.0, math.ceil((max(extrema) + 0.03) * 20) / 20)

    def x(value: float) -> float:
        return left + (value - epochs[0]) / (epochs[-1] - epochs[0]) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    tick_count = int(round((y_max - y_min) / 0.05))
    for index in range(tick_count + 1):
        tick = y_min + index * 0.05
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text(
            (left - 25, yy),
            f"{tick:.2f}",
            fill=TEXT,
            font=_font(25),
            anchor="rm",
        )
    for tick in range(10, epochs[-1] + 1, 10):
        xx = x(tick)
        draw.line((xx, top, xx, bottom), fill=GRID, width=1)
        draw.text((xx, bottom + 24), str(tick), fill=TEXT, font=_font(27), anchor="ma")

    for variant, color in (("classical", CLASSICAL), ("quantum", QUANTUM)):
        means = [item[0] for item in series[variant]]
        sds = [item[1] for item in series[variant]]
        upper = [
            (x(epoch), y(mean + sd))
            for epoch, mean, sd in zip(epochs, means, sds)
        ]
        lower = [
            (x(epoch), y(mean - sd))
            for epoch, mean, sd in zip(epochs, means, sds)
        ]
        draw.polygon(upper + list(reversed(lower)), fill=color + "2e")
        draw.line(
            [(x(epoch), y(mean)) for epoch, mean in zip(epochs, means)],
            fill=color,
            width=6,
            joint="curve",
        )

    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    draw.text(
        (width // 2, 42),
        "Esperimento 02: accuracy di validation",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text((width // 2, 925), "Epoca", fill=TEXT, font=_font(30), anchor="ma")
    _draw_vertical_label(
        image,
        "Accuracy",
        x=15,
        center_y=(top + bottom) // 2,
        size=30,
    )
    for legend_x, label, color in (
        (1040, "Classica", CLASSICAL),
        (1280, "Quantistica", QUANTUM),
    ):
        draw.line((legend_x, 125, legend_x + 65, 125), fill=color, width=7)
        draw.text((legend_x + 78, 125), label, fill=TEXT, font=_font(27), anchor="lm")
    draw.text(
        (left, 855),
        f"Media su {len(data['seeds'])} seed; banda = +/- 1 deviazione standard.",
        fill=MUTED,
        font=_font(25),
    )
    _save(image, output)


def paired_effects(data: dict, output: Path) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 260, 175, 1360, 765
    metrics = (
        ("test_accuracy", "Test accuracy"),
        ("test_auc", "Macro-AUC"),
        ("test_f1", "Macro-F1"),
    )
    by_variant = {
        variant: {run["seed"]: run for run in data["runs"][variant]}
        for variant in ("classical", "quantum")
    }
    effects = {}
    bounds = [0.0]
    for metric, _ in metrics:
        differences = [
            by_variant["quantum"][seed][metric]
            - by_variant["classical"][seed][metric]
            for seed in data["seeds"]
        ]
        stats = data["paired_statistics"][metric]
        ci = stats["mean_difference_ci"]
        effects[metric] = (differences, stats, ci)
        bounds.extend(differences)
        bounds.extend(ci)
    limit = max(0.02, math.ceil(max(abs(value) for value in bounds) * 20) / 20)
    x_min, x_max = -limit, limit

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    for index in range(-4, 5):
        tick = limit * index / 4
        xx = x(tick)
        draw.line(
            (xx, top, xx, bottom),
            fill=TEXT if index == 0 else GRID,
            width=4 if index == 0 else 2,
        )
        draw.text(
            (xx, bottom + 24),
            f"{tick:+.2f}",
            fill=TEXT,
            font=_font(25),
            anchor="ma",
        )

    row_height = (bottom - top) / len(metrics)
    jitter = (-34, -17, 0, 17, 34)
    for row, (metric, label) in enumerate(metrics):
        yy = top + row_height * (row + 0.5)
        differences, stats, (ci_low, ci_high) = effects[metric]
        mean = stats["differences"]["mean"]
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        for offset, difference in zip(jitter, differences):
            color = QUANTUM if difference > 0 else CLASSICAL
            draw.ellipse(
                (
                    x(difference) - 8,
                    yy + offset - 8,
                    x(difference) + 8,
                    yy + offset + 8,
                ),
                fill=color + "b8",
            )
        draw.line((x(ci_low), yy, x(ci_high), yy), fill=TEXT, width=9)
        draw.line((x(ci_low), yy - 17, x(ci_low), yy + 17), fill=TEXT, width=5)
        draw.line((x(ci_high), yy - 17, x(ci_high), yy + 17), fill=TEXT, width=5)
        draw.ellipse((x(mean) - 14, yy - 14, x(mean) + 14, yy + 14), fill=TEXT)
        draw.text(
            (left - 25, yy),
            label,
            fill=TEXT,
            font=_font(27, bold=True),
            anchor="rm",
        )
        t_test = stats["paired_t_test"]
        p_text = f"p={t_test['p_value']:.3f}" if t_test else "p=n.d."
        draw.text(
            (right + 20, yy),
            f"{mean:+.4f}\n[{ci_low:+.4f}, {ci_high:+.4f}]\n{p_text}",
            fill=TEXT,
            font=_font(22),
            anchor="lm",
            spacing=6,
        )

    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    draw.text(
        (width // 2, 42),
        "Esperimento 02: effetti appaiati sulle metriche di test",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text(
        (width // 2, 920),
        "Quantistica - classica  (valori positivi favoriscono la variante quantistica)",
        fill=TEXT,
        font=_font(27),
        anchor="ma",
    )
    draw.text(
        (left, 835),
        "Punti trasparenti = differenze per seed; punto nero = media; "
        "barra = IC al 95%.",
        fill=MUTED,
        font=_font(25),
    )
    _save(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if data.get("experiment") != "02_quantum_vit" or data.get("status") != "complete":
        raise ValueError("expected a complete experiment 02 aggregate")
    if len(data.get("seeds", [])) < 2:
        raise ValueError("at least two paired seeds are required")
    for variant in ("classical", "quantum"):
        if len(data["runs"][variant]) != len(data["seeds"]):
            raise ValueError(f"incomplete {variant} run list")
        for run in data["runs"][variant]:
            for field in ("test_accuracy", "test_auc", "test_f1"):
                if not math.isfinite(run[field]):
                    raise ValueError(f"non-finite {field} in seed {run['seed']}")

    learning_curves(data, args.output_dir / "experiment_02_learning_curves.png")
    paired_effects(data, args.output_dir / "experiment_02_paired_effects.png")


if __name__ == "__main__":
    main()
