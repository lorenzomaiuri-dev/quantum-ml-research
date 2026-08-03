"""Generate thesis figures from an experiment 01 comparison aggregate."""

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
BASELINE = "#6f7782"


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


def learning_curves(data: dict, output: Path) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 150, 155, 1520, 790

    histories = {
        variant: data["runs"][variant] for variant in ("classical", "quantum")
    }
    steps = [point["step"] for point in histories["classical"][0]["history"]]
    series: dict[str, list[tuple[float, float]]] = {}
    for variant, runs in histories.items():
        series[variant] = [
            _mean_sd([run["history"][index]["val_loss"] for run in runs])
            for index in range(len(steps))
        ]

    unigram = histories["classical"][0]["baselines"]["unigram_val_loss"]
    y_min = 2.4
    y_max = max(4.5, unigram + 0.15)

    def x(value: float) -> float:
        return left + (value - steps[0]) / (steps[-1] - steps[0]) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for tick in (2.5, 3.0, 3.5, 4.0, 4.5):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text((left - 25, yy), f"{tick:.1f}", fill=TEXT, font=_font(27), anchor="rm")
    for tick in (0, 300, 600, 900, 1200, 1500):
        xx = x(min(tick, steps[-1]))
        draw.line((xx, top, xx, bottom), fill=GRID, width=1)
        draw.text((xx, bottom + 24), str(tick), fill=TEXT, font=_font(27), anchor="ma")

    draw.line((left, y(unigram), right, y(unigram)), fill=BASELINE, width=4)
    draw.text(
        (right - 8, y(unigram) - 12),
        f"Baseline unigramma: {unigram:.3f}",
        fill=BASELINE,
        font=_font(27),
        anchor="rb",
    )

    for variant, color in (("classical", CLASSICAL), ("quantum", QUANTUM)):
        means = [item[0] for item in series[variant]]
        sds = [item[1] for item in series[variant]]
        upper = [(x(step), y(mean + sd)) for step, mean, sd in zip(steps, means, sds)]
        lower = [(x(step), y(mean - sd)) for step, mean, sd in zip(steps, means, sds)]
        draw.polygon(upper + list(reversed(lower)), fill=color + "2e")
        points = [(x(step), y(mean)) for step, mean in zip(steps, means)]
        draw.line(points, fill=color, width=6, joint="curve")

    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    draw.text(
        (width // 2, 42),
        "Esperimento 01: loss di validation durante l'addestramento",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text(
        (width // 2, 925),
        "Iterazione",
        fill=TEXT,
        font=_font(30),
        anchor="ma",
    )
    draw.text((45, (top + bottom) // 2), "Loss", fill=TEXT, font=_font(30), anchor="mm")

    legend_y = 125
    for legend_x, label, color in (
        (1040, "Classica", CLASSICAL),
        (1280, "Quantistica", QUANTUM),
    ):
        draw.line((legend_x, legend_y, legend_x + 65, legend_y), fill=color, width=7)
        draw.text((legend_x + 78, legend_y), label, fill=TEXT, font=_font(27), anchor="lm")
    draw.text(
        (left, 855),
        "Media su 10 seed; banda = +/- 1 deviazione standard.",
        fill=BASELINE,
        font=_font(25),
    )
    _save(image, output)


def paired_effects(data: dict, output: Path) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 220, 140, 1270, 770
    by_variant = {
        variant: {run["seed"]: run for run in data["runs"][variant]}
        for variant in ("classical", "quantum")
    }
    seeds = data["seeds"]
    differences = [
        by_variant["quantum"][seed]["selected_val_perplexity"]
        - by_variant["classical"][seed]["selected_val_perplexity"]
        for seed in seeds
    ]
    stats = data["paired_statistics"]["selected_val_perplexity"]
    mean = stats["differences"]["mean"]
    ci_low, ci_high = stats["mean_difference_ci"]
    x_min, x_max = -0.65, 0.65

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    row_height = (bottom - top) / (len(seeds) + 1)
    for tick in (-0.6, -0.3, 0.0, 0.3, 0.6):
        xx = x(tick)
        draw.line((xx, top, xx, bottom), fill=TEXT if tick == 0 else GRID, width=4 if tick == 0 else 2)
        draw.text((xx, bottom + 24), f"{tick:+.1f}", fill=TEXT, font=_font(27), anchor="ma")

    for index, (seed, difference) in enumerate(zip(seeds, differences)):
        yy = top + row_height * (index + 0.5)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        color = QUANTUM if difference > 0 else CLASSICAL
        draw.ellipse((x(difference) - 10, yy - 10, x(difference) + 10, yy + 10), fill=color)
        draw.text((left - 24, yy), str(seed), fill=TEXT, font=_font(25), anchor="rm")
        draw.text(
            (right + 18, yy),
            f"{difference:+.4f}",
            fill=color,
            font=_font(24, bold=True),
            anchor="lm",
        )

    yy = top + row_height * (len(seeds) + 0.5)
    draw.line((left, yy, right, yy), fill=GRID, width=2)
    draw.line((x(ci_low), yy, x(ci_high), yy), fill="#20242a", width=8)
    draw.line((x(ci_low), yy - 15, x(ci_low), yy + 15), fill="#20242a", width=5)
    draw.line((x(ci_high), yy - 15, x(ci_high), yy + 15), fill="#20242a", width=5)
    draw.ellipse((x(mean) - 13, yy - 13, x(mean) + 13, yy + 13), fill="#20242a")
    draw.text((left - 24, yy), "Media", fill=TEXT, font=_font(25, bold=True), anchor="rm")
    draw.text(
        (right + 18, yy),
        f"{mean:+.4f}",
        fill=TEXT,
        font=_font(24, bold=True),
        anchor="lm",
    )

    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    draw.text(
        (width // 2, 40),
        "Esperimento 01: differenze appaiate di perplexity",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text((left - 24, 105), "Seed", fill=TEXT, font=_font(27, bold=True), anchor="rm")
    draw.text(
        (width // 2, 930),
        "Perplexity quantistica - classica  (valori negativi favoriscono la variante quantistica)",
        fill=TEXT,
        font=_font(27),
        anchor="ma",
    )
    p_value = stats["paired_t_test"]["p_value"]
    draw.text(
        (left, 850),
        f"Media = {mean:+.4f}; IC 95% [{ci_low:+.4f}, {ci_high:+.4f}]; "
        f"test t appaiato p = {p_value:.3f}; n = {len(seeds)}.",
        fill=BASELINE,
        font=_font(25),
    )
    _save(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if data.get("experiment") != "01_quantum_gpt" or data.get("status") != "complete":
        raise ValueError("expected a complete experiment 01 aggregate")
    for run in data["runs"]["classical"] + data["runs"]["quantum"]:
        for field in ("selected_val_loss", "selected_val_perplexity"):
            if not math.isfinite(run[field]):
                raise ValueError(f"non-finite {field} in seed {run['seed']}")

    learning_curves(data, args.output_dir / "experiment_01_learning_curves.png")
    paired_effects(data, args.output_dir / "experiment_01_paired_effects.png")


if __name__ == "__main__":
    main()
