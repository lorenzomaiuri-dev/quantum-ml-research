"""Freeze compact experiment 04 evidence and generate thesis figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COSINE = "#2864a0"
RBF = "#d18b2c"
IQP = "#b23a48"
GRID = "#d8dde5"
TEXT = "#20242a"
MUTED = "#6f7782"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/dejavu") / name,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180), optimize=True)


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _compact_separation(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": data["schema_version"],
        "feature_map": data["feature_map"],
        "iqp_repeats": data["iqp_repeats"],
        "kernel_method": data["kernel_method"],
        "seeds": data["seeds"],
        "runs": [
            {
                key: run[key]
                for key in (
                    "seed",
                    "n_class0",
                    "n_class1",
                    "n_qubits",
                    "preprocessing",
                    "classical",
                    "classical_rbf",
                    "quantum",
                    "quantum_kernel_time_seconds",
                    "total_wall_time_seconds",
                )
                if key in run
            }
            for run in data["runs"]
        ],
        "paired_statistics": data["paired_statistics"],
        "paired_statistics_vs_rbf": data["paired_statistics_vs_rbf"],
    }


def _compact_model_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "selected_c",
        "mean_cv_auc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
        "test_macro_f1",
    )
    return {key: metrics[key] for key in fields}


def _compact_validation(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": data["schema_version"],
        "dataset": data["dataset"],
        "feature_map": data["feature_map"],
        "iqp_repeats": data["iqp_repeats"],
        "n_qubits": data["n_qubits"],
        "n_class0": data["n_class0"],
        "n_class1": data["n_class1"],
        "test_split": data["test_split"],
        "svc_c_grid": data["svc_c_grid"],
        "seeds": data["seeds"],
        "runs": [
            {
                "seed": run["seed"],
                "training_indices": run["training_indices"],
                "training_class_counts": run["training_class_counts"],
                "test_class_counts": run["test_class_counts"],
                "pca_solver": run["pca_solver"],
                "pca_explained_variance": run["pca_explained_variance"],
                "rbf_gamma": run["rbf_gamma"],
                "metrics": {
                    model: _compact_model_metrics(metrics)
                    for model, metrics in run["metrics"].items()
                },
            }
            for run in data["runs"]
        ],
        "paired_statistics": data["paired_statistics"],
    }


def _validate_separation(data: dict[str, Any], *, expected_runs: int) -> None:
    if data.get("experiment") != "04_quantum_kernel":
        raise ValueError("expected an experiment 04 separation aggregate")
    if data.get("schema_version") != "1.1" or data.get("feature_map") != "iqp":
        raise ValueError("expected schema 1.1 with the IQP feature map")
    if len(data.get("runs", [])) != expected_runs:
        raise ValueError(f"expected {expected_runs} separation runs")
    if [run["seed"] for run in data["runs"]] != data["seeds"]:
        raise ValueError("unordered or incomplete separation seeds")


def _validate_classifier(data: dict[str, Any]) -> None:
    if data.get("experiment") != "04_quantum_kernel_classifier_validation":
        raise ValueError("expected an experiment 04 classifier aggregate")
    if data.get("schema_version") != "1.1" or data.get("feature_map") != "iqp":
        raise ValueError("expected schema 1.1 with the IQP feature map")
    if len(data.get("runs", [])) != 20 or len(data.get("seeds", [])) != 20:
        raise ValueError("expected 20 classifier-validation runs")
    if data.get("test_split") != "official_test":
        raise ValueError("classifier validation must use the official test split")


def _statevector_error(primary: dict[str, Any], check: dict[str, Any]) -> float:
    primary_run = next(
        run for run in primary["runs"] if run["seed"] == check["runs"][0]["seed"]
    )
    check_run = check["runs"][0]
    errors = []
    for kernel in ("classical", "classical_rbf", "quantum"):
        for metric, value in primary_run[kernel].items():
            if metric in check_run[kernel] and isinstance(value, (int, float)):
                errors.append(abs(value - check_run[kernel][metric]))
    return max(errors, default=0.0)


def build_summary(
    primary_path: Path,
    control_paths: list[Path],
    statevector_path: Path,
    classifier_paths: list[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    primary = _read_json(primary_path)
    controls = [_read_json(path) for path in control_paths]
    statevector = _read_json(statevector_path)
    classifiers = [_read_json(path) for path in classifier_paths]
    _validate_separation(primary, expected_runs=20)
    for control in controls:
        _validate_separation(control, expected_runs=20)
    _validate_separation(statevector, expected_runs=1)
    for classifier in classifiers:
        _validate_classifier(classifier)

    control_by_regime = {
        f"{data['runs'][0]['n_class0']}_{data['runs'][0]['n_class1']}": data
        for data in controls
    }
    classifier_by_regime = {
        f"{data['n_class0']}_{data['n_class1']}": data for data in classifiers
    }
    if set(control_by_regime) != {"50_50", "27_73"}:
        raise ValueError("expected 50/50 and 27/73 separation controls")
    if set(classifier_by_regime) != {"80_20", "50_50", "27_73"}:
        raise ValueError("expected classifier validation in all three regimes")

    all_paths = [
        primary_path,
        *control_paths,
        statevector_path,
        *classifier_paths,
    ]
    summary = {
        "schema_version": "1.0",
        "experiment": "04_quantum_kernel_frozen_evidence",
        "status": "complete",
        "sources": [_source_record(path) for path in all_paths],
        "primary_separation": _compact_separation(primary),
        "separation_controls": {
            regime: _compact_separation(data)
            for regime, data in sorted(control_by_regime.items())
        },
        "pairwise_statevector_check": {
            "seed": statevector["runs"][0]["seed"],
            "max_abs_metric_error": _statevector_error(primary, statevector),
            "statevector_result": _compact_separation(statevector),
        },
        "classifier_validation": {
            regime: _compact_validation(data)
            for regime, data in sorted(classifier_by_regime.items())
        },
    }
    return summary, primary, classifier_by_regime


def fisher_effects(data: dict[str, Any], output: Path) -> None:
    width, height = 1700, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 360, 190, 1360, 750
    comparisons = (
        (
            "IQP - coseno",
            [
                run["quantum"]["fisher"] - run["classical"]["fisher"]
                for run in data["runs"]
            ],
            data["paired_statistics"]["fisher"],
        ),
        (
            "IQP - RBF",
            [
                run["quantum"]["fisher"] - run["classical_rbf"]["fisher"]
                for run in data["runs"]
            ],
            data["paired_statistics_vs_rbf"]["fisher"],
        ),
    )
    bounds = [0.0]
    for _, differences, stats in comparisons:
        bounds.extend(differences)
        bounds.extend(stats["mean_difference_ci"])
    x_min = math.floor((min(bounds) - 0.01) * 20) / 20
    x_max = math.ceil((max(bounds) + 0.01) * 20) / 20

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
            (xx, bottom + 25),
            f"{tick:+.2f}",
            fill=TEXT,
            font=_font(24),
            anchor="ma",
        )
        tick += 0.05

    row_height = (bottom - top) / len(comparisons)
    offsets = [(-65 + index * 130 / 19) for index in range(20)]
    for row, (label, differences, stats) in enumerate(comparisons):
        yy = top + row_height * (row + 0.5)
        mean = stats["differences"]["mean"]
        ci_low, ci_high = stats["mean_difference_ci"]
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        for offset, difference in zip(offsets, differences):
            color = IQP if difference > 0 else COSINE
            draw.ellipse(
                (
                    x(difference) - 7,
                    yy + offset - 7,
                    x(difference) + 7,
                    yy + offset + 7,
                ),
                fill=color + "a8",
            )
        draw.line((x(ci_low), yy, x(ci_high), yy), fill=TEXT, width=9)
        draw.line((x(ci_low), yy - 17, x(ci_low), yy + 17), fill=TEXT, width=5)
        draw.line((x(ci_high), yy - 17, x(ci_high), yy + 17), fill=TEXT, width=5)
        draw.ellipse((x(mean) - 14, yy - 14, x(mean) + 14, yy + 14), fill=TEXT)
        draw.text(
            (left - 28, yy),
            label,
            fill=TEXT,
            font=_font(29, bold=True),
            anchor="rm",
        )
        draw.text(
            (right + 25, yy),
            f"{mean:+.4f}\n[{ci_low:+.4f}, {ci_high:+.4f}]\n"
            f"p={stats['paired_t_test']['p_value']:.4f}",
            fill=TEXT,
            font=_font(23),
            anchor="lm",
            spacing=6,
        )

    draw.text(
        (width // 2, 42),
        "Esperimento 04: effetti appaiati sul criterio di Fisher",
        fill=TEXT,
        font=_font(39, bold=True),
        anchor="ma",
    )
    draw.text(
        (width // 2, 900),
        "Differenza IQP - baseline; valori negativi favoriscono il kernel classico",
        fill=MUTED,
        font=_font(27),
        anchor="ma",
    )
    draw.text(
        (left, 820),
        "Punti = 20 sottocampioni appaiati; punto nero = media; barra = IC 95%.",
        fill=MUTED,
        font=_font(24),
    )
    _save(image, output)


def svc_metrics(data_by_regime: dict[str, dict[str, Any]], output: Path) -> None:
    width, height = 1800, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, right = 145, 1730
    panel_top = (175, 515, 855)
    panel_height = 245
    metrics = (
        ("test_balanced_accuracy", "Balanced accuracy"),
        ("test_auc", "AUC"),
        ("test_macro_f1", "Macro-F1"),
    )
    regimes = ("80_20", "50_50", "27_73")
    models = (
        ("cosine_angle", "Coseno", COSINE),
        ("rbf_median_pca", "RBF", RBF),
        ("iqp", "IQP", IQP),
    )
    y_max = 0.85

    for panel, (metric, label) in enumerate(metrics):
        top = panel_top[panel]
        bottom = top + panel_height

        def y(value: float) -> float:
            return bottom - value / y_max * (bottom - top)

        for index in range(5):
            tick = index * 0.2
            yy = y(tick)
            draw.line((left, yy, right, yy), fill=GRID, width=2)
            draw.text(
                (left - 20, yy),
                f"{tick:.1f}",
                fill=TEXT,
                font=_font(21),
                anchor="rm",
            )
        group_width = (right - left) / len(regimes)
        offsets = (-75, 0, 75)
        for regime_index, regime in enumerate(regimes):
            center = left + group_width * (regime_index + 0.5)
            runs = data_by_regime[regime]["runs"]
            for offset, (model, _, color) in zip(offsets, models):
                values = [run["metrics"][model][metric] for run in runs]
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                xx = center + offset
                draw.rectangle((xx - 30, y(mean), xx + 30, bottom), fill=color)
                draw.line((xx, y(mean - sd), xx, y(mean + sd)), fill=TEXT, width=4)
                draw.line(
                    (xx - 13, y(mean - sd), xx + 13, y(mean - sd)),
                    fill=TEXT,
                    width=4,
                )
                draw.line(
                    (xx - 13, y(mean + sd), xx + 13, y(mean + sd)),
                    fill=TEXT,
                    width=4,
                )
            if panel == len(metrics) - 1:
                draw.text(
                    (center, bottom + 24),
                    regime.replace("_", "/"),
                    fill=TEXT,
                    font=_font(26, bold=True),
                    anchor="ma",
                )
        draw.line((left, top, left, bottom), fill=TEXT, width=3)
        draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
        draw.text(
            (left + 10, top + 10),
            label,
            fill=TEXT,
            font=_font(27, bold=True),
            anchor="la",
        )

    draw.text(
        (width // 2, 38),
        "Esperimento 04: validazione SVC sul test ufficiale",
        fill=TEXT,
        font=_font(40, bold=True),
        anchor="ma",
    )
    legend_x = 1060
    for index, (_, label, color) in enumerate(models):
        xx = legend_x + index * 205
        draw.rectangle((xx, 105, xx + 34, 133), fill=color)
        draw.text((xx + 46, 119), label, fill=TEXT, font=_font(24), anchor="lm")
    draw.text(
        (width // 2, 1260),
        "Regime di campionamento train classe 0/1; media +/- 1 DS su 20 seed",
        fill=MUTED,
        font=_font(25),
        anchor="ma",
    )
    _save(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path, nargs="?")
    parser.add_argument("--separation-controls", type=Path, nargs=2)
    parser.add_argument("--statevector-check", type=Path)
    parser.add_argument("--classifier-validations", type=Path, nargs=3)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        help="Regenerate figures directly from the canonical compact artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.frozen_summary:
        if args.primary or args.summary_output:
            parser.error("--frozen-summary cannot be combined with raw inputs")
        summary = _read_json(args.frozen_summary)
        if summary.get("experiment") != "04_quantum_kernel_frozen_evidence":
            parser.error("--frozen-summary is not an experiment 04 frozen artifact")
        primary = summary["primary_separation"]
        classifiers = summary["classifier_validation"]
    else:
        required = (
            args.primary,
            args.separation_controls,
            args.statevector_check,
            args.classifier_validations,
            args.summary_output,
        )
        if any(value is None for value in required):
            parser.error("raw mode requires all inputs and --summary-output")
        summary, primary, classifiers = build_summary(
            args.primary,
            args.separation_controls,
            args.statevector_check,
            args.classifier_validations,
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"summary: {args.summary_output}")
    fisher_effects(primary, args.output_dir / "experiment_04_fisher_effects.png")
    svc_metrics(classifiers, args.output_dir / "experiment_04_svc_metrics.png")
    print(f"figures: {args.output_dir}")


if __name__ == "__main__":
    main()
