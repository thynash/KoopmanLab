#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare Darcy experiment results:

    1. KNO Baseline
    2. Original/Data-enhanced VINO
    3. Full PEDVINO

Expected directory structure:

experiments/darcy/
│
├── results/
│   ├── baseline/
│   │   ├── history.json
│   │   └── metrics.json
│   │
│   ├── vino/
│   │   ├── history.json
│   │   └── metrics.json
│   │
│   └── pedvino/
│       ├── history.json
│       └── metrics.json
│
└── compare_results.py

Generated comparison outputs:

experiments/darcy/results/comparison/
├── validation_relative_l2.png
├── test_relative_l2.png
├── pedvino_loss_components.png
├── final_test_relative_l2.png
├── final_metrics_comparison.png
└── comparison_summary.json
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

RESULTS_DIR = THIS_DIR / "results"

BASELINE_DIR = RESULTS_DIR / "baseline"
VINO_DIR = RESULTS_DIR / "vino"
PEDVINO_DIR = RESULTS_DIR / "pedvino"

COMPARISON_DIR = RESULTS_DIR / "comparison"

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path):
    """
    Load a JSON file safely.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path}\n"
        )

    with open(path, "r") as file:
        return json.load(file)


def get_history(
    experiment_dir,
):
    return load_json(
        experiment_dir / "history.json"
    )


def get_metrics(
    experiment_dir,
):
    return load_json(
        experiment_dir / "metrics.json"
    )


# ============================================================
# HISTORY EXTRACTION
# ============================================================

def get_epochs(history):
    """
    Return epochs robustly.

    If explicit epoch numbers exist, use them.
    Otherwise infer them from the first list field.
    """

    if "epoch" in history:
        return np.asarray(
            history["epoch"],
            dtype=float,
        )

    for value in history.values():

        if isinstance(value, list):

            return np.arange(
                1,
                len(value) + 1,
            )

    return np.array([])


def get_history_value(
    history,
    possible_keys,
):
    """
    Return the first matching history key.

    This allows compatibility between the existing
    baseline/PEDVINO histories and the new VINO history.
    """

    for key in possible_keys:

        if key in history:

            return np.asarray(
                history[key],
                dtype=float,
            )

    return None


def align_epochs_and_values(
    history,
    values,
):
    """
    Make epoch/value arrays compatible in case a history
    file has no explicit epoch field or lengths differ.
    """

    if values is None:
        return None, None

    epochs = get_epochs(history)

    if len(epochs) == 0:

        epochs = np.arange(
            1,
            len(values) + 1,
        )

    length = min(
        len(epochs),
        len(values),
    )

    return (
        epochs[:length],
        values[:length],
    )


# ============================================================
# PLOT 1
# VALIDATION RELATIVE L2
# ============================================================

def plot_validation_l2(
    baseline,
    vino,
    pedvino,
    save_path,
):
    """
    Compare validation Relative L2 curves.
    """

    plt.figure(
        figsize=(10, 6)
    )

    experiments = [
        (
            "KNO",
            baseline,
            [
                "validation_relative_l2",
                "val_relative_l2",
            ],
        ),
        (
            "VINO",
            vino,
            [
                "validation_relative_l2",
                "val_relative_l2",
            ],
        ),
        (
            "PEDVINO",
            pedvino,
            [
                "validation_relative_l2",
                "val_relative_l2",
            ],
        ),
    ]

    plotted = False

    for (
        label,
        history,
        keys,
    ) in experiments:

        values = get_history_value(
            history,
            keys,
        )

        epochs, values = align_epochs_and_values(
            history,
            values,
        )

        if values is None:
            print(
                f"WARNING: Validation Relative L2 "
                f"not found for {label}."
            )
            continue

        plt.plot(
            epochs,
            values,
            label=label,
            linewidth=2,
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "WARNING: No validation curves available."
        )

        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation Relative L2 Error")

    plt.title(
        "Darcy: Validation Relative L2 Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# PLOT 2
# TEST RELATIVE L2
# ============================================================

def plot_test_l2(
    baseline,
    vino,
    pedvino,
    save_path,
):
    """
    Compare test Relative L2 curves.

    The VINO script may only evaluate the test set
    after training. In that case it will not have a
    per-epoch test curve, and the method is skipped here.
    Final test comparison is still shown separately.
    """

    plt.figure(
        figsize=(10, 6)
    )

    experiments = [
        (
            "KNO",
            baseline,
            [
                "test_relative_l2",
            ],
        ),
        (
            "VINO",
            vino,
            [
                "test_relative_l2",
            ],
        ),
        (
            "PEDVINO",
            pedvino,
            [
                "test_relative_l2",
            ],
        ),
    ]

    plotted = False

    for (
        label,
        history,
        keys,
    ) in experiments:

        values = get_history_value(
            history,
            keys,
        )

        epochs, values = align_epochs_and_values(
            history,
            values,
        )

        if values is None:
            print(
                f"INFO: Per-epoch test Relative L2 "
                f"not available for {label}; "
                f"skipping its test curve."
            )
            continue

        plt.plot(
            epochs,
            values,
            label=label,
            linewidth=2,
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "WARNING: No per-epoch test curves available."
        )

        return

    plt.xlabel("Epoch")
    plt.ylabel("Test Relative L2 Error")

    plt.title(
        "Darcy: Test Relative L2 Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# PLOT 3
# PEDVINO LOSS COMPONENTS
# ============================================================

def plot_pedvino_loss_components(
    pedvino,
    save_path,
):
    """
    Plot PEDVINO's individual loss components separately.

    These losses are PEDVINO-specific and are intentionally
    not compared numerically against VINO or KNO losses.
    """

    epochs = get_epochs(
        pedvino
    )

    components = [
        (
            "Prediction Loss",
            [
                "prediction_loss",
                "train_prediction_loss",
            ],
        ),
        (
            "Reconstruction Loss",
            [
                "reconstruction_loss",
                "train_reconstruction_loss",
            ],
        ),
        (
            "Energy Loss",
            [
                "energy_loss",
                "train_energy_loss",
            ],
        ),
        (
            "Gradient Loss",
            [
                "gradient_loss",
                "train_gradient_loss",
            ],
        ),
        (
            "Boundary Loss",
            [
                "boundary_loss",
                "train_boundary_loss",
            ],
        ),
    ]

    plt.figure(
        figsize=(11, 7)
    )

    plotted = False

    for (
        label,
        keys,
    ) in components:

        values = get_history_value(
            pedvino,
            keys,
        )

        if values is None:
            continue

        if len(epochs) == 0:
            current_epochs = np.arange(
                1,
                len(values) + 1,
            )
        else:
            length = min(
                len(epochs),
                len(values),
            )

            current_epochs = epochs[:length]
            values = values[:length]

        if np.isfinite(values).any():

            plt.plot(
                current_epochs,
                values,
                label=label,
                linewidth=2,
            )

            plotted = True

    if not plotted:

        plt.close()

        print(
            "WARNING: No PEDVINO individual loss "
            "components were found."
        )

        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.title(
        "Darcy: PEDVINO Individual Loss Components"
    )

    plt.yscale(
        "symlog",
        linthresh=1e-8,
    )

    plt.legend(
        loc="best"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# METRIC HELPER
# ============================================================

def metric_value(
    metrics,
    possible_keys,
    default=np.nan,
):
    """
    Retrieve a metric using multiple possible names.
    """

    for key in possible_keys:

        if key in metrics:

            try:
                return float(
                    metrics[key]
                )
            except (
                TypeError,
                ValueError,
            ):
                return default

    return default


# ============================================================
# PLOT 4
# FINAL TEST RELATIVE L2
# ============================================================

def plot_final_test_l2(
    baseline_metrics,
    vino_metrics,
    pedvino_metrics,
    save_path,
):
    """
    Compare final test Relative L2.
    """

    models = [
        "KNO",
        "VINO",
        "PEDVINO",
    ]

    values = [
        metric_value(
            baseline_metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),
        metric_value(
            vino_metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),
        metric_value(
            pedvino_metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),
    ]

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        models,
        values,
    )

    plt.ylabel(
        "Final Test Relative L2 Error"
    )

    plt.title(
        "Darcy: Final Test Relative L2 Comparison"
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for (
        bar,
        value,
    ) in zip(
        bars,
        values,
    ):

        if np.isfinite(value):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.4e}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# PLOT 5
# FINAL METRICS
# ============================================================

def plot_final_metrics(
    baseline_metrics,
    vino_metrics,
    pedvino_metrics,
    save_path,
):
    """
    Compare final:
        - Test Relative L2
        - Test MSE
    """

    models = [
        "KNO",
        "VINO",
        "PEDVINO",
    ]

    relative_l2 = np.array(
        [
            metric_value(
                baseline_metrics,
                [
                    "test_relative_l2",
                    "final_test_relative_l2",
                ],
            ),
            metric_value(
                vino_metrics,
                [
                    "test_relative_l2",
                    "final_test_relative_l2",
                ],
            ),
            metric_value(
                pedvino_metrics,
                [
                    "test_relative_l2",
                    "final_test_relative_l2",
                ],
            ),
        ]
    )

    mse = np.array(
        [
            metric_value(
                baseline_metrics,
                [
                    "test_mse",
                    "final_test_mse",
                    "mse",
                ],
            ),
            metric_value(
                vino_metrics,
                [
                    "test_mse",
                    "final_test_mse",
                    "mse",
                ],
            ),
            metric_value(
                pedvino_metrics,
                [
                    "test_mse",
                    "final_test_mse",
                    "mse",
                ],
            ),
        ]
    )

    x = np.arange(
        len(models)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        relative_l2,
        width,
        label="Test Relative L2",
    )

    plt.bar(
        x + width / 2,
        mse,
        width,
        label="Test MSE",
    )

    plt.xticks(
        x,
        models,
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.title(
        "Darcy: Final Metric Comparison"
    )

    plt.yscale(
        "log"
    )

    plt.legend()

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# PRINT RESULTS TABLE
# ============================================================

def print_summary_table(
    baseline_metrics,
    vino_metrics,
    pedvino_metrics,
):
    """
    Print a concise experiment summary.
    """

    experiments = [
        (
            "KNO",
            baseline_metrics,
        ),
        (
            "VINO",
            vino_metrics,
        ),
        (
            "PEDVINO",
            pedvino_metrics,
        ),
    ]

    print("\n" + "=" * 85)
    print("DARCY EXPERIMENT COMPARISON")
    print("=" * 85)

    print(
        f"{'Method':<15}"
        f"{'Test Rel L2':>20}"
        f"{'Test MSE':>20}"
        f"{'Parameters':>20}"
    )

    print("-" * 85)

    rows = []

    for (
        name,
        metrics,
    ) in experiments:

        test_l2 = metric_value(
            metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        )

        test_mse = metric_value(
            metrics,
            [
                "test_mse",
                "final_test_mse",
                "mse",
            ],
        )

        parameters = metric_value(
            metrics,
            [
                "parameter_count",
                "trainable_parameters",
                "num_parameters",
            ],
        )

        rows.append(
            {
                "method": name,
                "test_relative_l2": test_l2,
                "test_mse": test_mse,
                "parameter_count": parameters,
            }
        )

        parameter_text = (
            f"{int(parameters):,}"
            if np.isfinite(parameters)
            else "N/A"
        )

        print(
            f"{name:<15}"
            f"{test_l2:>20.6e}"
            f"{test_mse:>20.6e}"
            f"{parameter_text:>20}"
        )

    print("=" * 85)

    return rows


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    rows,
    baseline_metrics,
    vino_metrics,
    pedvino_metrics,
):
    """
    Save all final comparable metrics.
    """

    best_method = min(
        rows,
        key=lambda row: (
            row["test_relative_l2"]
            if np.isfinite(
                row["test_relative_l2"]
            )
            else float("inf")
        ),
    )

    summary = {
        "experiment": "darcy_comparison",

        "methods": rows,

        "best_method_by_test_relative_l2":
            best_method["method"],

        "best_test_relative_l2":
            best_method["test_relative_l2"],

        "raw_metrics": {
            "baseline":
                baseline_metrics,

            "vino":
                vino_metrics,

            "pedvino":
                pedvino_metrics,
        },
    }

    summary_path = (
        COMPARISON_DIR
        / "comparison_summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Saved: {summary_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY RESULTS COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load histories
    # --------------------------------------------------------

    print("\nLoading experiment histories...")

    baseline_history = get_history(
        BASELINE_DIR
    )

    vino_history = get_history(
        VINO_DIR
    )

    pedvino_history = get_history(
        PEDVINO_DIR
    )

    # --------------------------------------------------------
    # Load final metrics
    # --------------------------------------------------------

    print("Loading experiment metrics...")

    baseline_metrics = get_metrics(
        BASELINE_DIR
    )

    vino_metrics = get_metrics(
        VINO_DIR
    )

    pedvino_metrics = get_metrics(
        PEDVINO_DIR
    )

    # --------------------------------------------------------
    # Plot validation curves
    # --------------------------------------------------------

    print("\nGenerating plots...\n")

    plot_validation_l2(
        baseline=baseline_history,
        vino=vino_history,
        pedvino=pedvino_history,
        save_path=(
            COMPARISON_DIR
            / "validation_relative_l2.png"
        ),
    )

    # --------------------------------------------------------
    # Plot test curves
    # --------------------------------------------------------

    plot_test_l2(
        baseline=baseline_history,
        vino=vino_history,
        pedvino=pedvino_history,
        save_path=(
            COMPARISON_DIR
            / "test_relative_l2.png"
        ),
    )

    # --------------------------------------------------------
    # PEDVINO internal losses
    # --------------------------------------------------------

    plot_pedvino_loss_components(
        pedvino=pedvino_history,
        save_path=(
            COMPARISON_DIR
            / "pedvino_loss_components.png"
        ),
    )

    # --------------------------------------------------------
    # Final Relative L2
    # --------------------------------------------------------

    plot_final_test_l2(
        baseline_metrics=baseline_metrics,
        vino_metrics=vino_metrics,
        pedvino_metrics=pedvino_metrics,
        save_path=(
            COMPARISON_DIR
            / "final_test_relative_l2.png"
        ),
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    plot_final_metrics(
        baseline_metrics=baseline_metrics,
        vino_metrics=vino_metrics,
        pedvino_metrics=pedvino_metrics,
        save_path=(
            COMPARISON_DIR
            / "final_metrics_comparison.png"
        ),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    rows = print_summary_table(
        baseline_metrics=baseline_metrics,
        vino_metrics=vino_metrics,
        pedvino_metrics=pedvino_metrics,
    )

    save_summary(
        rows=rows,
        baseline_metrics=baseline_metrics,
        vino_metrics=vino_metrics,
        pedvino_metrics=pedvino_metrics,
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"\nOutputs saved in:\n"
        f"{COMPARISON_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
