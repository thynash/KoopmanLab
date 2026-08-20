
"""
Comparison script for Allen-Cahn experiments.

Compares:

    1. Baseline KNO
    2. PEDVINO

Expected experiment structure:

experiments/allen_cahn/
│
├── baseline/
│   ├── history.json
│   └── metrics.json
│
├── pedvino/
│   ├── history.json
│   └── metrics.json
│
└── results/
    └── comparison/

The script generates:

    validation_relative_l2.png
    test_relative_l2.png
    final_test_relative_l2.png
    final_metrics_comparison.png
    comparison_summary.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from experiments.allen_cahn import config


# ============================================================
# PATHS
# ============================================================

BASELINE_HISTORY_PATH = config.BASELINE_HISTORY_PATH
BASELINE_METRICS_PATH = config.BASELINE_METRICS_PATH

PEDVINO_HISTORY_PATH = config.PEDVINO_HISTORY_PATH
PEDVINO_METRICS_PATH = config.PEDVINO_METRICS_PATH

COMPARISON_DIR = os.path.join(
    config.RESULTS_DIR,
    "comparison",
)

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path, name):
    """
    Load a JSON experiment file safely.
    """

    if not os.path.exists(path):

        print(
            f"WARNING: {name} file not found:\n{path}"
        )

        return {}

    try:

        with open(path, "r") as file:

            data = json.load(file)

        return data

    except Exception as error:

        print(
            f"WARNING: Could not load {name}:\n"
            f"{error}"
        )

        return {}


# ============================================================
# GET HISTORY VALUES
# ============================================================

def get_history_value(
    history,
    possible_names,
):
    """
    Return the first matching history field.

    Handles minor naming differences between experiments.
    """

    for name in possible_names:

        if name in history:

            values = history[name]

            if values is not None:

                return values

    return None


# ============================================================
# GET METRIC
# ============================================================

def get_metric(
    metrics,
    possible_names,
    default=np.nan,
):
    """
    Return first available metric.
    """

    for name in possible_names:

        if name in metrics:

            return metrics[name]

    return default


# ============================================================
# EXTRACT EPOCHS
# ============================================================

def get_epochs(history):

    epochs = get_history_value(
        history,
        [
            "epoch",
            "epochs",
        ],
    )

    if epochs is None:

        return None

    return np.asarray(epochs)


# ============================================================
# EXTRACT VALIDATION RELATIVE L2
# ============================================================

def get_validation_l2(history):

    return get_history_value(
        history,
        [
            "validation_relative_l2",
            "val_relative_l2",
            "validation_l2",
            "val_l2",
        ],
    )


# ============================================================
# EXTRACT TEST RELATIVE L2
# ============================================================

def get_test_l2(history):

    return get_history_value(
        history,
        [
            "test_relative_l2",
            "test_l2",
        ],
    )


# ============================================================
# PLOT VALIDATION CURVES
# ============================================================

def plot_validation_relative_l2(
    baseline_history,
    pedvino_history,
):
    """
    Plot validation Relative L2 versus epoch.
    """

    plt.figure(
        figsize=(9, 6),
    )

    plotted = False

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_l2 = get_validation_l2(
        baseline_history,
    )

    baseline_epochs = get_epochs(
        baseline_history,
    )

    if baseline_l2 is not None:

        if baseline_epochs is None:

            baseline_epochs = np.arange(
                1,
                len(baseline_l2) + 1,
            )

        plt.plot(
            baseline_epochs,
            baseline_l2,
            label="Baseline KNO",
            linewidth=2,
        )

        plotted = True

    else:

        print(
            "WARNING: Validation Relative L2 "
            "not found for Baseline KNO."
        )

    # --------------------------------------------------------
    # PEDVINO
    # --------------------------------------------------------

    pedvino_l2 = get_validation_l2(
        pedvino_history,
    )

    pedvino_epochs = get_epochs(
        pedvino_history,
    )

    if pedvino_l2 is not None:

        if pedvino_epochs is None:

            pedvino_epochs = np.arange(
                1,
                len(pedvino_l2) + 1,
            )

        plt.plot(
            pedvino_epochs,
            pedvino_l2,
            label="PEDVINO",
            linewidth=2,
        )

        plotted = True

    else:

        print(
            "WARNING: Validation Relative L2 "
            "not found for PEDVINO."
        )

    # --------------------------------------------------------

    if not plotted:

        plt.close()

        print(
            "WARNING: No validation curves available."
        )

        return

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Validation Relative L2",
    )

    plt.title(
        "Allen-Cahn: Validation Relative L2",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        COMPARISON_DIR,
        "validation_relative_l2.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# PLOT TEST CURVES
# ============================================================

def plot_test_relative_l2(
    baseline_history,
    pedvino_history,
):
    """
    Plot test Relative L2 versus epoch.
    """

    plt.figure(
        figsize=(9, 6),
    )

    plotted = False

    experiments = [
        (
            "Baseline KNO",
            baseline_history,
        ),
        (
            "PEDVINO",
            pedvino_history,
        ),
    ]

    for name, history in experiments:

        test_l2 = get_test_l2(
            history,
        )

        epochs = get_epochs(
            history,
        )

        if test_l2 is None:

            print(
                f"INFO: Per-epoch test Relative L2 "
                f"not available for {name}; "
                f"skipping its test curve."
            )

            continue

        if epochs is None:

            epochs = np.arange(
                1,
                len(test_l2) + 1,
            )

        plt.plot(
            epochs,
            test_l2,
            label=name,
            linewidth=2,
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "WARNING: No per-epoch test curves available."
        )

        return

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Test Relative L2",
    )

    plt.title(
        "Allen-Cahn: Test Relative L2",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        COMPARISON_DIR,
        "test_relative_l2.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# FINAL TEST RELATIVE L2 BAR PLOT
# ============================================================

def plot_final_test_l2(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Compare final test Relative L2.
    """

    names = [
        "Baseline KNO",
        "PEDVINO",
    ]

    values = [

        get_metric(
            baseline_metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),

        get_metric(
            pedvino_metrics,
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),
    ]

    valid_names = []
    valid_values = []

    for name, value in zip(
        names,
        values,
    ):

        if np.isfinite(value):

            valid_names.append(name)
            valid_values.append(value)

        else:

            print(
                f"WARNING: Final test Relative L2 "
                f"not found for {name}."
            )

    if len(valid_values) == 0:

        print(
            "WARNING: No final test Relative L2 "
            "metrics available."
        )

        return

    plt.figure(
        figsize=(8, 6),
    )

    bars = plt.bar(
        valid_names,
        valid_values,
    )

    plt.ylabel(
        "Test Relative L2",
    )

    plt.title(
        "Allen-Cahn: Final Test Relative L2",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        valid_values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4e}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    save_path = os.path.join(
        COMPARISON_DIR,
        "final_test_relative_l2.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# MULTI-METRIC COMPARISON
# ============================================================

def plot_final_metrics(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Compare:

        Test Relative L2
        Test MSE
        Trainable Parameters
    """

    metrics_to_compare = [

        (
            "Test Relative L2",
            [
                "test_relative_l2",
                "final_test_relative_l2",
            ],
        ),

        (
            "Test MSE",
            [
                "test_mse",
                "final_test_mse",
            ],
        ),

        (
            "Parameters",
            [
                "trainable_parameters",
                "parameters",
            ],
        ),
    ]

    method_names = [
        "Baseline KNO",
        "PEDVINO",
    ]

    method_metrics = [
        baseline_metrics,
        pedvino_metrics,
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    for axis, (
        metric_name,
        possible_names,
    ) in zip(
        axes,
        metrics_to_compare,
    ):

        values = [

            get_metric(
                metrics,
                possible_names,
            )

            for metrics in method_metrics
        ]

        valid_names = []
        valid_values = []

        for name, value in zip(
            method_names,
            values,
        ):

            if np.isfinite(value):

                valid_names.append(name)
                valid_values.append(value)

        if len(valid_values) == 0:

            axis.set_title(
                f"{metric_name}\nUnavailable"
            )

            continue

        bars = axis.bar(
            valid_names,
            valid_values,
        )

        axis.set_title(
            metric_name,
        )

        axis.grid(
            axis="y",
            alpha=0.3,
        )

        for bar, value in zip(
            bars,
            valid_values,
        ):

            axis.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Allen-Cahn: Final Experiment Comparison",
        fontsize=14,
    )

    plt.tight_layout()

    save_path = os.path.join(
        COMPARISON_DIR,
        "final_metrics_comparison.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_comparison_summary(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Create comparison_summary.json.
    """

    summary = {

        "experiment": "Allen-Cahn",

        "methods": {

            "KNO": {

                "model": get_metric(
                    baseline_metrics,
                    ["model"],
                    default="Original KNO1d",
                ),

                "test_relative_l2": get_metric(
                    baseline_metrics,
                    [
                        "test_relative_l2",
                        "final_test_relative_l2",
                    ],
                ),

                "test_mse": get_metric(
                    baseline_metrics,
                    [
                        "test_mse",
                        "final_test_mse",
                    ],
                ),

                "trainable_parameters": get_metric(
                    baseline_metrics,
                    [
                        "trainable_parameters",
                        "parameters",
                    ],
                ),

                "best_epoch": get_metric(
                    baseline_metrics,
                    ["best_epoch"],
                ),

                "best_validation_relative_l2": get_metric(
                    baseline_metrics,
                    [
                        "best_validation_relative_l2",
                        "validation_relative_l2",
                    ],
                ),
            },

            "PEDVINO": {

                "model": get_metric(
                    pedvino_metrics,
                    ["model"],
                    default="PEDVINO",
                ),

                "test_relative_l2": get_metric(
                    pedvino_metrics,
                    [
                        "test_relative_l2",
                        "final_test_relative_l2",
                    ],
                ),

                "test_mse": get_metric(
                    pedvino_metrics,
                    [
                        "test_mse",
                        "final_test_mse",
                    ],
                ),

                "trainable_parameters": get_metric(
                    pedvino_metrics,
                    [
                        "trainable_parameters",
                        "parameters",
                    ],
                ),

                "best_epoch": get_metric(
                    pedvino_metrics,
                    ["best_epoch"],
                ),

                "best_validation_relative_l2": get_metric(
                    pedvino_metrics,
                    [
                        "best_validation_relative_l2",
                        "validation_relative_l2",
                    ],
                ),
            },
        },
    }

    return summary


# ============================================================
# PRINT TABLE
# ============================================================

def print_comparison_table(summary):

    print("\n" + "=" * 85)

    print(
        "ALLEN-CAHN EXPERIMENT COMPARISON"
    )

    print("=" * 85)

    print(
        f"{'Method':<20}"
        f"{'Test Rel L2':>20}"
        f"{'Test MSE':>20}"
        f"{'Parameters':>20}"
    )

    print("-" * 85)

    methods = summary["methods"]

    for method_name, method_data in methods.items():

        relative_l2 = method_data[
            "test_relative_l2"
        ]

        mse = method_data[
            "test_mse"
        ]

        parameters = method_data[
            "trainable_parameters"
        ]

        relative_l2_text = (
            f"{relative_l2:.6e}"
            if np.isfinite(relative_l2)
            else "N/A"
        )

        mse_text = (
            f"{mse:.6e}"
            if np.isfinite(mse)
            else "N/A"
        )

        parameters_text = (
            f"{int(parameters):,}"
            if np.isfinite(parameters)
            else "N/A"
        )

        print(
            f"{method_name:<20}"
            f"{relative_l2_text:>20}"
            f"{mse_text:>20}"
            f"{parameters_text:>20}"
        )

    print("=" * 85)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("LOADING ALLEN-CAHN EXPERIMENT RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD HISTORIES
    # --------------------------------------------------------

    print("\nLoading experiment histories...")

    baseline_history = load_json(
        BASELINE_HISTORY_PATH,
        "Baseline history",
    )

    pedvino_history = load_json(
        PEDVINO_HISTORY_PATH,
        "PEDVINO history",
    )

    # --------------------------------------------------------
    # LOAD METRICS
    # --------------------------------------------------------

    print("Loading experiment metrics...")

    baseline_metrics = load_json(
        BASELINE_METRICS_PATH,
        "Baseline metrics",
    )

    pedvino_metrics = load_json(
        PEDVINO_METRICS_PATH,
        "PEDVINO metrics",
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print("\nGenerating plots...")

    plot_validation_relative_l2(
        baseline_history,
        pedvino_history,
    )

    plot_test_relative_l2(
        baseline_history,
        pedvino_history,
    )

    plot_final_test_l2(
        baseline_metrics,
        pedvino_metrics,
    )

    plot_final_metrics(
        baseline_metrics,
        pedvino_metrics,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = create_comparison_summary(
        baseline_metrics,
        pedvino_metrics,
    )

    print_comparison_table(
        summary,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path = os.path.join(
        COMPARISON_DIR,
        "comparison_summary.json",
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
        f"\nSaved: {summary_path}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print(
        "\nOutputs saved in:"
    )

    print(
        COMPARISON_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
