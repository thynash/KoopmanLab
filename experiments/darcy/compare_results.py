import os
import json

import numpy as np
import matplotlib.pyplot as plt

from experiments.darcy import config


# ============================================================
# PATHS
# ============================================================

BASELINE_HISTORY_PATH = os.path.join(
    config.BASELINE_RESULTS_DIR,
    "history.json",
)

PEDVINO_HISTORY_PATH = os.path.join(
    config.PEDVINO_RESULTS_DIR,
    "history.json",
)

BASELINE_METRICS_PATH = os.path.join(
    config.BASELINE_RESULTS_DIR,
    "metrics.json",
)

PEDVINO_METRICS_PATH = os.path.join(
    config.PEDVINO_RESULTS_DIR,
    "metrics.json",
)

COMPARISON_DIR = getattr(
    config,
    "COMPARISON_RESULTS_DIR",
    os.path.join(
        config.RESULTS_DIR,
        "comparison",
    ),
)


# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    """
    Load a JSON file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def ensure_directory(path):
    """
    Create directory if necessary.
    """

    os.makedirs(
        path,
        exist_ok=True,
    )


def get_epochs(history):
    """
    Return epochs from history.

    Falls back to 1, 2, ..., N if epoch information
    is unavailable.
    """

    if (
        "epoch" in history
        and len(history["epoch"]) > 0
    ):
        return history["epoch"]

    # Find a usable metric length.
    for split in (
        "train",
        "validation",
        "test",
    ):
        if split in history:
            for values in history[split].values():
                return list(
                    range(
                        1,
                        len(values) + 1,
                    )
                )

    return []


def get_metric(
    history,
    split,
    metric_name,
):
    """
    Safely retrieve a metric list.

    Returns None if the metric does not exist.
    """

    if split not in history:
        return None

    if metric_name not in history[split]:
        return None

    values = history[split][metric_name]

    if values is None:
        return None

    if len(values) == 0:
        return None

    return values


def plot_two_curves(
    epochs_1,
    values_1,
    label_1,
    epochs_2,
    values_2,
    label_2,
    title,
    ylabel,
    save_path,
    log_scale=False,
):
    """
    Plot two experiment curves.
    """

    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    if (
        values_1 is not None
        and len(values_1) > 0
    ):
        plt.plot(
            epochs_1[:len(values_1)],
            values_1,
            label=label_1,
            linewidth=2,
        )
        plotted = True

    if (
        values_2 is not None
        and len(values_2) > 0
    ):
        plt.plot(
            epochs_2[:len(values_2)],
            values_2,
            label=label_2,
            linewidth=2,
        )
        plotted = True

    if not plotted:
        plt.close()
        print(
            f"Skipped plot because no data was found: "
            f"{title}"
        )
        return False

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )

    return True


# ============================================================
# PLOT 1
# VALIDATION RELATIVE L2 COMPARISON
# ============================================================

def plot_validation_l2(
    baseline_history,
    pedvino_history,
):
    """
    Compare validation Relative L2 over training.
    """

    baseline_epochs = get_epochs(
        baseline_history
    )

    pedvino_epochs = get_epochs(
        pedvino_history
    )

    baseline_values = get_metric(
        baseline_history,
        "validation",
        "relative_l2",
    )

    pedvino_values = get_metric(
        pedvino_history,
        "validation",
        "relative_l2",
    )

    save_path = os.path.join(
        COMPARISON_DIR,
        "01_validation_relative_l2.png",
    )

    plot_two_curves(
        baseline_epochs,
        baseline_values,
        "Baseline KNO",
        pedvino_epochs,
        pedvino_values,
        "PEDVINO",
        "Validation Relative L2 Error",
        "Relative L2 Error",
        save_path,
    )


# ============================================================
# PLOT 2
# TEST RELATIVE L2 COMPARISON
# ============================================================

def plot_test_l2(
    baseline_history,
    pedvino_history,
):
    """
    Compare test Relative L2 over training.
    """

    baseline_epochs = get_epochs(
        baseline_history
    )

    pedvino_epochs = get_epochs(
        pedvino_history
    )

    baseline_values = get_metric(
        baseline_history,
        "test",
        "relative_l2",
    )

    pedvino_values = get_metric(
        pedvino_history,
        "test",
        "relative_l2",
    )

    save_path = os.path.join(
        COMPARISON_DIR,
        "02_test_relative_l2.png",
    )

    plot_two_curves(
        baseline_epochs,
        baseline_values,
        "Baseline KNO",
        pedvino_epochs,
        pedvino_values,
        "PEDVINO",
        "Test Relative L2 Error",
        "Relative L2 Error",
        save_path,
    )


# ============================================================
# PLOT 3
# TRAINING PREDICTION LOSS COMPARISON
# ============================================================

def plot_prediction_loss(
    baseline_history,
    pedvino_history,
):
    """
    Compare the supervised prediction loss.

    This is a fair loss comparison because both models
    learn from the solution target.
    """

    baseline_epochs = get_epochs(
        baseline_history
    )

    pedvino_epochs = get_epochs(
        pedvino_history
    )

    baseline_values = get_metric(
        baseline_history,
        "train",
        "prediction_loss",
    )

    pedvino_values = get_metric(
        pedvino_history,
        "train",
        "prediction_loss",
    )

    save_path = os.path.join(
        COMPARISON_DIR,
        "03_prediction_loss.png",
    )

    plot_two_curves(
        baseline_epochs,
        baseline_values,
        "Baseline KNO",
        pedvino_epochs,
        pedvino_values,
        "PEDVINO",
        "Training Prediction Loss",
        "Prediction Loss",
        save_path,
        log_scale=True,
    )


# ============================================================
# PLOT 4
# PEDVINO INDIVIDUAL LOSS COMPONENTS
# ============================================================

def plot_pedvino_losses(
    pedvino_history,
):
    """
    Automatically detect and plot all individual PEDVINO
    training losses.

    Expected possible keys include:

        total_loss
        prediction_loss
        reconstruction_loss
        energy_loss
        gradient_loss
        boundary_loss

    Any additional future loss component will also be
    detected automatically if its name contains 'loss'.
    """

    epochs = get_epochs(
        pedvino_history
    )

    train_history = pedvino_history.get(
        "train",
        {},
    )

    if len(train_history) == 0:
        print(
            "No PEDVINO training history found."
        )
        return

    # --------------------------------------------------------
    # Prefer explicit component losses.
    # --------------------------------------------------------

    excluded_losses = {
        "total_loss",
    }

    loss_keys = []

    for key, values in train_history.items():

        if key in excluded_losses:
            continue

        if "loss" not in key.lower():
            continue

        if values is None:
            continue

        if len(values) == 0:
            continue

        loss_keys.append(key)

    if len(loss_keys) == 0:
        print(
            "No individual PEDVINO loss components found."
        )
        return

    plt.figure(
        figsize=(11, 7)
    )

    for loss_name in sorted(loss_keys):

        values = train_history[loss_name]

        plt.plot(
            epochs[:len(values)],
            values,
            linewidth=2,
            label=loss_name.replace(
                "_",
                " ",
            ).title(),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.title(
        "PEDVINO Individual Training Loss Components"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        COMPARISON_DIR,
        "04_pedvino_individual_losses.png",
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# PLOT 5
# FINAL TEST RELATIVE L2
# ============================================================

def plot_final_test_l2(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Bar plot comparing final test Relative L2.
    """

    baseline_value = baseline_metrics.get(
        "test_relative_l2",
        None,
    )

    pedvino_value = pedvino_metrics.get(
        "test_relative_l2",
        None,
    )

    if (
        baseline_value is None
        or pedvino_value is None
    ):
        print(
            "Skipped final test L2 plot because "
            "metrics.json does not contain "
            "test_relative_l2."
        )
        return

    models = [
        "Baseline KNO",
        "PEDVINO",
    ]

    values = [
        baseline_value,
        pedvino_value,
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        models,
        values,
    )

    plt.ylabel(
        "Test Relative L2 Error"
    )

    plt.title(
        "Final Test Relative L2 Comparison"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
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
        "05_final_test_relative_l2.png",
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# SUMMARY
# ============================================================

def get_improvement(
    baseline_value,
    pedvino_value,
):
    """
    Percentage improvement.

    Positive:
        PEDVINO has lower error.

    Negative:
        PEDVINO has higher error.
    """

    if baseline_value == 0:
        return None

    return (
        (baseline_value - pedvino_value)
        / baseline_value
        * 100.0
    )


def build_summary(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Create comparison summary.
    """

    baseline_test_l2 = baseline_metrics.get(
        "test_relative_l2",
        None,
    )

    pedvino_test_l2 = pedvino_metrics.get(
        "test_relative_l2",
        None,
    )

    improvement = None

    if (
        baseline_test_l2 is not None
        and pedvino_test_l2 is not None
    ):
        improvement = get_improvement(
            baseline_test_l2,
            pedvino_test_l2,
        )

    summary = {
        "experiment": "Darcy Flow",

        "baseline": baseline_metrics,

        "pedvino": pedvino_metrics,

        "pedvino_test_relative_l2_improvement_percent":
            improvement,
    }

    return summary


def print_summary(
    baseline_metrics,
    pedvino_metrics,
    summary,
):
    """
    Print a readable terminal comparison.
    """

    print()
    print("=" * 70)
    print("DARCY FLOW FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<35}"
        f"{'Baseline KNO':>18}"
        f"{'PEDVINO':>18}"
    )

    print("-" * 70)

    comparison_keys = [
        (
            "Best Epoch",
            "best_epoch",
            ".0f",
        ),
        (
            "Validation Relative L2",
            "best_validation_relative_l2",
            ".6e",
        ),
        (
            "Test Relative L2",
            "test_relative_l2",
            ".6e",
        ),
        (
            "Test MSE",
            "test_mse",
            ".6e",
        ),
        (
            "Trainable Parameters",
            "trainable_parameters",
            ",.0f",
        ),
        (
            "Training Time (s)",
            "training_time_seconds",
            ".3f",
        ),
    ]

    for (
        display_name,
        key,
        format_spec,
    ) in comparison_keys:

        baseline_value = baseline_metrics.get(
            key,
            None,
        )

        pedvino_value = pedvino_metrics.get(
            key,
            None,
        )

        baseline_text = (
            format(
                baseline_value,
                format_spec,
            )
            if baseline_value is not None
            else "N/A"
        )

        pedvino_text = (
            format(
                pedvino_value,
                format_spec,
            )
            if pedvino_value is not None
            else "N/A"
        )

        print(
            f"{display_name:<35}"
            f"{baseline_text:>18}"
            f"{pedvino_text:>18}"
        )

    print("-" * 70)

    improvement = summary.get(
        "pedvino_test_relative_l2_improvement_percent",
        None,
    )

    if improvement is not None:

        print(
            "PEDVINO Relative L2 improvement: "
            f"{improvement:.2f}%"
        )

    else:

        print(
            "PEDVINO Relative L2 improvement: N/A"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY FLOW - RESULTS COMPARISON")
    print("=" * 70)

    ensure_directory(
        COMPARISON_DIR
    )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    baseline_history = load_json(
        BASELINE_HISTORY_PATH
    )

    pedvino_history = load_json(
        PEDVINO_HISTORY_PATH
    )

    baseline_metrics = load_json(
        BASELINE_METRICS_PATH
    )

    pedvino_metrics = load_json(
        PEDVINO_METRICS_PATH
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print()
    print("Generating comparison plots...")
    print()

    plot_validation_l2(
        baseline_history,
        pedvino_history,
    )

    plot_test_l2(
        baseline_history,
        pedvino_history,
    )

    plot_prediction_loss(
        baseline_history,
        pedvino_history,
    )

    plot_pedvino_losses(
        pedvino_history
    )

    plot_final_test_l2(
        baseline_metrics,
        pedvino_metrics,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        baseline_metrics,
        pedvino_metrics,
    )

    summary_path = os.path.join(
        COMPARISON_DIR,
        "comparison_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print_summary(
        baseline_metrics,
        pedvino_metrics,
        summary,
    )

    print()
    print(
        f"Summary saved: {summary_path}"
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)

    print()
    print("All plots are saved in:")
    print(
        COMPARISON_DIR
    )


if __name__ == "__main__":
    main()
