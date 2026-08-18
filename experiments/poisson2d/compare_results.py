import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DEFAULT_EXPERIMENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DEFAULT_RESULTS_DIR = os.path.join(
    DEFAULT_EXPERIMENT_DIR,
    "results",
)

DEFAULT_BASELINE_DIR = os.path.join(
    DEFAULT_RESULTS_DIR,
    "baseline",
)

DEFAULT_PEDVINO_DIR = os.path.join(
    DEFAULT_RESULTS_DIR,
    "pedvino",
)

DEFAULT_COMPARISON_DIR = os.path.join(
    DEFAULT_RESULTS_DIR,
    "comparison",
)


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path):
    """
    Load a JSON file safely.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nRequired file was not found:\n{path}\n"
        )

    with open(path, "r") as file:
        return json.load(file)


def get_history_value(history, key, index):
    """
    Safely extract one history value.

    Returns NaN if the key/index is unavailable.
    """

    values = history.get(key, [])

    if index >= len(values):
        return np.nan

    value = values[index]

    if value is None:
        return np.nan

    return float(value)


# ============================================================
# HISTORY NORMALIZATION
# ============================================================

def normalize_history(history):
    """
    Convert different possible history formats into a common form.

    Expected output:

        {
            "epoch": [...],
            "train_total_loss": [...],
            "prediction_loss": [...],
            "reconstruction_loss": [...],
            "energy_loss": [...],
            "gradient_loss": [...],
            "boundary_loss": [...],
            "validation_relative_l2": [...],
            "test_relative_l2": [...]
        }
    """

    normalized = {}

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------

    if "epoch" in history:
        normalized["epoch"] = history["epoch"]

    elif "epochs" in history:
        normalized["epoch"] = history["epochs"]

    else:
        # Infer epoch count from available arrays.
        max_length = 0

        for value in history.values():
            if isinstance(value, list):
                max_length = max(
                    max_length,
                    len(value),
                )

        normalized["epoch"] = list(
            range(1, max_length + 1)
        )

    n_epochs = len(
        normalized["epoch"]
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def find_key(candidates):
        for key in candidates:
            if key in history:
                return key
        return None

    def extract(candidates):
        key = find_key(candidates)

        if key is None:
            return [np.nan] * n_epochs

        values = history[key]

        if not isinstance(values, list):
            return [np.nan] * n_epochs

        result = []

        for i in range(n_epochs):

            if i < len(values):
                value = values[i]

                if value is None:
                    result.append(np.nan)
                else:
                    result.append(float(value))

            else:
                result.append(np.nan)

        return result

    # --------------------------------------------------------
    # Total training loss
    # --------------------------------------------------------

    normalized["train_total_loss"] = extract([
        "train_total_loss",
        "total_loss",
        "train_loss",
    ])

    # --------------------------------------------------------
    # Prediction loss
    # --------------------------------------------------------

    normalized["prediction_loss"] = extract([
        "prediction_loss",
        "pred_loss",
        "train_prediction_loss",
    ])

    # --------------------------------------------------------
    # Reconstruction loss
    # --------------------------------------------------------

    normalized["reconstruction_loss"] = extract([
        "reconstruction_loss",
        "recon_loss",
        "train_reconstruction_loss",
    ])

    # --------------------------------------------------------
    # Energy loss
    # --------------------------------------------------------

    normalized["energy_loss"] = extract([
        "energy_loss",
        "variational_loss",
        "var_loss",
        "train_energy_loss",
    ])

    # --------------------------------------------------------
    # Gradient loss
    # --------------------------------------------------------

    normalized["gradient_loss"] = extract([
        "gradient_loss",
        "grad_loss",
        "train_gradient_loss",
    ])

    # --------------------------------------------------------
    # Boundary loss
    # --------------------------------------------------------

    normalized["boundary_loss"] = extract([
        "boundary_loss",
        "bc_loss",
        "train_boundary_loss",
    ])

    # --------------------------------------------------------
    # Validation Relative L2
    # --------------------------------------------------------

    normalized["validation_relative_l2"] = extract([
        "validation_relative_l2",
        "val_relative_l2",
        "val_l2",
    ])

    # --------------------------------------------------------
    # Test Relative L2
    # --------------------------------------------------------

    normalized["test_relative_l2"] = extract([
        "test_relative_l2",
        "test_l2",
    ])

    return normalized


# ============================================================
# PLOT SETTINGS
# ============================================================

def prepare_axis(ax):
    """
    Apply common formatting.
    """

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )


# ============================================================
# PLOT 1
# VALIDATION RELATIVE L2
# ============================================================

def plot_validation_l2(
    baseline,
    pedvino,
    save_path,
):
    """
    Compare validation Relative L2 over training.
    """

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        baseline["epoch"],
        baseline["validation_relative_l2"],
        label="KNO",
        linewidth=2,
    )

    plt.plot(
        pedvino["epoch"],
        pedvino["validation_relative_l2"],
        label="PEDVINO",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Relative L2 Error")

    plt.title(
        "Validation Relative L2 Error Comparison"
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
    pedvino,
    save_path,
):
    """
    Compare test Relative L2 over training.
    """

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        baseline["epoch"],
        baseline["test_relative_l2"],
        label="KNO",
        linewidth=2,
    )

    plt.plot(
        pedvino["epoch"],
        pedvino["test_relative_l2"],
        label="PEDVINO",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Relative L2 Error")

    plt.title(
        "Test Relative L2 Error Comparison"
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
# PEDVINO INDIVIDUAL LOSS COMPONENTS
# ============================================================

def plot_pedvino_loss_components(
    pedvino,
    save_path,
):
    """
    Plot PEDVINO's individual objective components.

    This intentionally does NOT combine KNO and PEDVINO losses.

    The plot shows the internal optimization behaviour of PEDVINO:

        L_pred
        L_recon
        L_energy
        L_grad
        L_bc
    """

    epochs = np.asarray(
        pedvino["epoch"]
    )

    prediction_loss = np.asarray(
        pedvino["prediction_loss"],
        dtype=float,
    )

    reconstruction_loss = np.asarray(
        pedvino["reconstruction_loss"],
        dtype=float,
    )

    energy_loss = np.asarray(
        pedvino["energy_loss"],
        dtype=float,
    )

    gradient_loss = np.asarray(
        pedvino["gradient_loss"],
        dtype=float,
    )

    boundary_loss = np.asarray(
        pedvino["boundary_loss"],
        dtype=float,
    )

    plt.figure(
        figsize=(11, 7)
    )

    # --------------------------------------------------------
    # Plot only components that actually exist.
    # --------------------------------------------------------

    plotted_anything = False

    if np.isfinite(prediction_loss).any():

        plt.plot(
            epochs,
            prediction_loss,
            label="Prediction Loss",
            linewidth=2,
        )

        plotted_anything = True

    if np.isfinite(reconstruction_loss).any():

        plt.plot(
            epochs,
            reconstruction_loss,
            label="Reconstruction Loss",
            linewidth=2,
        )

        plotted_anything = True

    if np.isfinite(energy_loss).any():

        plt.plot(
            epochs,
            energy_loss,
            label="Energy Loss",
            linewidth=2,
        )

        plotted_anything = True

    if np.isfinite(gradient_loss).any():

        plt.plot(
            epochs,
            gradient_loss,
            label="Gradient Loss",
            linewidth=2,
        )

        plotted_anything = True

    if np.isfinite(boundary_loss).any():

        plt.plot(
            epochs,
            boundary_loss,
            label="Boundary Loss",
            linewidth=2,
        )

        plotted_anything = True

    if not plotted_anything:

        plt.close()

        print(
            "\nWARNING: No PEDVINO individual loss "
            "components were found in history.json."
        )

        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.title(
        "PEDVINO Individual Loss Components"
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
# PLOT 4
# FINAL TEST RELATIVE L2
# ============================================================

def plot_final_test_l2(
    baseline_metrics,
    pedvino_metrics,
    save_path,
):
    """
    Bar chart of final test Relative L2.
    """

    models = [
        "KNO",
        "PEDVINO",
    ]

    values = [
        baseline_metrics[
            "test_relative_l2"
        ],
        pedvino_metrics[
            "test_relative_l2"
        ],
    ]

    plt.figure(
        figsize=(7, 6)
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
        axis="y",
        linestyle="--",
        alpha=0.4,
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

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

def create_summary(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Create the final comparison dictionary.
    """

    baseline_test = float(
        baseline_metrics[
            "test_relative_l2"
        ]
    )

    pedvino_test = float(
        pedvino_metrics[
            "test_relative_l2"
        ]
    )

    improvement = (
        (baseline_test - pedvino_test)
        / baseline_test
        * 100.0
    )

    summary = {
        "KNO": baseline_metrics,
        "PEDVINO": pedvino_metrics,
        "PEDVINO_relative_l2_improvement_percent":
            improvement,
    }

    return summary


def print_final_comparison(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Print important final metrics.
    """

    print()
    print("=" * 70)
    print("POISSON2D FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<35}"
        f"{'KNO':>16}"
        f"{'PEDVINO':>16}"
    )

    print("-" * 70)

    rows = [
        (
            "Best Epoch",
            baseline_metrics.get(
                "best_epoch",
                np.nan,
            ),
            pedvino_metrics.get(
                "best_epoch",
                np.nan,
            ),
        ),
        (
            "Validation Relative L2",
            baseline_metrics.get(
                "best_validation_relative_l2",
                np.nan,
            ),
            pedvino_metrics.get(
                "best_validation_relative_l2",
                np.nan,
            ),
        ),
        (
            "Test Relative L2",
            baseline_metrics.get(
                "test_relative_l2",
                np.nan,
            ),
            pedvino_metrics.get(
                "test_relative_l2",
                np.nan,
            ),
        ),
        (
            "Test MSE",
            baseline_metrics.get(
                "test_mse",
                np.nan,
            ),
            pedvino_metrics.get(
                "test_mse",
                np.nan,
            ),
        ),
        (
            "Trainable Parameters",
            baseline_metrics.get(
                "trainable_parameters",
                np.nan,
            ),
            pedvino_metrics.get(
                "trainable_parameters",
                np.nan,
            ),
        ),
        (
            "Training Time (s)",
            baseline_metrics.get(
                "training_time_seconds",
                np.nan,
            ),
            pedvino_metrics.get(
                "training_time_seconds",
                np.nan,
            ),
        ),
    ]

    for name, baseline_value, pedvino_value in rows:

        if name in [
            "Best Epoch",
            "Trainable Parameters",
        ]:

            baseline_string = (
                f"{int(baseline_value):d}"
                if np.isfinite(
                    baseline_value
                )
                else "N/A"
            )

            pedvino_string = (
                f"{int(pedvino_value):d}"
                if np.isfinite(
                    pedvino_value
                )
                else "N/A"
            )

        else:

            baseline_string = (
                f"{baseline_value:.6e}"
                if np.isfinite(
                    baseline_value
                )
                else "N/A"
            )

            pedvino_string = (
                f"{pedvino_value:.6e}"
                if np.isfinite(
                    pedvino_value
                )
                else "N/A"
            )

        print(
            f"{name:<35}"
            f"{baseline_string:>16}"
            f"{pedvino_string:>16}"
        )

    print("-" * 70)

    baseline_test = float(
        baseline_metrics[
            "test_relative_l2"
        ]
    )

    pedvino_test = float(
        pedvino_metrics[
            "test_relative_l2"
        ]
    )

    improvement = (
        (baseline_test - pedvino_test)
        / baseline_test
        * 100.0
    )

    print(
        f"PEDVINO Relative L2 improvement: "
        f"{improvement:.2f}%"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare KNO and PEDVINO Poisson2D experiments."
        )
    )

    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=DEFAULT_BASELINE_DIR,
    )

    parser.add_argument(
        "--pedvino_dir",
        type=str,
        default=DEFAULT_PEDVINO_DIR,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_COMPARISON_DIR,
    )

    args = parser.parse_args()

    baseline_dir = args.baseline_dir
    pedvino_dir = args.pedvino_dir
    comparison_dir = args.output_dir

    os.makedirs(
        comparison_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Required paths
    # --------------------------------------------------------

    baseline_history_path = os.path.join(
        baseline_dir,
        "history.json",
    )

    pedvino_history_path = os.path.join(
        pedvino_dir,
        "history.json",
    )

    baseline_metrics_path = os.path.join(
        baseline_dir,
        "metrics.json",
    )

    pedvino_metrics_path = os.path.join(
        pedvino_dir,
        "metrics.json",
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        f"\nBaseline directory:\n{baseline_dir}"
    )

    print(
        f"\nPEDVINO directory:\n{pedvino_dir}"
    )

    baseline_history_raw = load_json(
        baseline_history_path
    )

    pedvino_history_raw = load_json(
        pedvino_history_path
    )

    baseline_metrics = load_json(
        baseline_metrics_path
    )

    pedvino_metrics = load_json(
        pedvino_metrics_path
    )

    baseline_history = normalize_history(
        baseline_history_raw
    )

    pedvino_history = normalize_history(
        pedvino_history_raw
    )

    # ========================================================
    # PLOT 1
    # ========================================================

    plot_validation_l2(
        baseline=baseline_history,
        pedvino=pedvino_history,
        save_path=os.path.join(
            comparison_dir,
            "01_validation_relative_l2.png",
        ),
    )

    # ========================================================
    # PLOT 2
    # ========================================================

    plot_test_l2(
        baseline=baseline_history,
        pedvino=pedvino_history,
        save_path=os.path.join(
            comparison_dir,
            "02_test_relative_l2.png",
        ),
    )

    # ========================================================
    # PLOT 3
    #
    # Only PEDVINO internal components.
    #
    # No meaningless mixing with KNO losses.
    # ========================================================

    plot_pedvino_loss_components(
        pedvino=pedvino_history,
        save_path=os.path.join(
            comparison_dir,
            "03_pedvino_individual_loss_components.png",
        ),
    )

    # ========================================================
    # PLOT 4
    # ========================================================

    plot_final_test_l2(
        baseline_metrics=baseline_metrics,
        pedvino_metrics=pedvino_metrics,
        save_path=os.path.join(
            comparison_dir,
            "04_final_test_relative_l2.png",
        ),
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print_final_comparison(
        baseline_metrics=baseline_metrics,
        pedvino_metrics=pedvino_metrics,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = create_summary(
        baseline_metrics=baseline_metrics,
        pedvino_metrics=pedvino_metrics,
    )

    summary_path = os.path.join(
        comparison_dir,
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

    print()
    print(
        f"Summary saved: {summary_path}"
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)

    print(
        "\nGenerated plots:"
    )

    print(
        "1. 01_validation_relative_l2.png"
    )

    print(
        "2. 02_test_relative_l2.png"
    )

    print(
        "3. 03_pedvino_individual_loss_components.png"
    )

    print(
        "4. 04_final_test_relative_l2.png"
    )

    print(
        f"\nAll plots are saved in:\n{comparison_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
