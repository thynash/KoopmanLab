# ============================================================
# BURGERS 1D
# BASELINE KNO vs FULL PEDVINO COMPARISON
# ============================================================

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT DIRECTORIES
#
# Expected experiment structure:
#
# experiments/burgers/
# ├── train_baseline.py
# ├── train_pedvino.py
# ├── compare_results.py
# │
# └── results/
#     ├── baseline/
#     │   ├── history.json
#     │   └── metrics.json
#     │
#     ├── pedvino/
#     │   ├── history.json
#     │   └── metrics.json
#     │
#     └── comparison/
# ============================================================

DEFAULT_BASELINE_DIR = (
    "experiments/burgers/results/baseline"
)

DEFAULT_PEDVINO_DIR = (
    "experiments/burgers/results/pedvino"
)

DEFAULT_COMPARISON_DIR = (
    "experiments/burgers/results/comparison"
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

    with open(
        path,
        "r",
    ) as file:

        return json.load(file)


# ============================================================
# METRIC EXTRACTION
# ============================================================

def get_metric(
    metrics,
    name,
    split=None,
    default=np.nan,
):
    """
    Extract a metric from either:

    Format A:
        {
            "test_relative_l2": ...,
            "test_mse": ...
        }

    or our current Burgers format:

        {
            "train": {...},
            "validation": {...},
            "test": {
                "relative_l2": ...,
                "mse": ...
            }
        }
    """

    # --------------------------------------------------------
    # Direct top-level key
    # --------------------------------------------------------

    if name in metrics:
        return metrics[name]

    # --------------------------------------------------------
    # Nested split
    # --------------------------------------------------------

    if (
        split is not None
        and split in metrics
        and isinstance(metrics[split], dict)
    ):

        nested = metrics[split]

        if name in nested:
            return nested[name]

    return default


# ============================================================
# NORMALIZE HISTORY
# ============================================================

def normalize_history(history):
    """
    Ensure all expected arrays exist.

    Missing PEDVINO-only fields are filled with NaN.
    This allows the baseline and PEDVINO histories to be
    processed by the same plotting code.
    """

    normalized = dict(history)

    # --------------------------------------------------------
    # Determine number of epochs
    # --------------------------------------------------------

    if "epoch" not in normalized:

        raise KeyError(
            "history.json does not contain an 'epoch' field."
        )

    num_epochs = len(
        normalized["epoch"]
    )

    # --------------------------------------------------------
    # Required/common fields
    # --------------------------------------------------------

    required_fields = [
        "validation_relative_l2",
        "test_relative_l2",
    ]

    for field in required_fields:

        if field not in normalized:

            normalized[field] = (
                [np.nan] * num_epochs
            )

    # --------------------------------------------------------
    # Loss fields
    #
    # Baseline and PEDVINO naming is normalized here.
    # --------------------------------------------------------

    aliases = {
        "prediction_loss": [
            "prediction_loss",
            "train_prediction_loss",
            "pred_loss",
        ],

        "reconstruction_loss": [
            "reconstruction_loss",
            "train_reconstruction_loss",
            "recon_loss",
        ],

        "energy_loss": [
            "energy_loss",
            "variational_loss",
        ],

        "gradient_loss": [
            "gradient_loss",
            "grad_loss",
        ],

        "boundary_loss": [
            "boundary_loss",
            "bc_loss",
        ],
    }

    for canonical_name, possible_names in aliases.items():

        if canonical_name in normalized:
            continue

        found = False

        for possible_name in possible_names:

            if possible_name in normalized:

                normalized[
                    canonical_name
                ] = normalized[
                    possible_name
                ]

                found = True
                break

        if not found:

            normalized[
                canonical_name
            ] = [np.nan] * num_epochs

    # --------------------------------------------------------
    # Make sure every history array has epoch length.
    # --------------------------------------------------------

    for key, value in normalized.items():

        if not isinstance(value, list):
            continue

        if len(value) == num_epochs:
            continue

        # Do not silently use malformed core fields.
        if key in [
            "epoch",
            "validation_relative_l2",
            "test_relative_l2",
        ]:
            raise ValueError(
                f"History field '{key}' has length "
                f"{len(value)}, but epoch has length "
                f"{num_epochs}."
            )

    return normalized


# ============================================================
# SAFE NUMERIC ARRAY
# ============================================================

def to_numeric_array(
    values,
    expected_length=None,
):
    """
    Convert history values into a numeric NumPy array.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if (
        expected_length is not None
        and len(array) != expected_length
    ):
        raise ValueError(
            "History length mismatch: "
            f"expected {expected_length}, "
            f"received {len(array)}."
        )

    return array


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
    Compare validation Relative L2 during training.
    """

    baseline_epochs = to_numeric_array(
        baseline["epoch"]
    )

    pedvino_epochs = to_numeric_array(
        pedvino["epoch"]
    )

    baseline_l2 = to_numeric_array(
        baseline["validation_relative_l2"],
        len(baseline_epochs),
    )

    pedvino_l2 = to_numeric_array(
        pedvino["validation_relative_l2"],
        len(pedvino_epochs),
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        baseline_epochs,
        baseline_l2,
        label="KNO Baseline",
        linewidth=2,
    )

    plt.plot(
        pedvino_epochs,
        pedvino_l2,
        label="PEDVINO",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Relative L2 Error")

    plt.title(
        "Burgers 1D: Validation Relative L2 Comparison"
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

    print(
        f"Saved: {save_path}"
    )


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

    Test values are logged for diagnostics only.
    Model selection itself remains validation based.
    """

    baseline_epochs = to_numeric_array(
        baseline["epoch"]
    )

    pedvino_epochs = to_numeric_array(
        pedvino["epoch"]
    )

    baseline_l2 = to_numeric_array(
        baseline["test_relative_l2"],
        len(baseline_epochs),
    )

    pedvino_l2 = to_numeric_array(
        pedvino["test_relative_l2"],
        len(pedvino_epochs),
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        baseline_epochs,
        baseline_l2,
        label="KNO Baseline",
        linewidth=2,
    )

    plt.plot(
        pedvino_epochs,
        pedvino_l2,
        label="PEDVINO",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Relative L2 Error")

    plt.title(
        "Burgers 1D: Test Relative L2 Comparison"
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

    print(
        f"Saved: {save_path}"
    )


# ============================================================
# PLOT 3
# PEDVINO INDIVIDUAL LOSS COMPONENTS
# ============================================================

def plot_pedvino_loss_components(
    pedvino,
    save_path,
):
    """
    Plot PEDVINO's internal objective components.

    Components:

        L_pred
        L_recon
        L_energy
        L_grad
        L_bc

    These are intentionally shown only for PEDVINO.
    We do not mix physics-specific losses with the baseline.
    """

    epochs = to_numeric_array(
        pedvino["epoch"]
    )

    num_epochs = len(epochs)

    prediction_loss = to_numeric_array(
        pedvino["prediction_loss"],
        num_epochs,
    )

    reconstruction_loss = to_numeric_array(
        pedvino["reconstruction_loss"],
        num_epochs,
    )

    energy_loss = to_numeric_array(
        pedvino["energy_loss"],
        num_epochs,
    )

    gradient_loss = to_numeric_array(
        pedvino["gradient_loss"],
        num_epochs,
    )

    boundary_loss = to_numeric_array(
        pedvino["boundary_loss"],
        num_epochs,
    )

    plt.figure(
        figsize=(11, 7)
    )

    plotted_anything = False

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if np.isfinite(prediction_loss).any():

        plt.plot(
            epochs,
            prediction_loss,
            label="Prediction Loss",
            linewidth=2,
        )

        plotted_anything = True

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    if np.isfinite(reconstruction_loss).any():

        plt.plot(
            epochs,
            reconstruction_loss,
            label="Reconstruction Loss",
            linewidth=2,
        )

        plotted_anything = True

    # --------------------------------------------------------
    # Energy / Variational
    # --------------------------------------------------------

    if np.isfinite(energy_loss).any():

        plt.plot(
            epochs,
            energy_loss,
            label="Energy Loss",
            linewidth=2,
        )

        plotted_anything = True

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    if np.isfinite(gradient_loss).any():

        plt.plot(
            epochs,
            gradient_loss,
            label="Gradient Loss",
            linewidth=2,
        )

        plotted_anything = True

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

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
            "\nWARNING: No PEDVINO loss components "
            "were found."
        )

        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.title(
        "Burgers 1D: PEDVINO Individual Loss Components"
    )

    # Loss terms can have very different scales.
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

    print(
        f"Saved: {save_path}"
    )


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
    Compare the final test Relative L2 obtained from
    the best validation checkpoint.
    """

    baseline_test_l2 = float(
        get_metric(
            baseline_metrics,
            "relative_l2",
            split="test",
            default=get_metric(
                baseline_metrics,
                "test_relative_l2",
            ),
        )
    )

    pedvino_test_l2 = float(
        get_metric(
            pedvino_metrics,
            "relative_l2",
            split="test",
            default=get_metric(
                pedvino_metrics,
                "test_relative_l2",
            ),
        )
    )

    models = [
        "KNO",
        "PEDVINO",
    ]

    values = [
        baseline_test_l2,
        pedvino_test_l2,
    ]

    plt.figure(
        figsize=(7, 6)
    )

    bars = plt.bar(
        models,
        values,
    )

    plt.ylabel("Final Test Relative L2 Error")

    plt.title(
        "Burgers 1D: Final Test Error Comparison"
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, value in zip(
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
            )

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


# ============================================================
# FINAL METRIC EXTRACTION
# ============================================================

def extract_final_results(metrics):
    """
    Extract final experiment results into one normalized
    dictionary.
    """

    validation_l2 = get_metric(
        metrics,
        "relative_l2",
        split="validation",
        default=get_metric(
            metrics,
            "best_validation_relative_l2",
        ),
    )

    test_l2 = get_metric(
        metrics,
        "relative_l2",
        split="test",
        default=get_metric(
            metrics,
            "test_relative_l2",
        ),
    )

    test_mse = get_metric(
        metrics,
        "mse",
        split="test",
        default=get_metric(
            metrics,
            "test_mse",
        ),
    )

    train_l2 = get_metric(
        metrics,
        "relative_l2",
        split="train",
    )

    train_mse = get_metric(
        metrics,
        "mse",
        split="train",
    )

    return {
        "best_epoch":
            metrics.get(
                "best_epoch",
                np.nan,
            ),

        "validation_relative_l2":
            validation_l2,

        "test_relative_l2":
            test_l2,

        "test_mse":
            test_mse,

        "train_relative_l2":
            train_l2,

        "train_mse":
            train_mse,

        "trainable_parameters":
            metrics.get(
                "trainable_parameters",
                np.nan,
            ),

        "training_time_seconds":
            metrics.get(
                "training_time_seconds",
                np.nan,
            ),

        "training_time_minutes":
            metrics.get(
                "training_time_minutes",
                np.nan,
            ),
    }


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

def format_value(
    value,
    integer=False,
):
    """
    Format a metric safely.
    """

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"

    if not np.isfinite(numeric_value):
        return "N/A"

    if integer:
        return f"{int(numeric_value)}"

    return f"{numeric_value:.6e}"


def print_final_comparison(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Print the final experiment comparison.
    """

    baseline = extract_final_results(
        baseline_metrics
    )

    pedvino = extract_final_results(
        pedvino_metrics
    )

    print()
    print("=" * 78)
    print("BURGERS 1D: FINAL KNO vs PEDVINO COMPARISON")
    print("=" * 78)

    print(
        f"{'Metric':<35}"
        f"{'KNO':>20}"
        f"{'PEDVINO':>20}"
    )

    print("-" * 78)

    rows = [
        (
            "Best Epoch",
            baseline["best_epoch"],
            pedvino["best_epoch"],
            True,
        ),

        (
            "Validation Relative L2",
            baseline["validation_relative_l2"],
            pedvino["validation_relative_l2"],
            False,
        ),

        (
            "Test Relative L2",
            baseline["test_relative_l2"],
            pedvino["test_relative_l2"],
            False,
        ),

        (
            "Test MSE",
            baseline["test_mse"],
            pedvino["test_mse"],
            False,
        ),

        (
            "Trainable Parameters",
            baseline["trainable_parameters"],
            pedvino["trainable_parameters"],
            True,
        ),

        (
            "Training Time (seconds)",
            baseline["training_time_seconds"],
            pedvino["training_time_seconds"],
            False,
        ),

        (
            "Training Time (minutes)",
            baseline["training_time_minutes"],
            pedvino["training_time_minutes"],
            False,
        ),
    ]

    for (
        name,
        baseline_value,
        pedvino_value,
        is_integer,
    ) in rows:

        baseline_string = format_value(
            baseline_value,
            integer=is_integer,
        )

        pedvino_string = format_value(
            pedvino_value,
            integer=is_integer,
        )

        print(
            f"{name:<35}"
            f"{baseline_string:>20}"
            f"{pedvino_string:>20}"
        )

    print("-" * 78)

    # --------------------------------------------------------
    # Relative L2 improvement
    # --------------------------------------------------------

    try:

        baseline_test = float(
            baseline["test_relative_l2"]
        )

        pedvino_test = float(
            pedvino["test_relative_l2"]
        )

        if (
            np.isfinite(baseline_test)
            and np.isfinite(pedvino_test)
            and baseline_test > 0.0
        ):

            improvement = (
                (baseline_test - pedvino_test)
                / baseline_test
                * 100.0
            )

            if improvement > 0.0:

                print(
                    "PEDVINO Test Relative L2 "
                    f"Improvement: {improvement:.2f}%"
                )

            elif improvement < 0.0:

                print(
                    "PEDVINO Test Relative L2 "
                    f"Change: {improvement:.2f}% "
                    "(negative means worse than baseline)"
                )

            else:

                print(
                    "PEDVINO and baseline have identical "
                    "Test Relative L2."
                )

        else:

            print(
                "Test Relative L2 improvement could not "
                "be computed."
            )

    except (
        TypeError,
        ValueError,
    ):

        print(
            "Test Relative L2 improvement could not "
            "be computed."
        )

    print("=" * 78)


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    baseline_metrics,
    pedvino_metrics,
):
    """
    Create a machine-readable comparison summary.
    """

    baseline = extract_final_results(
        baseline_metrics
    )

    pedvino = extract_final_results(
        pedvino_metrics
    )

    baseline_test = baseline[
        "test_relative_l2"
    ]

    pedvino_test = pedvino[
        "test_relative_l2"
    ]

    improvement_percent = np.nan

    try:

        baseline_test_float = float(
            baseline_test
        )

        pedvino_test_float = float(
            pedvino_test
        )

        if (
            np.isfinite(baseline_test_float)
            and np.isfinite(pedvino_test_float)
            and baseline_test_float > 0.0
        ):

            improvement_percent = (
                (
                    baseline_test_float
                    - pedvino_test_float
                )
                / baseline_test_float
                * 100.0
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    def convert(value):

        if isinstance(
            value,
            (np.integer, np.floating),
        ):
            value = value.item()

        if isinstance(value, float):

            if not np.isfinite(value):
                return None

        return value

    return {
        "experiment":
            "Burgers 1D",

        "comparison":
            "KNO vs PEDVINO-KNO1d",

        "baseline": {
            key: convert(value)
            for key, value in baseline.items()
        },

        "pedvino": {
            key: convert(value)
            for key, value in pedvino.items()
        },

        "test_relative_l2_improvement_percent":
            convert(improvement_percent),

        "pedvino_better_than_baseline":
            (
                bool(improvement_percent > 0.0)
                if np.isfinite(improvement_percent)
                else None
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Burgers 1D KNO baseline and "
            "PEDVINO experiments."
        )
    )

    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=DEFAULT_BASELINE_DIR,
        help=(
            "Directory containing baseline "
            "history.json and metrics.json."
        ),
    )

    parser.add_argument(
        "--pedvino_dir",
        type=str,
        default=DEFAULT_PEDVINO_DIR,
        help=(
            "Directory containing PEDVINO "
            "history.json and metrics.json."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_COMPARISON_DIR,
        help=(
            "Directory where comparison plots "
            "and summary will be saved."
        ),
    )

    args = parser.parse_args()

    baseline_dir = args.baseline_dir
    pedvino_dir = args.pedvino_dir
    comparison_dir = args.output_dir

    os.makedirs(
        comparison_dir,
        exist_ok=True,
    )

    # ========================================================
    # PATHS
    # ========================================================

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

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    print("=" * 70)
    print("LOADING BURGERS 1D EXPERIMENT RESULTS")
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
    # VALIDATION RELATIVE L2
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
    # TEST RELATIVE L2
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
    # PEDVINO LOSS COMPONENTS
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
    # FINAL TEST RELATIVE L2
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
    # PRINT FINAL COMPARISON
    # ========================================================

    print_final_comparison(
        baseline_metrics=baseline_metrics,
        pedvino_metrics=pedvino_metrics,
    )

    # ========================================================
    # CREATE SUMMARY
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

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("BURGERS 1D COMPARISON COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

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
        "5. comparison_summary.json"
    )

    print(
        f"\nAll results saved in:\n{comparison_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
