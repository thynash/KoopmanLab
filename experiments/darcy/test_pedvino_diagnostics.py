import os
import sys
import copy
import math

import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# IMPORTS
# ============================================================

from koopmanlab.model_pedvino import PEDVINO

from experiments.darcy import config
from experiments.darcy.dataset import get_darcy_loaders

from experiments.darcy.train_utils import (
    set_seed,
    get_device,
)


# ============================================================
# BASIC SETTINGS
# ============================================================

EPS = getattr(
    config,
    "LOSS_EPS",
    1e-8,
)

OVERFIT_STEPS = 100

PRINT_SAMPLES = 5


# ============================================================
# BUILD PEDVINO MODEL
# ============================================================

def build_pedvino_model():
    """
    Build the EXACT Darcy PEDVINO architecture.

    Important:
        coefficient a(x,y) -> solution u(x,y)

    The forcing is NOT concatenated to the neural operator
    input. It is supplied separately to the Darcy physics
    loss during training.

    This matches the one-channel KNO convention.
    """

    model = PEDVINO(
        backbone="KNO2d",

        t_len=config.T_LEN,

        operator_size=config.OPERATOR_SIZE,

        modes_x=config.MODES_X,
        modes_y=config.MODES_Y,

        decompose=config.DECOMPOSE,

        linear_type=config.LINEAR_TYPE,

        normalization=config.NORMALIZATION,

        hidden_size=config.PHYSICS_HIDDEN_SIZE,

        dx=config.DX,
        dy=config.DY,
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_best_model(
    model,
    device,
):

    checkpoint_path = config.PEDVINO_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "\nPEDVINO checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print(f"  {key}")

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

    else:

        raise KeyError(
            "Could not find model weights."
        )

    return checkpoint


# ============================================================
# UNPACK BATCH
# ============================================================

def unpack_batch(batch):
    """
    Current Darcy dataset returns:

        coefficient,
        solution,
        forcing

    A two-item fallback is retained only for compatibility.
    """

    if len(batch) == 3:

        coefficient, solution, forcing = batch

        return (
            coefficient,
            solution,
            forcing,
        )

    if len(batch) == 2:

        coefficient, solution = batch

        return (
            coefficient,
            solution,
            None,
        )

    raise ValueError(
        f"Unexpected batch structure with "
        f"{len(batch)} tensors."
    )


# ============================================================
# MODEL OUTPUT
# ============================================================

def forward_model(
    model,
    coefficient,
):
    """
    Handle either:

        prediction

    or:

        prediction, reconstruction
    """

    output = model(coefficient)

    if isinstance(output, tuple):

        prediction = output[0]

        if len(output) > 1:
            reconstruction = output[1]
        else:
            reconstruction = None

    else:

        prediction = output
        reconstruction = None

    return prediction, reconstruction


# ============================================================
# RELATIVE L2
# ============================================================

def relative_l2_per_sample(
    prediction,
    target,
    eps=EPS,
):

    batch_size = prediction.shape[0]

    prediction_flat = prediction.reshape(
        batch_size,
        -1,
    )

    target_flat = target.reshape(
        batch_size,
        -1,
    )

    numerator = torch.linalg.vector_norm(
        prediction_flat - target_flat,
        dim=1,
    )

    denominator = torch.linalg.vector_norm(
        target_flat,
        dim=1,
    ).clamp_min(eps)

    return numerator / denominator


# ============================================================
# TENSOR STATISTICS
# ============================================================

def tensor_statistics(tensor):

    tensor = tensor.detach().float()

    return {
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "absmax": tensor.abs().max().item(),
        "norm": torch.linalg.vector_norm(
            tensor
        ).item(),
        "rms": torch.sqrt(
            torch.mean(tensor.pow(2))
        ).item(),
    }


def print_statistics(
    name,
    tensor,
):

    stats = tensor_statistics(tensor)

    print(f"\n{name}")
    print("-" * 70)

    print(
        f"Shape   : {tuple(tensor.shape)}"
    )

    print(
        f"Dtype   : {tensor.dtype}"
    )

    print(
        f"Device  : {tensor.device}"
    )

    print(
        f"Min     : {stats['min']:.8e}"
    )

    print(
        f"Max     : {stats['max']:.8e}"
    )

    print(
        f"Mean    : {stats['mean']:.8e}"
    )

    print(
        f"Std     : {stats['std']:.8e}"
    )

    print(
        f"RMS     : {stats['rms']:.8e}"
    )

    print(
        f"AbsMax  : {stats['absmax']:.8e}"
    )

    print(
        f"Norm    : {stats['norm']:.8e}"
    )

    return stats


# ============================================================
# CHECK NUMERICAL VALIDITY
# ============================================================

def check_tensor_validity(
    name,
    tensor,
):

    finite = torch.isfinite(tensor).all().item()

    num_nan = torch.isnan(
        tensor
    ).sum().item()

    num_inf = torch.isinf(
        tensor
    ).sum().item()

    print(
        f"{name:<30}"
        f"finite={str(finite):<6} "
        f"NaN={num_nan:<8} "
        f"Inf={num_inf:<8}"
    )

    return finite


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

@torch.no_grad()
def inspect_loader_distribution(
    data_loader,
    device,
    name,
):

    coefficient_values = []
    solution_values = []
    forcing_values = []

    coefficient_norms = []
    solution_norms = []
    forcing_norms = []

    for batch in data_loader:

        (
            coefficient,
            solution,
            forcing,
        ) = unpack_batch(batch)

        coefficient = coefficient.to(device)
        solution = solution.to(device)

        if forcing is not None:
            forcing = forcing.to(device)

        coefficient_values.append(
            coefficient.reshape(-1)
        )

        solution_values.append(
            solution.reshape(-1)
        )

        coefficient_norms.append(
            torch.linalg.vector_norm(
                coefficient.reshape(
                    coefficient.shape[0],
                    -1,
                ),
                dim=1,
            )
        )

        solution_norms.append(
            torch.linalg.vector_norm(
                solution.reshape(
                    solution.shape[0],
                    -1,
                ),
                dim=1,
            )
        )

        if forcing is not None:

            forcing_values.append(
                forcing.reshape(-1)
            )

            forcing_norms.append(
                torch.linalg.vector_norm(
                    forcing.reshape(
                        forcing.shape[0],
                        -1,
                    ),
                    dim=1,
                )
            )

    coefficient_values = torch.cat(
        coefficient_values
    )

    solution_values = torch.cat(
        solution_values
    )

    coefficient_norms = torch.cat(
        coefficient_norms
    )

    solution_norms = torch.cat(
        solution_norms
    )

    print("\n")
    print("=" * 70)
    print(
        f"{name.upper()} DATA DISTRIBUTION"
    )
    print("=" * 70)

    print_statistics(
        "Coefficient a",
        coefficient_values,
    )

    print_statistics(
        "Solution u",
        solution_values,
    )

    print("\nPer-sample coefficient norm")
    print(
        f"Mean: {coefficient_norms.mean().item():.8e}"
    )
    print(
        f"Std : {coefficient_norms.std().item():.8e}"
    )
    print(
        f"Min : {coefficient_norms.min().item():.8e}"
    )
    print(
        f"Max : {coefficient_norms.max().item():.8e}"
    )

    print("\nPer-sample solution norm")
    print(
        f"Mean: {solution_norms.mean().item():.8e}"
    )
    print(
        f"Std : {solution_norms.std().item():.8e}"
    )
    print(
        f"Min : {solution_norms.min().item():.8e}"
    )
    print(
        f"Max : {solution_norms.max().item():.8e}"
    )

    result = {
        "coefficient_mean":
            coefficient_values.mean().item(),

        "coefficient_std":
            coefficient_values.std().item(),

        "solution_mean":
            solution_values.mean().item(),

        "solution_std":
            solution_values.std().item(),

        "solution_norm_mean":
            solution_norms.mean().item(),

        "solution_norm_std":
            solution_norms.std().item(),
    }

    if len(forcing_values) > 0:

        forcing_values = torch.cat(
            forcing_values
        )

        forcing_norms = torch.cat(
            forcing_norms
        )

        print_statistics(
            "Forcing f",
            forcing_values,
        )

        print("\nPer-sample forcing norm")
        print(
            f"Mean: {forcing_norms.mean().item():.8e}"
        )
        print(
            f"Std : {forcing_norms.std().item():.8e}"
        )
        print(
            f"Min : {forcing_norms.min().item():.8e}"
        )
        print(
            f"Max : {forcing_norms.max().item():.8e}"
        )

        result.update(
            {
                "forcing_mean":
                    forcing_values.mean().item(),

                "forcing_std":
                    forcing_values.std().item(),

                "forcing_norm_mean":
                    forcing_norms.mean().item(),

                "forcing_norm_std":
                    forcing_norms.std().item(),
            }
        )

    return result


# ============================================================
# DIAGNOSE ONE SPLIT
# ============================================================

@torch.no_grad()
def diagnose_split(
    model,
    data_loader,
    device,
    split_name,
):

    model.eval()

    total_mse = 0.0
    total_relative_l2 = 0.0
    total_samples = 0

    prediction_norms = []
    target_norms = []
    error_norms = []
    relative_errors = []

    first_batch = None

    for batch_index, batch in enumerate(data_loader):

        (
            coefficient,
            solution,
            forcing,
        ) = unpack_batch(batch)

        coefficient = coefficient.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        if forcing is not None:

            forcing = forcing.to(
                device,
                non_blocking=True,
            )

        prediction, reconstruction = forward_model(
            model=model,
            coefficient=coefficient,
        )

        batch_size = solution.shape[0]

        mse = F.mse_loss(
            prediction,
            solution,
        )

        relative_errors_batch = (
            relative_l2_per_sample(
                prediction,
                solution,
            )
        )

        relative_l2_mean = (
            relative_errors_batch.mean()
        )

        total_mse += (
            mse.item() * batch_size
        )

        total_relative_l2 += (
            relative_l2_mean.item()
            * batch_size
        )

        total_samples += batch_size

        prediction_norm_batch = (
            torch.linalg.vector_norm(
                prediction.reshape(
                    batch_size,
                    -1,
                ),
                dim=1,
            )
        )

        target_norm_batch = (
            torch.linalg.vector_norm(
                solution.reshape(
                    batch_size,
                    -1,
                ),
                dim=1,
            )
        )

        error_norm_batch = (
            torch.linalg.vector_norm(
                (
                    prediction - solution
                ).reshape(
                    batch_size,
                    -1,
                ),
                dim=1,
            )
        )

        prediction_norms.append(
            prediction_norm_batch.cpu()
        )

        target_norms.append(
            target_norm_batch.cpu()
        )

        error_norms.append(
            error_norm_batch.cpu()
        )

        relative_errors.append(
            relative_errors_batch.cpu()
        )

        if batch_index == 0:

            first_batch = {
                "coefficient":
                    coefficient.detach().clone(),

                "solution":
                    solution.detach().clone(),

                "forcing":
                    (
                        forcing.detach().clone()
                        if forcing is not None
                        else None
                    ),

                "prediction":
                    prediction.detach().clone(),

                "reconstruction":
                    (
                        reconstruction.detach().clone()
                        if reconstruction is not None
                        else None
                    ),
            }

    prediction_norms = torch.cat(
        prediction_norms
    )

    target_norms = torch.cat(
        target_norms
    )

    error_norms = torch.cat(
        error_norms
    )

    relative_errors = torch.cat(
        relative_errors
    )

    results = {
        "mse":
            total_mse / total_samples,

        "relative_l2":
            total_relative_l2 / total_samples,

        "prediction_norm_mean":
            prediction_norms.mean().item(),

        "prediction_norm_std":
            prediction_norms.std().item(),

        "target_norm_mean":
            target_norms.mean().item(),

        "target_norm_std":
            target_norms.std().item(),

        "error_norm_mean":
            error_norms.mean().item(),

        "relative_l2_min":
            relative_errors.min().item(),

        "relative_l2_max":
            relative_errors.max().item(),

        "relative_l2_median":
            relative_errors.median().item(),

        "relative_l2_std":
            relative_errors.std().item(),

        "first_batch":
            first_batch,
    }

    print("\n")
    print("=" * 70)
    print(
        f"{split_name.upper()} MODEL DIAGNOSTICS"
    )
    print("=" * 70)

    print(
        f"MSE                    : "
        f"{results['mse']:.8e}"
    )

    print(
        f"Mean Relative L2       : "
        f"{results['relative_l2']:.8e}"
    )

    print(
        f"Median Relative L2     : "
        f"{results['relative_l2_median']:.8e}"
    )

    print(
        f"Relative L2 Std        : "
        f"{results['relative_l2_std']:.8e}"
    )

    print(
        f"Best Sample Rel L2     : "
        f"{results['relative_l2_min']:.8e}"
    )

    print(
        f"Worst Sample Rel L2    : "
        f"{results['relative_l2_max']:.8e}"
    )

    print(
        f"Mean Target Norm       : "
        f"{results['target_norm_mean']:.8e}"
    )

    print(
        f"Mean Prediction Norm   : "
        f"{results['prediction_norm_mean']:.8e}"
    )

    print(
        f"Mean Error Norm        : "
        f"{results['error_norm_mean']:.8e}"
    )

    ratio = (
        results["prediction_norm_mean"]
        /
        max(
            results["target_norm_mean"],
            EPS,
        )
    )

    print(
        f"Pred/Target Norm Ratio : "
        f"{ratio:.8e}"
    )

    return results


# ============================================================
# FIRST BATCH OUTPUT INSPECTION
# ============================================================

def inspect_first_batch(
    results,
    split_name,
):

    batch = results["first_batch"]

    print("\n")
    print("=" * 70)
    print(
        f"{split_name.upper()} FIRST BATCH DETAIL"
    )
    print("=" * 70)

    print_statistics(
        "Coefficient",
        batch["coefficient"],
    )

    print_statistics(
        "True Solution",
        batch["solution"],
    )

    print_statistics(
        "Prediction",
        batch["prediction"],
    )

    if batch["forcing"] is not None:

        print_statistics(
            "Forcing",
            batch["forcing"],
        )

    if batch["reconstruction"] is not None:

        print_statistics(
            "Reconstruction",
            batch["reconstruction"],
        )

        reconstruction_mse = F.mse_loss(
            batch["reconstruction"],
            batch["coefficient"],
        )

        print(
            f"\nReconstruction MSE: "
            f"{reconstruction_mse.item():.8e}"
        )

    error = (
        batch["prediction"]
        - batch["solution"]
    )

    print_statistics(
        "Prediction Error",
        error,
    )

    relative_errors = relative_l2_per_sample(
        batch["prediction"],
        batch["solution"],
    )

    print(
        f"\nFirst {min(PRINT_SAMPLES, len(relative_errors))} "
        f"sample Relative L2 errors:"
    )

    for index in range(
        min(
            PRINT_SAMPLES,
            len(relative_errors),
        )
    ):

        print(
            f"  Sample {index:02d}: "
            f"{relative_errors[index].item():.8e}"
        )


# ============================================================
# TRAIN MODE vs EVAL MODE
# ============================================================

@torch.no_grad()
def diagnose_train_eval_difference(
    model,
    data_loader,
    device,
):

    batch = next(iter(data_loader))

    (
        coefficient,
        solution,
        forcing,
    ) = unpack_batch(batch)

    coefficient = coefficient.to(device)
    solution = solution.to(device)

    # --------------------------------------------------------
    # EVAL
    # --------------------------------------------------------

    model.eval()

    prediction_eval, reconstruction_eval = (
        forward_model(
            model,
            coefficient,
        )
    )

    eval_mse = F.mse_loss(
        prediction_eval,
        solution,
    )

    eval_l2 = relative_l2_per_sample(
        prediction_eval,
        solution,
    ).mean()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    prediction_train, reconstruction_train = (
        forward_model(
            model,
            coefficient,
        )
    )

    train_mse = F.mse_loss(
        prediction_train,
        solution,
    )

    train_l2 = relative_l2_per_sample(
        prediction_train,
        solution,
    ).mean()

    prediction_difference = F.mse_loss(
        prediction_train,
        prediction_eval,
    )

    print("\n")
    print("=" * 70)
    print(
        "TRAIN MODE vs EVAL MODE"
    )
    print("=" * 70)

    print(
        f"Train-mode MSE        : "
        f"{train_mse.item():.8e}"
    )

    print(
        f"Train-mode Relative L2: "
        f"{train_l2.item():.8e}"
    )

    print(
        f"Eval-mode MSE         : "
        f"{eval_mse.item():.8e}"
    )

    print(
        f"Eval-mode Relative L2 : "
        f"{eval_l2.item():.8e}"
    )

    print(
        f"Prediction Difference : "
        f"{prediction_difference.item():.8e}"
    )

    if prediction_difference.item() > 1e-6:

        print(
            "\nWARNING: train/eval outputs differ."
        )

        print(
            "This can indicate mode-dependent layers "
            "such as BatchNorm or Dropout."
        )

    else:

        print(
            "\nPASS: train/eval outputs are effectively "
            "identical."
        )


# ============================================================
# CHECK MODEL INPUT SENSITIVITY
# ============================================================

@torch.no_grad()
def diagnose_input_sensitivity(
    model,
    data_loader,
    device,
):

    batch = next(iter(data_loader))

    (
        coefficient,
        solution,
        forcing,
    ) = unpack_batch(batch)

    coefficient = coefficient.to(device)

    model.eval()

    prediction, _ = forward_model(
        model,
        coefficient,
    )

    shuffled = coefficient[
        torch.randperm(
            coefficient.shape[0],
            device=device,
        )
    ]

    shuffled_prediction, _ = forward_model(
        model,
        shuffled,
    )

    zero_input = torch.zeros_like(
        coefficient
    )

    zero_prediction, _ = forward_model(
        model,
        zero_input,
    )

    shuffled_difference = F.mse_loss(
        prediction,
        shuffled_prediction,
    )

    zero_difference = F.mse_loss(
        prediction,
        zero_prediction,
    )

    print("\n")
    print("=" * 70)
    print(
        "INPUT SENSITIVITY DIAGNOSTIC"
    )
    print("=" * 70)

    print(
        f"Prediction MSE after shuffling "
        f"coefficients: "
        f"{shuffled_difference.item():.8e}"
    )

    print(
        f"Prediction MSE between actual and "
        f"zero input: "
        f"{zero_difference.item():.8e}"
    )

    if (
        shuffled_difference.item() < 1e-10
        and zero_difference.item() < 1e-10
    ):

        print(
            "\nWARNING: model appears almost insensitive "
            "to the Darcy coefficient input."
        )

    else:

        print(
            "\nModel output changes when the coefficient "
            "input changes."
        )


# ============================================================
# CHECKPOINT METRIC CONSISTENCY
# ============================================================

def diagnose_checkpoint_consistency(
    checkpoint,
    train_results,
    val_results,
    test_results,
):

    print("\n")
    print("=" * 70)
    print(
        "CHECKPOINT CONSISTENCY"
    )
    print("=" * 70)

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "metrics" not in checkpoint:

        print(
            "No metrics stored in checkpoint."
        )

        return

    metrics = checkpoint["metrics"]

    for split_name, current_results in [
        ("train", train_results),
        ("validation", val_results),
        ("test", test_results),
    ]:

        if split_name not in metrics:
            continue

        stored = metrics[split_name]

        print(
            f"\nStored {split_name} metrics:"
        )

        print(stored)

        if "relative_l2" in stored:

            difference = abs(
                stored["relative_l2"]
                - current_results["relative_l2"]
            )

            print(
                f"Current Relative L2 : "
                f"{current_results['relative_l2']:.8e}"
            )

            print(
                f"Absolute difference : "
                f"{difference:.8e}"
            )


# ============================================================
# GRADIENT / PARAMETER CHECK
# ============================================================

def diagnose_backward(
    model,
    data_loader,
    device,
):

    batch = next(iter(data_loader))

    (
        coefficient,
        solution,
        forcing,
    ) = unpack_batch(batch)

    coefficient = coefficient.to(device)
    solution = solution.to(device)

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    prediction, reconstruction = forward_model(
        model,
        coefficient,
    )

    prediction_loss = F.mse_loss(
        prediction,
        solution,
    )

    prediction_loss.backward()

    total_grad_sq = 0.0

    parameters_with_grad = 0

    parameters_without_grad = 0

    nan_gradients = 0

    inf_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is None:

            parameters_without_grad += 1
            continue

        parameters_with_grad += 1

        gradient = parameter.grad.detach()

        if torch.isnan(gradient).any():
            nan_gradients += 1

        if torch.isinf(gradient).any():
            inf_gradients += 1

        total_grad_sq += (
            gradient.norm(2).item() ** 2
        )

    total_grad_norm = math.sqrt(
        total_grad_sq
    )

    print("\n")
    print("=" * 70)
    print(
        "BACKWARD / GRADIENT DIAGNOSTIC"
    )
    print("=" * 70)

    print(
        f"Prediction loss          : "
        f"{prediction_loss.item():.8e}"
    )

    print(
        f"Parameters with gradient : "
        f"{parameters_with_grad}"
    )

    print(
        f"Parameters without grad  : "
        f"{parameters_without_grad}"
    )

    print(
        f"Global gradient norm     : "
        f"{total_grad_norm:.8e}"
    )

    print(
        f"NaN gradient tensors     : "
        f"{nan_gradients}"
    )

    print(
        f"Inf gradient tensors     : "
        f"{inf_gradients}"
    )

    model.zero_grad(
        set_to_none=True
    )


# ============================================================
# SINGLE-BATCH OVERFIT TEST
# ============================================================

def single_batch_overfit_test(
    device,
    train_loader,
):

    print("\n")
    print("=" * 70)
    print(
        "SINGLE-BATCH OVERFIT TEST"
    )
    print("=" * 70)

    print(
        "This uses prediction MSE ONLY."
    )

    print(
        "It does NOT modify the loaded checkpoint model."
    )

    batch = next(iter(train_loader))

    (
        coefficient,
        solution,
        forcing,
    ) = unpack_batch(batch)

    coefficient = coefficient.to(device)
    solution = solution.to(device)

    model = build_pedvino_model().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.0,
    )

    model.train()

    initial_loss = None

    for step in range(
        1,
        OVERFIT_STEPS + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction, _ = forward_model(
            model,
            coefficient,
        )

        loss = F.mse_loss(
            prediction,
            solution,
        )

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()

        optimizer.step()

        if (
            step == 1
            or step % 20 == 0
            or step == OVERFIT_STEPS
        ):

            with torch.no_grad():

                prediction_eval, _ = forward_model(
                    model,
                    coefficient,
                )

                rel_l2 = relative_l2_per_sample(
                    prediction_eval,
                    solution,
                ).mean()

            print(
                f"Step [{step:03d}/{OVERFIT_STEPS}] | "
                f"MSE: {loss.item():.8e} | "
                f"Rel L2: {rel_l2.item():.8e}"
            )

    final_loss = loss.item()

    reduction = (
        initial_loss
        /
        max(final_loss, EPS)
    )

    print("\nOverfit loss reduction factor:")
    print(
        f"{reduction:.4e}x"
    )

    if reduction < 10.0:

        print(
            "\nWARNING: model cannot substantially fit "
            "even one batch using prediction loss alone."
        )

        print(
            "This suggests an architecture, input, "
            "output, or optimization problem."
        )

    else:

        print(
            "\nPASS: the architecture can fit one batch."
        )

        print(
            "If full-dataset validation still fails, the "
            "problem is more likely generalization, "
            "physics-loss scaling, or training dynamics."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    train_data,
    val_data,
    test_data,
    train_model,
    val_model,
    test_model,
):

    print("\n")
    print("=" * 92)
    print(
        "FINAL PEDVINO DARCY DIAGNOSTIC SUMMARY"
    )
    print("=" * 92)

    print(
        f"{'Split':<14}"
        f"{'Coeff Mean':>14}"
        f"{'Sol Mean':>14}"
        f"{'Sol Std':>14}"
        f"{'MSE':>14}"
        f"{'Rel L2':>14}"
        f"{'Pred/True':>14}"
    )

    print("-" * 98)

    rows = [
        (
            "Train",
            train_data,
            train_model,
        ),
        (
            "Validation",
            val_data,
            val_model,
        ),
        (
            "Test",
            test_data,
            test_model,
        ),
    ]

    for name, data_stats, model_stats in rows:

        norm_ratio = (
            model_stats["prediction_norm_mean"]
            /
            max(
                model_stats["target_norm_mean"],
                EPS,
            )
        )

        print(
            f"{name:<14}"
            f"{data_stats['coefficient_mean']:>14.4e}"
            f"{data_stats['solution_mean']:>14.4e}"
            f"{data_stats['solution_std']:>14.4e}"
            f"{model_stats['mse']:>14.4e}"
            f"{model_stats['relative_l2']:>14.4e}"
            f"{norm_ratio:>14.4e}"
        )

    print("=" * 92)

    train_val_gap = (
        val_model["relative_l2"]
        /
        max(
            train_model["relative_l2"],
            EPS,
        )
    )

    train_test_gap = (
        test_model["relative_l2"]
        /
        max(
            train_model["relative_l2"],
            EPS,
        )
    )

    print(
        f"\nValidation / Train Relative L2 ratio: "
        f"{train_val_gap:.8e}"
    )

    print(
        f"Test / Train Relative L2 ratio      : "
        f"{train_test_gap:.8e}"
    )

    print("\nINTERPRETATION:")

    if train_val_gap > 5.0:

        print(
            "- Large train/validation gap detected."
        )

        print(
            "  Investigate overfitting, split mismatch, "
            "or training-only behavior."
        )

    else:

        print(
            "- No extreme train/validation Relative L2 "
            "gap is present."
        )

    val_norm_ratio = (
        val_model["prediction_norm_mean"]
        /
        max(
            val_model["target_norm_mean"],
            EPS,
        )
    )

    if val_norm_ratio > 2.0:

        print(
            "- Validation prediction norm is much larger "
            "than the target norm."
        )

        print(
            "  This indicates output-scale explosion."
        )

    elif val_norm_ratio < 0.5:

        print(
            "- Validation prediction norm is much smaller "
            "than the target norm."
        )

        print(
            "  This indicates output collapse or severe "
            "under-prediction."
        )

    else:

        print(
            "- Prediction and target norms are on a "
            "similar scale."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DARCY FLOW - PEDVINO EXTENDED DIAGNOSTICS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    set_seed(
        config.SEED
    )

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Seed: {config.SEED}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_darcy_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
    )

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Train samples      : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # BATCH STRUCTURE
    # --------------------------------------------------------

    batch = next(
        iter(train_loader)
    )

    print("\nBATCH STRUCTURE")
    print("-" * 70)

    print(
        f"Number of tensors returned: "
        f"{len(batch)}"
    )

    for index, tensor in enumerate(batch):

        print(
            f"Batch item {index}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"min={tensor.min().item():.6e}, "
            f"max={tensor.max().item():.6e}"
        )

    # --------------------------------------------------------
    # NUMERICAL VALIDITY
    # --------------------------------------------------------

    print("\nNUMERICAL VALIDITY CHECK")
    print("-" * 70)

    for index, tensor in enumerate(batch):

        check_tensor_validity(
            f"Batch item {index}",
            tensor,
        )

    # --------------------------------------------------------
    # DATA DISTRIBUTIONS
    # --------------------------------------------------------

    train_data = inspect_loader_distribution(
        train_loader,
        device,
        "Train",
    )

    val_data = inspect_loader_distribution(
        val_loader,
        device,
        "Validation",
    )

    test_data = inspect_loader_distribution(
        test_loader,
        device,
        "Test",
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "BUILDING PEDVINO"
    )
    print("=" * 70)

    model = build_pedvino_model().to(
        device
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = load_best_model(
        model=model,
        device=device,
    )

    if "epoch" in checkpoint:

        print(
            f"\nLoaded checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "metrics" in checkpoint:

        print(
            "\nStored checkpoint metrics:"
        )

        print(
            checkpoint["metrics"]
        )

    # --------------------------------------------------------
    # TRAIN/EVAL MODE
    # --------------------------------------------------------

    diagnose_train_eval_difference(
        model=model,
        data_loader=train_loader,
        device=device,
    )

    # --------------------------------------------------------
    # INPUT SENSITIVITY
    # --------------------------------------------------------

    diagnose_input_sensitivity(
        model=model,
        data_loader=train_loader,
        device=device,
    )

    # --------------------------------------------------------
    # GRADIENT CHECK
    # --------------------------------------------------------

    diagnose_backward(
        model=model,
        data_loader=train_loader,
        device=device,
    )

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    train_model = diagnose_split(
        model=model,
        data_loader=train_loader,
        device=device,
        split_name="Train",
    )

    val_model = diagnose_split(
        model=model,
        data_loader=val_loader,
        device=device,
        split_name="Validation",
    )

    test_model = diagnose_split(
        model=model,
        data_loader=test_loader,
        device=device,
        split_name="Test",
    )

    # --------------------------------------------------------
    # FIRST BATCH DETAILS
    # --------------------------------------------------------

    inspect_first_batch(
        train_model,
        "Train",
    )

    inspect_first_batch(
        val_model,
        "Validation",
    )

    inspect_first_batch(
        test_model,
        "Test",
    )

    # --------------------------------------------------------
    # CHECKPOINT CONSISTENCY
    # --------------------------------------------------------

    diagnose_checkpoint_consistency(
        checkpoint=checkpoint,
        train_results=train_model,
        val_results=val_model,
        test_results=test_model,
    )

    # --------------------------------------------------------
    # SINGLE BATCH FITTING TEST
    # --------------------------------------------------------

    single_batch_overfit_test(
        device=device,
        train_loader=train_loader,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_model=train_model,
        val_model=val_model,
        test_model=test_model,
    )

    print("\n")
    print("=" * 70)
    print(
        "DIAGNOSTICS COMPLETED"
    )
    print("=" * 70)

    print(
        "\nRun this before modifying train_pedvino.py."
    )

    print(
        "\nThe most important outputs to send me are:"
    )

    print(
        "1. Train / Validation / Test data distributions"
    )

    print(
        "2. Input sensitivity diagnostic"
    )

    print(
        "3. Train/Eval mode difference"
    )

    print(
        "4. Train / Validation / Test Relative L2 summary"
    )

    print(
        "5. Checkpoint consistency"
    )

    print(
        "6. Single-batch overfit test"
    )

    print(
        "\nThese will tell us whether the failure is caused "
        "by the model architecture, data scaling, input "
        "handling, checkpoint selection, or PEDVINO training "
        "dynamics."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
