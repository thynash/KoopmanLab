import os
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds for reproducible experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Return the preferred training device.
    """

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# BATCH UNPACKING
# ============================================================

def unpack_batch(batch):
    """
    Robustly unpack PDE dataset batches.

    Supported formats
    -----------------

    2-item dataset:
        input_field, solution

    3-item dataset:
        input_field, solution, forcing

    Returns
    -------

    input_field
        Input PDE field.

    solution
        Target solution.

    forcing
        PDE forcing if available, otherwise None.

    Notes
    -----
    Darcy dataset currently returns:

        coefficient, solution, forcing

    This helper keeps train/evaluation code compatible with
    both baseline and physics-informed experiments.
    """

    if not isinstance(
        batch,
        (tuple, list),
    ):
        raise TypeError(
            "Expected batch to be a tuple or list, "
            f"but received {type(batch)}."
        )

    if len(batch) == 2:

        input_field, solution = batch

        forcing = None

    elif len(batch) == 3:

        input_field, solution, forcing = batch

    else:

        raise ValueError(
            "Unsupported batch format. Expected either "
            "(input_field, solution) or "
            "(input_field, solution, forcing), "
            f"but received {len(batch)} items."
        )

    return (
        input_field,
        solution,
        forcing,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
):
    """
    Evaluate a neural operator.

    Compatible with both:

        (input, solution)

    and:

        (input, solution, forcing)

    Darcy dataset:

        coefficient, solution, forcing

    The forcing field is not needed for standard prediction
    evaluation and is therefore ignored here.

    Metrics
    -------

    mse:
        Mean squared error over all samples.

    relative_l2:
        Global relative L2 error:

            ||prediction - target||_2
            ------------------------
                  ||target||_2
    """

    model.eval()

    total_squared_error = 0.0

    total_target_squared = 0.0

    total_mse_sum = 0.0

    total_samples = 0

    for batch in data_loader:

        # ----------------------------------------------------
        # Robustly unpack either 2 or 3 tensors.
        # ----------------------------------------------------

        input_field, solution, _ = unpack_batch(
            batch
        )

        # ----------------------------------------------------
        # Device transfer
        # ----------------------------------------------------

        input_field = input_field.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward pass
        #
        # KNO returns:
        #
        # prediction, reconstruction
        # ----------------------------------------------------

        output = model(
            input_field
        )

        # Support models returning either a tensor or
        # (prediction, reconstruction).
        if isinstance(
            output,
            (tuple, list),
        ):

            prediction = output[0]

        else:

            prediction = output

        # ----------------------------------------------------
        # Batch size
        # ----------------------------------------------------

        batch_size = input_field.shape[0]

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        mse = F.mse_loss(
            prediction,
            solution,
            reduction="mean",
        )

        # ----------------------------------------------------
        # Relative L2 components
        # ----------------------------------------------------

        squared_error = torch.sum(
            (prediction - solution).pow(2)
        )

        target_squared = torch.sum(
            solution.pow(2)
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        total_mse_sum += (
            mse.item()
            * batch_size
        )

        total_squared_error += (
            squared_error.item()
        )

        total_target_squared += (
            target_squared.item()
        )

        total_samples += batch_size

    # ========================================================
    # Safety
    # ========================================================

    if total_samples == 0:

        raise RuntimeError(
            "Evaluation loader produced zero samples."
        )

    # ========================================================
    # Final metrics
    # ========================================================

    mse = (
        total_mse_sum
        / total_samples
    )

    relative_l2 = (
        total_squared_error
        / (
            total_target_squared
            + 1e-12
        )
    ) ** 0.5

    return {

        "mse":
            float(mse),

        "relative_l2":
            float(relative_l2),
    }


# ============================================================
# HISTORY INITIALIZATION
# ============================================================

def initialize_history():
    """
    Create a standard experiment history dictionary.

    The structure is intentionally flexible so that both
    baseline KNO and PEDVINO can use the same utility.
    """

    return {

        "epoch": [],

        "train": {
            "total_loss": [],
            "prediction_loss": [],
            "reconstruction_loss": [],
        },

        "validation": {
            "mse": [],
            "relative_l2": [],
        },

        "test": {
            "mse": [],
            "relative_l2": [],
        },
    }


# ============================================================
# APPEND HISTORY
# ============================================================

def append_history(
    history,
    epoch,
    train_metrics,
    val_metrics,
    test_metrics,
):
    """
    Append one epoch of metrics.

    Additional PEDVINO losses are automatically created if
    they are present in train_metrics.
    """

    history["epoch"].append(
        int(epoch)
    )

    # --------------------------------------------------------
    # Training metrics
    # --------------------------------------------------------

    for key, value in train_metrics.items():

        if key not in history["train"]:

            history["train"][key] = []

        history["train"][key].append(
            float(value)
        )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    for key, value in val_metrics.items():

        if key not in history["validation"]:

            history["validation"][key] = []

        history["validation"][key].append(
            float(value)
        )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    for key, value in test_metrics.items():

        if key not in history["test"]:

            history["test"][key] = []

        history["test"][key].append(
            float(value)
        )


# ============================================================
# JSON SAVING
# ============================================================

def save_json(
    data,
    path,
):
    """
    Save experiment data as formatted JSON.
    """

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    def convert_to_serializable(value):

        if isinstance(
            value,
            torch.Tensor,
        ):

            if value.numel() == 1:

                return value.item()

            return value.detach().cpu().tolist()

        if isinstance(
            value,
            np.ndarray,
        ):

            return value.tolist()

        if isinstance(
            value,
            np.integer,
        ):

            return int(value)

        if isinstance(
            value,
            np.floating,
        ):

            return float(value)

        if isinstance(
            value,
            np.bool_,
        ):

            return bool(value)

        raise TypeError(
            f"Object of type "
            f"{type(value).__name__} "
            f"is not JSON serializable."
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=convert_to_serializable,
        )


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    path,
    scheduler=None,
):
    """
    Save a complete training checkpoint.

    Stores:

        - model state
        - optimizer state
        - scheduler state if available
        - epoch
        - metrics
    """

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    checkpoint = {

        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics,
    }

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# EXPERIMENT TIMER
# ============================================================

class ExperimentTimer:
    """
    Simple experiment timer.
    """

    def __init__(self):

        self.start_time = None

        self.end_time = None

    def start(self):

        self.start_time = time.time()

        self.end_time = None

    def stop(self):

        self.end_time = time.time()

        return self.elapsed()

    def elapsed(self):

        if self.start_time is None:

            return 0.0

        if self.end_time is None:

            current_time = time.time()

        else:

            current_time = self.end_time

        return float(
            current_time
            - self.start_time
        )
