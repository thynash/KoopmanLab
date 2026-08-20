"""
Training utilities for the Burgers 1D experiments.

Shared by:
    - train_baseline.py
    - train_pedvino.py
"""

import json
import os
import random
import time

import numpy as np
import torch


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour for reproducible experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Return the preferred training device.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# RELATIVE L2 ERROR
# ============================================================

def relative_l2_error(
    prediction,
    target,
    eps=1e-8,
):
    """
    Compute the mean relative L2 error over a batch.

        ||prediction - target||_2
        ------------------------
              ||target||_2

    Expected shape:

        [B, N, C]

    but works for arbitrary tensors whose first dimension
    is the batch dimension.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have identical shapes. "
            f"Got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )

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
        ord=2,
        dim=1,
    )

    denominator = torch.linalg.vector_norm(
        target_flat,
        ord=2,
        dim=1,
    ).clamp_min(eps)

    return (
        numerator / denominator
    ).mean()


# ============================================================
# MEAN SQUARED ERROR
# ============================================================

def mean_squared_error(
    prediction,
    target,
):
    """
    Standard mean squared error.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have identical shapes."
        )

    return torch.mean(
        (prediction - target).pow(2)
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
    Evaluate a model on a DataLoader.

    The Burgers dataset returns:

        initial_state : [B, N, 1]
        solution      : [B, N, 1]

    Returns:
        {
            "relative_l2": float,
            "mse": float,
            "num_samples": int,
        }
    """

    model.eval()

    total_relative_l2 = 0.0
    total_mse = 0.0
    total_samples = 0

    for initial_state, solution in data_loader:

        initial_state = initial_state.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # MODEL FORWARD
        #
        # Both the baseline KNO1d and PEDVINO models are
        # expected to return either:
        #
        #   prediction
        #
        # or:
        #
        #   prediction, reconstruction
        # ----------------------------------------------------

        output = model(initial_state)

        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output

        batch_size = initial_state.shape[0]

        batch_relative_l2 = relative_l2_error(
            prediction,
            solution,
        )

        batch_mse = mean_squared_error(
            prediction,
            solution,
        )

        total_relative_l2 += (
            batch_relative_l2.item()
            * batch_size
        )

        total_mse += (
            batch_mse.item()
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation DataLoader contains no samples."
        )

    return {
        "relative_l2": (
            total_relative_l2
            / total_samples
        ),

        "mse": (
            total_mse
            / total_samples
        ),

        "num_samples": total_samples,
    }


# ============================================================
# HISTORY
# ============================================================

def initialize_history():
    """
    Initialize the standard experiment history dictionary.
    """

    return {
        "epoch": [],

        "train_loss": [],

        "train_prediction_loss": [],

        "train_reconstruction_loss": [],

        "train_relative_l2": [],

        "val_relative_l2": [],

        "val_mse": [],

        "learning_rate": [],
    }


def append_history(
    history,
    epoch,
    train_loss,
    train_prediction_loss,
    train_reconstruction_loss,
    train_relative_l2,
    val_metrics,
    learning_rate,
):
    """
    Append one epoch to the experiment history.

    PEDVINO-specific losses such as energy, gradient and
    boundary loss can be added separately by train_pedvino.py.
    """

    history["epoch"].append(
        int(epoch)
    )

    history["train_loss"].append(
        float(train_loss)
    )

    history["train_prediction_loss"].append(
        float(train_prediction_loss)
    )

    history["train_reconstruction_loss"].append(
        float(train_reconstruction_loss)
    )

    history["train_relative_l2"].append(
        float(train_relative_l2)
    )

    history["val_relative_l2"].append(
        float(val_metrics["relative_l2"])
    )

    history["val_mse"].append(
        float(val_metrics["mse"])
    )

    history["learning_rate"].append(
        float(learning_rate)
    )


# ============================================================
# JSON SAVING
# ============================================================

def save_json(
    data,
    path,
):
    """
    Save a dictionary as formatted JSON.
    """

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
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
        )


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metrics=None,
    extra=None,
):
    """
    Save a complete training checkpoint.
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
        "model_state_dict":
            model.state_dict(),
    }

    if optimizer is not None:
        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = int(epoch)

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """
    Load a training checkpoint.

    Returns the full checkpoint dictionary.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Stop training when a monitored metric has not improved
    for a specified number of epochs.

    Lower metric values are considered better.
    """

    def __init__(
        self,
        patience,
        min_delta=0.0,
    ):

        if patience <= 0:
            raise ValueError(
                "patience must be positive."
            )

        self.patience = int(
            patience
        )

        self.min_delta = float(
            min_delta
        )

        self.best_value = float("inf")

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        value,
    ):
        """
        Update early stopping state.

        Returns:
            improved : bool
        """

        value = float(value)

        improved = (
            value
            < self.best_value
            - self.min_delta
        )

        if improved:

            self.best_value = value

            self.counter = 0

        else:

            self.counter += 1

            if (
                self.counter
                >= self.patience
            ):
                self.should_stop = True

        return improved


# ============================================================
# EXPERIMENT TIMER
# ============================================================

class ExperimentTimer:
    """
    Track elapsed experiment time.
    """

    def __init__(self):

        self.start_time = None

        self.end_time = None

    def start(self):

        self.start_time = time.time()

        self.end_time = None

    def stop(self):

        if self.start_time is None:
            raise RuntimeError(
                "Timer has not been started."
            )

        self.end_time = time.time()

    @property
    def elapsed_seconds(self):

        if self.start_time is None:
            return 0.0

        if self.end_time is None:
            return (
                time.time()
                - self.start_time
            )

        return (
            self.end_time
            - self.start_time
        )

    @property
    def elapsed_minutes(self):

        return (
            self.elapsed_seconds
            / 60.0
        )


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(
    optimizer,
):
    """
    Return the current learning rate.
    """

    if len(
        optimizer.param_groups
    ) == 0:
        raise RuntimeError(
            "Optimizer has no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )
