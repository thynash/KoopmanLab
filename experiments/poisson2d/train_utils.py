import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds.

    Call this separately before each model training run so that
    Baseline KNO and PEDVINO are trained under the same random
    environment as far as reproducibility allows.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Use CUDA when available.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# METRICS
# ============================================================

def relative_l2_error(
    prediction,
    target,
    eps=1e-12,
):
    """
    Mean relative L2 error over a batch.

                ||prediction - target||_2
        ------------------------------------------
                    ||target||_2 + eps

    Each sample is evaluated independently, then averaged.
    """

    batch_size = prediction.shape[0]

    prediction = prediction.reshape(batch_size, -1)
    target = target.reshape(batch_size, -1)

    numerator = torch.norm(
        prediction - target,
        p=2,
        dim=1,
    )

    denominator = torch.norm(
        target,
        p=2,
        dim=1,
    )

    relative_error = numerator / (
        denominator + eps
    )

    return relative_error.mean()


def mse_error(prediction, target):
    """
    Mean squared error.
    """

    return F.mse_loss(
        prediction,
        target,
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
    Evaluate a model using identical metrics.

    Returns
    -------
    metrics : dict
        mse
        relative_l2
    """

    model.eval()

    total_mse = 0.0
    total_l2 = 0.0
    total_samples = 0

    for forcing, solution in data_loader:

        forcing = forcing.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        prediction, reconstruction = model(
            forcing
        )

        batch_size = forcing.shape[0]

        batch_mse = mse_error(
            prediction,
            solution,
        )

        batch_l2 = relative_l2_error(
            prediction,
            solution,
        )

        total_mse += (
            batch_mse.item() * batch_size
        )

        total_l2 += (
            batch_l2.item() * batch_size
        )

        total_samples += batch_size

    return {
        "mse": total_mse / total_samples,
        "relative_l2": total_l2 / total_samples,
    }


# ============================================================
# HISTORY
# ============================================================

def initialize_history():
    """
    Create the same basic history structure for both models.
    """

    return {
        "epoch": [],

        "train_total_loss": [],
        "train_prediction_loss": [],
        "train_reconstruction_loss": [],

        "val_mse": [],
        "val_relative_l2": [],

        "test_mse": [],
        "test_relative_l2": [],
    }


def append_history(
    history,
    epoch,
    train_metrics,
    val_metrics,
    test_metrics,
):
    """
    Add one epoch to a common history format.
    """

    history["epoch"].append(epoch)

    history["train_total_loss"].append(
        float(train_metrics["total_loss"])
    )

    history["train_prediction_loss"].append(
        float(train_metrics["prediction_loss"])
    )

    history["train_reconstruction_loss"].append(
        float(train_metrics["reconstruction_loss"])
    )

    history["val_mse"].append(
        float(val_metrics["mse"])
    )

    history["val_relative_l2"].append(
        float(val_metrics["relative_l2"])
    )

    history["test_mse"].append(
        float(test_metrics["mse"])
    )

    history["test_relative_l2"].append(
        float(test_metrics["relative_l2"])
    )


# ============================================================
# SAVING
# ============================================================

def save_json(data, path):
    """
    Save dictionaries/lists as formatted JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    path,
):
    """
    Save a reproducible model checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ============================================================
# EXPERIMENT TIMER
# ============================================================

class ExperimentTimer:
    """
    Simple timer for fair runtime reporting.
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            return 0.0

        return time.time() - self.start_time
