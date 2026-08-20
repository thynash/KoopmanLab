import os
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set all random seeds for reproducible experiments.
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
    Return CUDA device if available, otherwise CPU.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# ALLEN-CAHN DATASET
# ============================================================

class AllenCahnDataset(Dataset):
    """
    Dataset for Allen-Cahn operator learning.

    Learning problem:

        u_0(x) -> u(x, T)

    Dataset file must contain:

        initial_state : [N, Nx, 1]
        solution      : [N, Nx, 1]
    """

    def __init__(self, dataset_path):

        if not os.path.exists(dataset_path):

            raise FileNotFoundError(
                f"Allen-Cahn dataset not found:\n"
                f"{dataset_path}"
            )

        print("\nLoading Allen-Cahn dataset from:")
        print(dataset_path)

        data = torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=False,
        )

        # ----------------------------------------------------
        # Expected dataset keys
        # ----------------------------------------------------

        if "initial_state" not in data:

            raise KeyError(
                "Dataset does not contain key "
                "'initial_state'."
            )

        if "solution" not in data:

            raise KeyError(
                "Dataset does not contain key "
                "'solution'."
            )

        self.initial_state = (
            data["initial_state"]
            .float()
            .contiguous()
        )

        self.solution = (
            data["solution"]
            .float()
            .contiguous()
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if len(self.initial_state) != len(self.solution):

            raise ValueError(
                "Initial state and solution have "
                "different numbers of samples."
            )

        print("\nDataset loaded successfully.")

        print(
            f"Initial state shape: "
            f"{tuple(self.initial_state.shape)}"
        )

        print(
            f"Solution shape:      "
            f"{tuple(self.solution.shape)}"
        )

    def __len__(self):

        return self.initial_state.shape[0]

    def __getitem__(self, index):

        return (
            self.initial_state[index],
            self.solution[index],
        )


# ============================================================
# DATA LOADERS
# ============================================================

def get_allen_cahn_loaders(
    batch_size,
    num_workers=0,
    seed=42,
):
    """
    Load and split the Allen-Cahn dataset.

    Expected split:

        Total : 2000
        Train : 1400
        Val   : 300
        Test  : 300

    The split is deterministic for a given seed.
    """

    # Import here to avoid circular imports
    from experiments.allen_cahn import config

    dataset = AllenCahnDataset(
        config.DATASET_PATH
    )

    total_samples = len(dataset)

    train_size = config.TRAIN_SIZE
    val_size = config.VAL_SIZE
    test_size = config.TEST_SIZE

    expected_total = (
        train_size
        + val_size
        + test_size
    )

    if total_samples != expected_total:

        raise ValueError(
            f"Dataset contains {total_samples} samples, "
            f"but configured split requires "
            f"{expected_total} samples "
            f"({train_size} train + "
            f"{val_size} val + "
            f"{test_size} test)."
        )

    # --------------------------------------------------------
    # Deterministic split
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [
            train_size,
            val_size,
            test_size,
        ],
        generator=generator,
    )

    # --------------------------------------------------------
    # DataLoader settings
    # --------------------------------------------------------

    pin_memory = torch.cuda.is_available()

    persistent_workers = (
        num_workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# RELATIVE L2 ERROR
# ============================================================

def relative_l2_error(
    prediction,
    target,
    eps=1e-12,
):
    """
    Compute mean relative L2 error over a batch.

                ||prediction - target||_2
        RelL2 = -------------------------
                    ||target||_2 + eps
    """

    batch_size = prediction.shape[0]

    prediction_flat = prediction.reshape(
        batch_size,
        -1,
    )

    target_flat = target.reshape(
        batch_size,
        -1,
    )

    numerator = torch.norm(
        prediction_flat - target_flat,
        dim=1,
        p=2,
    )

    denominator = torch.norm(
        target_flat,
        dim=1,
        p=2,
    )

    relative_error = numerator / (
        denominator + eps
    )

    return relative_error.mean()


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
    Evaluate KNO / PEDVINO-compatible model.

    Expected dataset batch:

        initial_state, solution

    Expected model output:

        prediction, reconstruction

    Metrics:

        MSE
        Relative L2
    """

    model.eval()

    total_mse_sum = 0.0
    total_relative_l2_sum = 0.0

    total_samples = 0

    for batch in data_loader:

        initial_state, solution = batch

        initial_state = initial_state.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction, _ = model(
            initial_state
        )

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        mse = torch.mean(
            (prediction - solution) ** 2
        )

        # ----------------------------------------------------
        # Relative L2
        # ----------------------------------------------------

        relative_l2 = relative_l2_error(
            prediction,
            solution,
        )

        batch_size = initial_state.shape[0]

        total_mse_sum += (
            mse.item()
            * batch_size
        )

        total_relative_l2_sum += (
            relative_l2.item()
            * batch_size
        )

        total_samples += batch_size

    return {

        "mse":
            total_mse_sum
            / max(total_samples, 1),

        "relative_l2":
            total_relative_l2_sum
            / max(total_samples, 1),
    }


# ============================================================
# HISTORY
# ============================================================

def initialize_history():
    """
    Initialize experiment history.

    This format is shared by the Allen-Cahn baseline
    and PEDVINO experiments.
    """

    return {

        "epoch": [],

        # ----------------------------------------------------
        # Training losses
        # ----------------------------------------------------

        "train_total_loss": [],

        "train_prediction_loss": [],

        "train_reconstruction_loss": [],

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        "val_mse": [],

        "val_relative_l2": [],

        # ----------------------------------------------------
        # Test metrics
        # ----------------------------------------------------

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
    Append one epoch of experiment metrics.

    This signature is intentionally compatible with:

        append_history(
            history=history,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
    """

    history["epoch"].append(
        int(epoch)
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history["train_total_loss"].append(
        float(
            train_metrics.get(
                "total_loss",
                0.0,
            )
        )
    )

    history[
        "train_prediction_loss"
    ].append(
        float(
            train_metrics.get(
                "prediction_loss",
                0.0,
            )
        )
    )

    history[
        "train_reconstruction_loss"
    ].append(
        float(
            train_metrics.get(
                "reconstruction_loss",
                0.0,
            )
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    history["val_mse"].append(
        float(
            val_metrics.get(
                "mse",
                0.0,
            )
        )
    )

    history[
        "val_relative_l2"
    ].append(
        float(
            val_metrics.get(
                "relative_l2",
                0.0,
            )
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    history["test_mse"].append(
        float(
            test_metrics.get(
                "mse",
                0.0,
            )
        )
    )

    history[
        "test_relative_l2"
    ].append(
        float(
            test_metrics.get(
                "relative_l2",
                0.0,
            )
        )
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def save_json(
    data,
    path,
):
    """
    Save dictionary as JSON.
    """

    directory = os.path.dirname(path)

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
            f"is not JSON serializable"
        )

    with open(
        path,
        "w",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=convert_to_serializable,
        )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    path,
):
    """
    Save experiment checkpoint.
    """

    directory = os.path.dirname(path)

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

    def elapsed(self):
        """
        Return elapsed time in seconds.

        If stop() has not been called yet,
        measure time until the current moment.
        """

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
