#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darcy 2D - Original VINO-style Experiment
==========================================

Problem:
    -div(a(x,y) grad(u(x,y))) = f(x,y)
    u = 0 on boundary

Operator:
    a(x,y) -> u(x,y)

Experiment:
    Train      : 1400
    Validation : 300
    Test       : 300

Model:
    Boundary-aware Fourier Neural Operator

Boundary condition:
    u_hat(x,y) =
        raw_FNO(x,y)
        * x * y * (x - 1) * (y - 1)

Loss:
    L_total =
        lambda_data * L_data
        + lambda_physics * L_VINO

where:
    L_VINO =
        mean[
            0.5 * a * |grad(u_hat)|^2 - f * u_hat
        ]

The exact solution is used for the data-enhanced VINO
supervision term and validation/testing.
"""

import os
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    random_split,
)


# ============================================================
# PATH SETUP
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


import config
from dataset import load_dataset


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

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

    return torch.device(
        config.DEVICE
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
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
    Per-sample relative L2 error, averaged over batch.
    """

    prediction = prediction.reshape(
        prediction.shape[0],
        -1,
    )

    target = target.reshape(
        target.shape[0],
        -1,
    )

    numerator = torch.linalg.vector_norm(
        prediction - target,
        ord=2,
        dim=1,
    )

    denominator = torch.linalg.vector_norm(
        target,
        ord=2,
        dim=1,
    ).clamp_min(eps)

    return torch.mean(
        numerator / denominator
    )


# ============================================================
# SPECTRAL CONVOLUTION 2D
# ============================================================

class SpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x,
        modes_y,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = (
            1.0
            / (
                in_channels
                * out_channels
            )
        )

        self.weights_pos = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

        self.weights_neg = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )


    @staticmethod
    def compl_mul2d(
        input_,
        weights,
    ):

        return torch.einsum(
            "bixy,ioxy->boxy",
            input_,
            weights,
        )


    def forward(
        self,
        x,
    ):

        batch_size = x.shape[0]

        x_ft = torch.fft.rfft2(
            x,
            norm="ortho",
        )

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes_x = min(
            self.modes_x,
            x.shape[-2],
        )

        modes_y = min(
            self.modes_y,
            x.shape[-1] // 2 + 1,
        )

        out_ft[
            :,
            :,
            :modes_x,
            :modes_y,
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
            self.weights_pos[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
        )

        out_ft[
            :,
            :,
            -modes_x:,
            :modes_y,
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                -modes_x:,
                :modes_y,
            ],
            self.weights_neg[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
        )

        return torch.fft.irfft2(
            out_ft,
            s=(
                x.shape[-2],
                x.shape[-1],
            ),
            norm="ortho",
        )


# ============================================================
# FNO BLOCK
# ============================================================

class FNOBlock2d(nn.Module):

    def __init__(
        self,
        width,
        modes_x,
        modes_y,
        activate=True,
    ):

        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes_x,
            modes_y,
        )

        self.local = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.activate = activate


    def forward(
        self,
        x,
    ):

        output = (
            self.spectral(x)
            + self.local(x)
        )

        if self.activate:

            output = F.gelu(output)

        return output


# ============================================================
# BOUNDARY-AWARE FNO
# ============================================================

class DarcyVINO(nn.Module):

    def __init__(
        self,
        modes_x,
        modes_y,
        width,
        depth,
    ):

        super().__init__()

        self.modes_x = modes_x
        self.modes_y = modes_y

        self.width = width
        self.depth = depth

        # Input:
        # coefficient a + coordinate x + coordinate y
        self.input_projection = nn.Linear(
            3,
            width,
        )

        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    width,
                    modes_x,
                    modes_y,
                    activate=(
                        layer_index < depth - 1
                    ),
                )
                for layer_index in range(depth)
            ]
        )

        self.projection_1 = nn.Linear(
            width,
            width,
        )

        self.projection_2 = nn.Linear(
            width,
            1,
        )


    def get_grid(
        self,
        batch_size,
        size_x,
        size_y,
        device,
        dtype,
    ):

        x = torch.linspace(
            0.0,
            1.0,
            size_x,
            device=device,
            dtype=dtype,
        )

        y = torch.linspace(
            0.0,
            1.0,
            size_y,
            device=device,
            dtype=dtype,
        )

        grid_x, grid_y = torch.meshgrid(
            x,
            y,
            indexing="ij",
        )

        grid = torch.stack(
            [
                grid_x,
                grid_y,
            ],
            dim=-1,
        )

        grid = grid.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        return grid


    def forward(
        self,
        coefficient,
    ):

        batch_size = coefficient.shape[0]
        size_x = coefficient.shape[1]
        size_y = coefficient.shape[2]

        grid = self.get_grid(
            batch_size=batch_size,
            size_x=size_x,
            size_y=size_y,
            device=coefficient.device,
            dtype=coefficient.dtype,
        )

        # [B, X, Y, 1] + coordinates
        x = torch.cat(
            [
                coefficient,
                grid,
            ],
            dim=-1,
        )

        x = self.input_projection(x)

        # [B, X, Y, W] -> [B, W, X, Y]
        x = x.permute(
            0,
            3,
            1,
            2,
        )

        for block in self.blocks:

            x = block(x)

        # [B, W, X, Y] -> [B, X, Y, W]
        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = F.gelu(
            self.projection_1(x)
        )

        raw_output = self.projection_2(x)

        # ----------------------------------------------------
        # HARD DIRICHLET BOUNDARY CONDITION
        #
        # Original VINO style:
        #
        # m(x,y) = x*y*(x-1)*(y-1)
        #
        # u = raw_output * m
        # ----------------------------------------------------

        multiplier = (
            grid[..., 0:1]
            * grid[..., 1:2]
            * (grid[..., 0:1] - 1.0)
            * (grid[..., 1:2] - 1.0)
        )

        return raw_output * multiplier


# ============================================================
# VINO DARCY VARIATIONAL LOSS
# ============================================================

class VinoDarcyLoss(nn.Module):

    """
    Darcy energy functional:

        Pi[u] =
            integral(
                0.5 * a * |grad u|^2
                - f * u
            ) dx dy

    The training loss is:

        L_total =
            lambda_data * L_data
            + lambda_physics * Pi[u]

    For stable optimization, the energy is averaged
    over the physical domain.
    """

    def __init__(
        self,
        dx,
        dy,
        lambda_data,
        lambda_physics,
    ):

        super().__init__()

        self.dx = float(dx)
        self.dy = float(dy)

        self.lambda_data = float(
            lambda_data
        )

        self.lambda_physics = float(
            lambda_physics
        )


    def gradients(
        self,
        u,
    ):

        # u:
        # [B, X, Y, 1]

        u = u.squeeze(-1)

        du_dx = torch.zeros_like(u)
        du_dy = torch.zeros_like(u)

        # Central differences
        du_dx[
            :,
            1:-1,
            :,
        ] = (
            u[:, 2:, :]
            - u[:, :-2, :]
        ) / (
            2.0 * self.dx
        )

        du_dy[
            :,
            :,
            1:-1,
        ] = (
            u[:, :, 2:]
            - u[:, :, :-2]
        ) / (
            2.0 * self.dy
        )

        # One-sided boundaries
        du_dx[
            :,
            0,
            :,
        ] = (
            u[:, 1, :]
            - u[:, 0, :]
        ) / self.dx

        du_dx[
            :,
            -1,
            :,
        ] = (
            u[:, -1, :]
            - u[:, -2, :]
        ) / self.dx

        du_dy[
            :,
            :,
            0,
        ] = (
            u[:, :, 1]
            - u[:, :, 0]
        ) / self.dy

        du_dy[
            :,
            :,
            -1,
        ] = (
            u[:, :, -1]
            - u[:, :, -2]
        ) / self.dy

        return du_dx, du_dy


    def forward(
        self,
        prediction,
        solution,
        coefficient,
        forcing,
    ):

        # ----------------------------------------------------
        # DATA LOSS
        # ----------------------------------------------------

        data_loss = relative_l2_error(
            prediction,
            solution,
        )

        # ----------------------------------------------------
        # VARIATIONAL ENERGY
        # ----------------------------------------------------

        du_dx, du_dy = self.gradients(
            prediction
        )

        a = coefficient.squeeze(-1)
        f = forcing.squeeze(-1)
        u = prediction.squeeze(-1)

        gradient_squared = (
            du_dx.pow(2)
            + du_dy.pow(2)
        )

        energy_density = (
            0.5
            * a
            * gradient_squared
            - f * u
        )

        energy = torch.mean(
            energy_density
        )

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_loss = (
            self.lambda_data
            * data_loss
            + self.lambda_physics
            * energy
        )

        return {
            "total_loss":
                total_loss,

            "data_loss":
                data_loss,

            "physics_loss":
                energy,
        }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_sum = 0.0
    data_sum = 0.0
    physics_sum = 0.0
    samples = 0

    for (
        coefficient,
        solution,
        forcing,
    ) in loader:

        coefficient = coefficient.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        forcing = forcing.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            coefficient
        )

        losses = criterion(
            prediction=prediction,
            solution=solution,
            coefficient=coefficient,
            forcing=forcing,
        )

        total_loss = losses[
            "total_loss"
        ]

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.VINO_GRADIENT_CLIP,
        )

        optimizer.step()

        batch_size = coefficient.shape[0]

        total_sum += (
            total_loss.detach().item()
            * batch_size
        )

        data_sum += (
            losses["data_loss"].detach().item()
            * batch_size
        )

        physics_sum += (
            losses["physics_loss"].detach().item()
            * batch_size
        )

        samples += batch_size

    return {
        "total_loss":
            total_sum / samples,

        "data_loss":
            data_sum / samples,

        "physics_loss":
            physics_sum / samples,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    relative_l2_sum = 0.0
    mse_sum = 0.0
    samples = 0

    for (
        coefficient,
        solution,
        forcing,
    ) in loader:

        coefficient = coefficient.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            coefficient
        )

        batch_size = coefficient.shape[0]

        relative_l2 = relative_l2_error(
            prediction,
            solution,
        )

        mse = F.mse_loss(
            prediction,
            solution,
        )

        relative_l2_sum += (
            relative_l2.item()
            * batch_size
        )

        mse_sum += (
            mse.item()
            * batch_size
        )

        samples += batch_size

    return {
        "relative_l2":
            relative_l2_sum / samples,

        "mse":
            mse_sum / samples,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY 2D - VINO EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        config.SEED
    )

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed  : {config.SEED}")

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    os.makedirs(
        config.VINO_RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset(
        config.DATASET_PATH
    )

    expected_samples = (
        config.TRAIN_SIZE
        + config.VAL_SIZE
        + config.TEST_SIZE
    )

    if len(dataset) != expected_samples:

        raise RuntimeError(
            f"Dataset has {len(dataset)} samples, "
            f"expected {expected_samples}."
        )

    generator = torch.Generator().manual_seed(
        config.SEED
    )

    train_dataset, val_dataset, test_dataset = (
        random_split(
            dataset,
            [
                config.TRAIN_SIZE,
                config.VAL_SIZE,
                config.TEST_SIZE,
            ],
            generator=generator,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.VINO_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VINO_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.VINO_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    print("\nDataset split:")
    print(
        f"Train: {len(train_dataset)}"
    )
    print(
        f"Val  : {len(val_dataset)}"
    )
    print(
        f"Test : {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DarcyVINO(
        modes_x=config.VINO_MODES_X,
        modes_y=config.VINO_MODES_Y,
        width=config.VINO_WIDTH,
        depth=config.VINO_DEPTH,
    ).to(device)

    parameter_count = count_parameters(
        model
    )

    print("\nModel: Darcy VINO-FNO")
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Target parameters   : ~84,000"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.VINO_LEARNING_RATE,
        weight_decay=config.VINO_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.VINO_GAMMA,
        patience=config.VINO_PATIENCE,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = VinoDarcyLoss(
        dx=config.DX,
        dy=config.DY,
        lambda_data=config.VINO_LAMBDA_DATA,
        lambda_physics=config.VINO_LAMBDA_PHYSICS,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "train_total_loss": [],
        "train_data_loss": [],
        "train_physics_loss": [],
        "val_relative_l2": [],
        "val_mse": [],
        "learning_rate": [],
    }

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    best_val_relative_l2 = float("inf")
    best_epoch = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining started...\n")

    for epoch in range(
        1,
        config.VINO_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step(
            val_metrics["relative_l2"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history[
            "train_total_loss"
        ].append(
            train_metrics["total_loss"]
        )

        history[
            "train_data_loss"
        ].append(
            train_metrics["data_loss"]
        )

        history[
            "train_physics_loss"
        ].append(
            train_metrics["physics_loss"]
        )

        history[
            "val_relative_l2"
        ].append(
            val_metrics["relative_l2"]
        )

        history[
            "val_mse"
        ].append(
            val_metrics["mse"]
        )

        history[
            "learning_rate"
        ].append(
            current_lr
        )

        if (
            val_metrics["relative_l2"]
            < best_val_relative_l2
        ):

            best_val_relative_l2 = (
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_relative_l2":
                        best_val_relative_l2,

                    "parameter_count":
                        parameter_count,

                    "model_config": {
                        "modes_x":
                            config.VINO_MODES_X,

                        "modes_y":
                            config.VINO_MODES_Y,

                        "width":
                            config.VINO_WIDTH,

                        "depth":
                            config.VINO_DEPTH,
                    },
                },
                config.VINO_CHECKPOINT_PATH,
            )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{config.VINO_EPOCHS} | "
            f"Train Total: "
            f"{train_metrics['total_loss']:.6e} | "
            f"Data: "
            f"{train_metrics['data_loss']:.6e} | "
            f"VINO: "
            f"{train_metrics['physics_loss']:.6e} | "
            f"Val RelL2: "
            f"{val_metrics['relative_l2']:.6e} | "
            f"LR: "
            f"{current_lr:.2e} | "
            f"Time: "
            f"{elapsed:.2f}s"
        )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        config.VINO_CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    print("\n" + "=" * 70)
    print("FINAL VINO RESULTS")
    print("=" * 70)

    print(
        f"Best epoch        : {best_epoch}"
    )

    print(
        f"Best Val Rel L2   : "
        f"{best_val_relative_l2:.8e}"
    )

    print(
        f"Test Relative L2  : "
        f"{test_metrics['relative_l2']:.8e}"
    )

    print(
        f"Test MSE          : "
        f"{test_metrics['mse']:.8e}"
    )

    print(
        f"Parameters        : "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        config.VINO_HISTORY_PATH,
        "w",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "experiment":
            "darcy_vino",

        "method":
            "Data-enhanced VINO",

        "best_epoch":
            best_epoch,

        "best_val_relative_l2":
            best_val_relative_l2,

        "test_relative_l2":
            test_metrics["relative_l2"],

        "test_mse":
            test_metrics["mse"],

        "parameter_count":
            parameter_count,

        "train_size":
            len(train_dataset),

        "val_size":
            len(val_dataset),

        "test_size":
            len(test_dataset),

        "epochs":
            config.VINO_EPOCHS,

        "batch_size":
            config.VINO_BATCH_SIZE,

        "learning_rate":
            config.VINO_LEARNING_RATE,

        "vino_modes_x":
            config.VINO_MODES_X,

        "vino_modes_y":
            config.VINO_MODES_Y,

        "vino_width":
            config.VINO_WIDTH,

        "vino_depth":
            config.VINO_DEPTH,

        "lambda_data":
            config.VINO_LAMBDA_DATA,

        "lambda_physics":
            config.VINO_LAMBDA_PHYSICS,

        "boundary_condition":
            "hard_dirichlet_xy_xminus1_yminus1",
    }

    with open(
        config.VINO_METRICS_PATH,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        f"  Checkpoint: "
        f"{config.VINO_CHECKPOINT_PATH}"
    )

    print(
        f"  History   : "
        f"{config.VINO_HISTORY_PATH}"
    )

    print(
        f"  Metrics   : "
        f"{config.VINO_METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import time

    main()
