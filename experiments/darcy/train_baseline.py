import os
import torch
import torch.nn.functional as F

from koopmanlab.models import kno

from experiments.darcy import config
from experiments.darcy.dataset import get_darcy_loaders
from experiments.darcy.train_utils import (
    set_seed,
    get_device,
    evaluate,
    initialize_history,
    append_history,
    save_json,
    save_checkpoint,
    ExperimentTimer,
)


# ============================================================
# MODEL
# ============================================================

def build_baseline_model():
    """
    Build the original KNO2d baseline.

    Darcy learning task:

        a(x, y) -> u(x, y)

    where:
        a = permeability / coefficient field
        u = Darcy PDE solution

    The forcing field is stored in the dataset but is not used
    by the baseline KNO model.
    """

    encoder = kno.encoder_conv2d(
        t_len=config.T_LEN,
        op_size=config.OPERATOR_SIZE,
    )

    decoder = kno.decoder_conv2d(
        t_len=config.T_LEN,
        op_size=config.OPERATOR_SIZE,
    )

    model = kno.KNO2d(
        encoder=encoder,
        decoder=decoder,
        op_size=config.OPERATOR_SIZE,
        modes_x=config.MODES_X,
        modes_y=config.MODES_Y,
        decompose=config.DECOMPOSE,
        linear_type=config.LINEAR_TYPE,
        normalization=config.NORMALIZATION,
    )

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
):
    """
    Train one epoch.

    Dataset format:

        coefficient, solution, forcing

    Baseline objective:

        L =
            lambda_pred  * L_prediction
          + lambda_recon * L_reconstruction

    where:

        L_prediction =
            MSE(predicted_solution, solution)

        L_reconstruction =
            MSE(reconstructed_coefficient, coefficient)
    """

    model.train()

    total_loss_sum = 0.0
    prediction_loss_sum = 0.0
    reconstruction_loss_sum = 0.0

    total_samples = 0

    for batch in data_loader:

        # ----------------------------------------------------
        # Darcy dataset returns:
        #
        # coefficient = a(x,y)
        # solution    = u(x,y)
        # forcing     = f(x,y)
        #
        # Baseline ignores forcing.
        # ----------------------------------------------------

        coefficient, solution, _ = batch

        coefficient = coefficient.to(
            device,
            non_blocking=True,
        )

        solution = solution.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Zero gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction, reconstruction = model(
            coefficient
        )

        # ----------------------------------------------------
        # Prediction loss
        # ----------------------------------------------------

        prediction_loss = F.mse_loss(
            prediction,
            solution,
        )

        # ----------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------

        reconstruction_loss = F.mse_loss(
            reconstruction,
            coefficient,
        )

        # ----------------------------------------------------
        # Total baseline objective
        # ----------------------------------------------------

        total_loss = (
            config.LAMBDA_PRED
            * prediction_loss
            +
            config.LAMBDA_RECON
            * reconstruction_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        gradient_clip = getattr(
            config,
            "GRADIENT_CLIP",
            None,
        )

        if gradient_clip is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = coefficient.shape[0]

        total_loss_sum += (
            total_loss.detach().item()
            * batch_size
        )

        prediction_loss_sum += (
            prediction_loss.detach().item()
            * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

    # --------------------------------------------------------
    # Epoch metrics
    # --------------------------------------------------------

    return {
        "total_loss":
            total_loss_sum / max(total_samples, 1),

        "prediction_loss":
            prediction_loss_sum / max(total_samples, 1),

        "reconstruction_loss":
            reconstruction_loss_sum / max(total_samples, 1),
    }


# ============================================================
# CREATE OPTIMIZER
# ============================================================

def build_optimizer(model):
    """
    Build optimizer from config.
    """

    optimizer_name = getattr(
        config,
        "OPTIMIZER",
        "adamw",
    ).lower()

    if optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    elif optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    return optimizer


# ============================================================
# CREATE SCHEDULER
# ============================================================

def build_scheduler(optimizer):
    """
    Build learning-rate scheduler.
    """

    scheduler_name = getattr(
        config,
        "SCHEDULER",
        "none",
    ).lower()

    if scheduler_name == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=config.MIN_LEARNING_RATE,
        )

    elif scheduler_name == "none":

        scheduler = None

    else:

        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )

    return scheduler


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY FLOW - BASELINE KNO EXPERIMENT")
    print("=" * 70)

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")

    # ========================================================
    # CREATE RESULT DIRECTORY
    # ========================================================

    os.makedirs(
        config.BASELINE_RESULTS_DIR,
        exist_ok=True,
    )

    # ========================================================
    # DATA
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_darcy_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
    )

    print("\nDataset split:")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val  : {len(val_loader.dataset)}")
    print(f"Test : {len(test_loader.dataset)}")

    # ========================================================
    # MODEL
    # ========================================================

    model = build_baseline_model().to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: Original KNO2d")
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = build_optimizer(model)

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = build_scheduler(
        optimizer
    )

    # ========================================================
    # HISTORY
    # ========================================================

    history = initialize_history()

    best_val_l2 = float("inf")
    best_epoch = -1

    timer = ExperimentTimer()
    timer.start()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        config.EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        # ----------------------------------------------------
        # Save epoch history
        # ----------------------------------------------------

        append_history(
            history=history,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["relative_l2"]
            < best_val_l2
        ):

            best_val_l2 = (
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "validation": val_metrics,
                    "test": test_metrics,
                },
                path=config.BASELINE_CHECKPOINT_PATH,
            )

        # ----------------------------------------------------
        # Scheduler step
        # ----------------------------------------------------

        if scheduler is not None:

            scheduler.step()

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if (
            epoch % config.PRINT_EVERY == 0
        ):

            print(
                f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
                f"Train Loss: "
                f"{train_metrics['total_loss']:.6e} | "
                f"Pred: "
                f"{train_metrics['prediction_loss']:.6e} | "
                f"Recon: "
                f"{train_metrics['reconstruction_loss']:.6e} | "
                f"Val L2: "
                f"{val_metrics['relative_l2']:.6e} | "
                f"Test L2: "
                f"{test_metrics['relative_l2']:.6e} | "
                f"LR: "
                f"{current_lr:.3e}"
            )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    if best_epoch < 0:

        raise RuntimeError(
            "No best checkpoint was saved."
        )

    checkpoint = torch.load(
        config.BASELINE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_val_metrics = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    final_test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    elapsed_time = timer.elapsed()

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    if getattr(
        config,
        "SAVE_HISTORY",
        True,
    ):

        save_json(
            history,
            config.BASELINE_HISTORY_PATH,
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_metrics = {

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        "model": "Original KNO2d",
        "experiment": "DarcyFlow",

        "seed": config.SEED,

        # ----------------------------------------------------
        # Training configuration
        # ----------------------------------------------------

        "epochs_configured": config.EPOCHS,

        "epochs_completed": len(
            history.get("epoch", [])
        ),

        "batch_size": config.BATCH_SIZE,

        "learning_rate": config.LEARNING_RATE,

        "weight_decay": config.WEIGHT_DECAY,

        "optimizer": getattr(
            config,
            "OPTIMIZER",
            "adamw",
        ),

        "scheduler": getattr(
            config,
            "SCHEDULER",
            "none",
        ),

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "trainable_parameters":
            num_parameters,

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        "best_epoch":
            best_epoch,

        "best_validation_relative_l2":
            float(
                final_val_metrics[
                    "relative_l2"
                ]
            ),

        "best_validation_mse":
            float(
                final_val_metrics[
                    "mse"
                ]
            ),

        # ----------------------------------------------------
        # Final test
        # ----------------------------------------------------

        "test_relative_l2":
            float(
                final_test_metrics[
                    "relative_l2"
                ]
            ),

        "test_mse":
            float(
                final_test_metrics[
                    "mse"
                ]
            ),

        # ----------------------------------------------------
        # Training time
        # ----------------------------------------------------

        "training_time_seconds":
            float(elapsed_time),
    }

    # ========================================================
    # SAVE METRICS.JSON
    # ========================================================

    if getattr(
        config,
        "SAVE_METRICS",
        True,
    ):

        save_json(
            final_metrics,
            config.BASELINE_METRICS_PATH,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DARCY BASELINE TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best epoch              : "
        f"{best_epoch}"
    )

    print(
        f"Best validation L2      : "
        f"{final_val_metrics['relative_l2']:.6e}"
    )

    print(
        f"Best validation MSE     : "
        f"{final_val_metrics['mse']:.6e}"
    )

    print(
        f"Final test L2           : "
        f"{final_test_metrics['relative_l2']:.6e}"
    )

    print(
        f"Final test MSE          : "
        f"{final_test_metrics['mse']:.6e}"
    )

    print(
        f"Training time           : "
        f"{elapsed_time:.2f} seconds"
    )

    print("\nSaved files:")

    print(
        f"Best checkpoint : "
        f"{config.BASELINE_CHECKPOINT_PATH}"
    )

    print(
        f"History         : "
        f"{config.BASELINE_HISTORY_PATH}"
    )

    print(
        f"Metrics         : "
        f"{config.BASELINE_METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
