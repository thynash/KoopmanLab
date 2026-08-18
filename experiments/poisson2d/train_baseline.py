import os
import torch
import torch.nn.functional as F

from koopmanlab.models import kno

from experiments.poisson2d import config
from experiments.poisson2d.dataset import get_poisson2d_loaders
from experiments.poisson2d.train_utils import (
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
    Build the ORIGINAL KNO2d baseline.

    No physics encoder.
    No physics decoder.
    No variational loss.

    This is the benchmark against which PEDVINO will be compared.
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
    Train one epoch using the original KNO objective:

        L = lambda_pred * L_pred
          + lambda_recon * L_recon

    where:

        L_pred  = MSE(prediction, solution)
        L_recon = MSE(reconstruction, forcing)
    """

    model.train()

    total_loss_sum = 0.0
    prediction_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
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

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        prediction, reconstruction = model(
            forcing
        )

        # ----------------------------------------------------
        # Original KNO losses
        # ----------------------------------------------------

        prediction_loss = F.mse_loss(
            prediction,
            solution,
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            forcing,
        )

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

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate weighted batch statistics
        # ----------------------------------------------------

        batch_size = forcing.shape[0]

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        prediction_loss_sum += (
            prediction_loss.item()
            * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "total_loss":
            total_loss_sum / total_samples,

        "prediction_loss":
            prediction_loss_sum / total_samples,

        "reconstruction_loss":
            reconstruction_loss_sum / total_samples,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("POISSON 2D - BASELINE KNO EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = (
        get_poisson2d_loaders(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            seed=config.SEED,
        )
    )

    print("\nDataset split:")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val  : {len(val_loader.dataset)}")
    print(f"Test : {len(test_loader.dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_baseline_model().to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: Original KNO2d")
    print(f"Trainable parameters: {num_parameters:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = initialize_history()

    best_val_l2 = float("inf")
    best_epoch = -1

    timer = ExperimentTimer()
    timer.start()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, config.EPOCHS + 1):

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        append_history(
            history=history,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        # ----------------------------------------------------
        # Save best model using validation L2 only
        # ----------------------------------------------------

        if val_metrics["relative_l2"] < best_val_l2:

            best_val_l2 = (
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            checkpoint_path = os.path.join(
                config.BASELINE_RESULTS_DIR,
                "best_model.pt",
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "validation": val_metrics,
                    "test": test_metrics,
                },
                path=checkpoint_path,
            )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
            f"Train Loss: {train_metrics['total_loss']:.6e} | "
            f"Pred: {train_metrics['prediction_loss']:.6e} | "
            f"Recon: {train_metrics['reconstruction_loss']:.6e} | "
            f"Val L2: {val_metrics['relative_l2']:.6e} | "
            f"Test L2: {test_metrics['relative_l2']:.6e}"
        )

    # ========================================================
    # Final evaluation using BEST validation checkpoint
    # ========================================================

    checkpoint_path = os.path.join(
        config.BASELINE_RESULTS_DIR,
        "best_model.pt",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

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
    # Save history
    # ========================================================

    history_path = os.path.join(
        config.BASELINE_RESULTS_DIR,
        "history.json",
    )

    save_json(
        history,
        history_path,
    )

    # ========================================================
    # Save final metrics
    # ========================================================

    final_metrics = {
        "model": "Original KNO2d",
        "experiment": "Poisson2D",

        "seed": config.SEED,

        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,

        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,

        "trainable_parameters": num_parameters,

        "best_epoch": best_epoch,

        "best_validation_relative_l2":
            final_val_metrics["relative_l2"],

        "best_validation_mse":
            final_val_metrics["mse"],

        "test_relative_l2":
            final_test_metrics["relative_l2"],

        "test_mse":
            final_test_metrics["mse"],

        "training_time_seconds":
            elapsed_time,
    }

    metrics_path = os.path.join(
        config.BASELINE_RESULTS_DIR,
        "metrics.json",
    )

    save_json(
        final_metrics,
        metrics_path,
    )

    # ========================================================
    # Final output
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE KNO EXPERIMENT COMPLETED")
    print("=" * 70)

    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation Relative L2: "
        f"{final_val_metrics['relative_l2']:.6e}"
    )
    print(
        "Final test Relative L2: "
        f"{final_test_metrics['relative_l2']:.6e}"
    )
    print(
        "Final test MSE: "
        f"{final_test_metrics['mse']:.6e}"
    )
    print(
        f"Training time: {elapsed_time:.2f} seconds"
    )

    print("\nResults saved to:")
    print(config.BASELINE_RESULTS_DIR)


if __name__ == "__main__":
    main()
