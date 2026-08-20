import os

import torch
import torch.nn.functional as F

from koopmanlab.models import kno

from experiments.burgers import config
from experiments.burgers.dataset import get_burgers_loaders
from experiments.burgers.train_utils import (
    set_seed,
    get_device,
    evaluate,
    relative_l2_error,
    initialize_history,
    append_history,
    save_json,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    ExperimentTimer,
    get_learning_rate,
)


# ============================================================
# MODEL
# ============================================================

def build_baseline_model():
    """
    Build the ORIGINAL KNO1d baseline.

    Architecture:

        Burgers initial condition
                ↓
            KNO1d Encoder
                ↓
            Koopman Operator
                ↓
            KNO1d Decoder
                ↓
            Prediction

    No physics encoder.
    No physics decoder.
    No variational loss.

    This is the data-driven baseline against which
    PEDVINO-KNO1d will be compared.
    """

    encoder = kno.encoder_mlp(
        t_len=config.T_LEN,
        op_size=config.OPERATOR_SIZE,
    )

    decoder = kno.decoder_mlp(
        t_len=config.T_LEN,
        op_size=config.OPERATOR_SIZE,
    )

    model = kno.KNO1d(
        encoder=encoder,
        decoder=decoder,
        op_size=config.OPERATOR_SIZE,
        modes_x=config.MODES_X,
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

        L_recon = MSE(reconstruction, initial_state)

    The Burgers operator learns:

        u(x, 0)  --->  u(x, T)
    """

    model.train()

    total_loss_sum = 0.0
    prediction_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    relative_l2_sum = 0.0

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

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        prediction, reconstruction = model(
            initial_state
        )

        # ----------------------------------------------------
        # SHAPE SAFETY
        # ----------------------------------------------------

        if prediction.shape != solution.shape:
            raise RuntimeError(
                "Prediction and solution shapes do not match. "
                f"Prediction: {tuple(prediction.shape)}, "
                f"Solution: {tuple(solution.shape)}"
            )

        if reconstruction.shape != initial_state.shape:
            raise RuntimeError(
                "Reconstruction and initial_state shapes "
                "do not match. "
                f"Reconstruction: {tuple(reconstruction.shape)}, "
                f"Initial state: {tuple(initial_state.shape)}"
            )

        # ----------------------------------------------------
        # ORIGINAL KNO LOSSES
        # ----------------------------------------------------

        prediction_loss = F.mse_loss(
            prediction,
            solution,
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            initial_state,
        )

        total_loss = (
            config.LAMBDA_PRED
            * prediction_loss
            +
            config.LAMBDA_RECON
            * reconstruction_loss
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        total_loss.backward()

        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        if (
            config.GRADIENT_CLIP is not None
            and config.GRADIENT_CLIP > 0.0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.GRADIENT_CLIP,
            )

        optimizer.step()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        batch_size = initial_state.shape[0]

        batch_relative_l2 = relative_l2_error(
            prediction.detach(),
            solution,
            eps=config.LOSS_EPS,
        )

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

        relative_l2_sum += (
            batch_relative_l2.item()
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Training DataLoader contains no samples."
        )

    return {
        "total_loss":
            total_loss_sum / total_samples,

        "prediction_loss":
            prediction_loss_sum / total_samples,

        "reconstruction_loss":
            reconstruction_loss_sum / total_samples,

        "relative_l2":
            relative_l2_sum / total_samples,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("BURGERS 1D - BASELINE KNO1d EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_burgers_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
    )

    print("\nDataset split:")
    print(
        f"Train: {len(train_loader.dataset)}"
    )
    print(
        f"Val  : {len(val_loader.dataset)}"
    )
    print(
        f"Test : {len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_baseline_model().to(
        device
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: Original KNO1d")
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = None

    if config.SCHEDULER.lower() == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS,
                eta_min=config.MIN_LEARNING_RATE,
            )
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = initialize_history()

    best_val_l2 = float("inf")
    best_epoch = -1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = None

    if config.EARLY_STOPPING:

        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
        )

    # --------------------------------------------------------
    # TIMER
    # --------------------------------------------------------

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
        # TRAIN
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        # ----------------------------------------------------
        # TEST
        #
        # Recorded every epoch for diagnostics only.
        # Model selection is ALWAYS based on validation L2.
        # ----------------------------------------------------

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        current_learning_rate = (
            get_learning_rate(
                optimizer
            )
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        append_history(
            history=history,
            epoch=epoch,
            train_loss=train_metrics[
                "total_loss"
            ],
            train_prediction_loss=train_metrics[
                "prediction_loss"
            ],
            train_reconstruction_loss=train_metrics[
                "reconstruction_loss"
            ],
            train_relative_l2=train_metrics[
                "relative_l2"
            ],
            val_metrics=val_metrics,
            learning_rate=current_learning_rate,
        )

        # Add test history for final comparison.
        if "test_relative_l2" not in history:
            history["test_relative_l2"] = []

        if "test_mse" not in history:
            history["test_mse"] = []

        history["test_relative_l2"].append(
            float(
                test_metrics["relative_l2"]
            )
        )

        history["test_mse"].append(
            float(
                test_metrics["mse"]
            )
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        #
        # Validation relative L2 ONLY.
        # ----------------------------------------------------

        improved = (
            val_metrics["relative_l2"]
            < best_val_l2
        )

        if improved:

            best_val_l2 = (
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            save_checkpoint(
                path=config.BASELINE_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "validation": val_metrics,
                    "test": test_metrics,
                },
                extra={
                    "experiment":
                        "burgers_baseline",
                    "model":
                        "KNO1d",
                    "best_val_relative_l2":
                        best_val_l2,
                },
            )

        # ----------------------------------------------------
        # SCHEDULER STEP
        # ----------------------------------------------------

        if scheduler is not None:
            scheduler.step()

        # ----------------------------------------------------
        # CONSOLE OUTPUT
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
            f"Train Loss: "
            f"{train_metrics['total_loss']:.6e} | "
            f"Pred: "
            f"{train_metrics['prediction_loss']:.6e} | "
            f"Recon: "
            f"{train_metrics['reconstruction_loss']:.6e} | "
            f"Train L2: "
            f"{train_metrics['relative_l2']:.6e} | "
            f"Val L2: "
            f"{val_metrics['relative_l2']:.6e} | "
            f"Test L2: "
            f"{test_metrics['relative_l2']:.6e} | "
            f"LR: {current_learning_rate:.3e}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if early_stopping is not None:

            early_stopping.step(
                val_metrics["relative_l2"]
            )

            if early_stopping.should_stop:

                print(
                    "\nEarly stopping triggered."
                )

                print(
                    f"Best epoch: {best_epoch}"
                )

                print(
                    f"Best validation L2: "
                    f"{best_val_l2:.6e}"
                )

                break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    timer.stop()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Elapsed time: "
        f"{timer.elapsed_minutes:.2f} minutes"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation relative L2: "
        f"{best_val_l2:.6e}"
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    print("\nLoading best validation checkpoint...")

    checkpoint = load_checkpoint(
        path=config.BASELINE_CHECKPOINT_PATH,
        model=model,
        device=device,
    )

    best_checkpoint_epoch = checkpoint.get(
        "epoch",
        best_epoch,
    )

    # ========================================================
    # FINAL EVALUATION
    #
    # This is the metric that will be used in the final
    # Baseline vs PEDVINO comparison.
    # ========================================================

    final_train_metrics = evaluate(
        model=model,
        data_loader=train_loader,
        device=device,
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

    print("\n" + "=" * 70)
    print("BEST CHECKPOINT - FINAL RESULTS")
    print("=" * 70)

    print(
        f"Checkpoint epoch: "
        f"{best_checkpoint_epoch}"
    )

    print(
        f"Train Relative L2: "
        f"{final_train_metrics['relative_l2']:.6e}"
    )

    print(
        f"Validation Relative L2: "
        f"{final_val_metrics['relative_l2']:.6e}"
    )

    print(
        f"Test Relative L2: "
        f"{final_test_metrics['relative_l2']:.6e}"
    )

    print(
        f"Test MSE: "
        f"{final_test_metrics['mse']:.6e}"
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    if config.SAVE_HISTORY:

        save_json(
            history,
            config.BASELINE_HISTORY_PATH,
        )

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    final_metrics = {
        "experiment":
            "Burgers 1D",

        "model":
            "Original KNO1d",

        "dataset_path":
            config.DATASET_PATH,

        "dataset_size":
            config.NUM_SAMPLES,

        "best_epoch":
            int(best_checkpoint_epoch),

        "training_time_seconds":
            float(timer.elapsed_seconds),

        "training_time_minutes":
            float(timer.elapsed_minutes),

        "best_validation_relative_l2":
            float(
                final_val_metrics["relative_l2"]
            ),

        "train": {
            key: float(value)
            for key, value
            in final_train_metrics.items()
            if key != "num_samples"
        },

        "validation": {
            key: float(value)
            for key, value
            in final_val_metrics.items()
            if key != "num_samples"
        },

        "test": {
            key: float(value)
            for key, value
            in final_test_metrics.items()
            if key != "num_samples"
        },

        "config": {
            "grid_size":
                config.GRID_SIZE_X,

            "viscosity":
                config.VISCOSITY,

            "final_time":
                config.FINAL_TIME,

            "operator_size":
                config.OPERATOR_SIZE,

            "modes_x":
                config.MODES_X,

            "decompose":
                config.DECOMPOSE,

            "epochs":
                config.EPOCHS,

            "batch_size":
                config.BATCH_SIZE,

            "learning_rate":
                config.LEARNING_RATE,

            "weight_decay":
                config.WEIGHT_DECAY,
        },
    }

    if config.SAVE_METRICS:

        save_json(
            final_metrics,
            config.BASELINE_METRICS_PATH,
        )

    print("\nSaved:")
    print(
        f"Checkpoint: "
        f"{config.BASELINE_CHECKPOINT_PATH}"
    )

    print(
        f"History: "
        f"{config.BASELINE_HISTORY_PATH}"
    )

    print(
        f"Metrics: "
        f"{config.BASELINE_METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
