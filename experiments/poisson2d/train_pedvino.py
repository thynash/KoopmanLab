import os
import torch

from koopmanlab.model_pedvino import PEDVINO
from koopmanlab.pde_functionals import PoissonFunctional
from koopmanlab.variational_loss import GeneralVariationalLoss
from koopmanlab.pedvino_loss import PEDVINOLoss

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

def build_pedvino_model():
    """
    Build the proposed PEDVINO model.
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
# LOSS
# ============================================================

def build_loss():

    functional = PoissonFunctional()

    variational_loss = GeneralVariationalLoss(
        functional=functional,
        spatial_dim=2,
        dx=config.DX,
        dy=config.DY,
        reduction="none",
        mode="functional",
    )

    criterion = PEDVINOLoss(
        variational_loss=variational_loss,
        lambda_pred=config.LAMBDA_PRED,
        lambda_recon=config.LAMBDA_RECON,
        lambda_energy=config.LAMBDA_ENERGY,
        lambda_grad=config.LAMBDA_GRAD,
        lambda_bc=config.LAMBDA_BC,
        energy_loss_type=getattr(
            config,
            "ENERGY_LOSS_TYPE",
            "relative",
        ),
        energy_eps=getattr(
            config,
            "ENERGY_EPS",
            1e-8,
        ),
        use_relative_prediction_loss=getattr(
            config,
            "USE_RELATIVE_PREDICTION_LOSS",
            True,
        ),
        prediction_eps=getattr(
            config,
            "LOSS_EPS",
            1e-8,
        ),
    )

    return criterion

# ============================================================
# PHYSICS WARM-UP
# ============================================================

def get_warmup_factor(epoch, warmup_epochs):
    """
    Linear warm-up factor.
    """

    if warmup_epochs is None or warmup_epochs <= 0:
        return 1.0

    return min(
        1.0,
        float(epoch) / float(warmup_epochs),
    )


def update_physics_weights(
    criterion,
    epoch,
):
    """
    Update PEDVINO physics weights according to config warm-up.

    The base weights remain defined in config.py.
    """

    energy_factor = get_warmup_factor(
        epoch,
        config.ENERGY_WARMUP_EPOCHS,
    )

    grad_factor = get_warmup_factor(
        epoch,
        config.GRAD_WARMUP_EPOCHS,
    )

    bc_factor = get_warmup_factor(
        epoch,
        config.BC_WARMUP_EPOCHS,
    )

    if hasattr(criterion, "lambda_energy"):
        criterion.lambda_energy = (
            float(config.LAMBDA_ENERGY)
            * energy_factor
        )

    if hasattr(criterion, "lambda_grad"):
        criterion.lambda_grad = (
            float(config.LAMBDA_GRAD)
            * grad_factor
        )

    if hasattr(criterion, "lambda_bc"):
        criterion.lambda_bc = (
            float(config.LAMBDA_BC)
            * bc_factor
        )

    return {
        "energy_weight": (
            float(config.LAMBDA_ENERGY)
            * energy_factor
        ),
        "grad_weight": (
            float(config.LAMBDA_GRAD)
            * grad_factor
        ),
        "bc_weight": (
            float(config.LAMBDA_BC)
            * bc_factor
        ),
    }


# ============================================================
# LOSS COMPONENT HELPER
# ============================================================

def get_component(
    output,
    possible_names,
    reference_tensor,
):
    """
    Safely extract a loss component.

    Different PEDVINOLoss implementations may use slightly
    different dictionary key names.
    """

    for name in possible_names:
        if name in output:
            value = output[name]

            if torch.is_tensor(value):
                return value

            return torch.tensor(
                float(value),
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

    return torch.zeros(
        (),
        device=reference_tensor.device,
        dtype=reference_tensor.dtype,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    epoch,
):
    """
    Train PEDVINO for one epoch.
    """

    model.train()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_weights = update_physics_weights(
        criterion=criterion,
        epoch=epoch,
    )

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    recon_loss_sum = 0.0
    energy_loss_sum = 0.0
    grad_loss_sum = 0.0
    bc_loss_sum = 0.0

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
        # Forward model
        # ----------------------------------------------------

        model_output = model(forcing)

        # PEDVINO is expected to return:
        #
        # prediction, reconstruction
        #
        prediction, reconstruction = model_output

        # ----------------------------------------------------
        # Composite loss
        #
        # IMPORTANT:
        #
        # Poisson forcing is passed as params because the
        # Poisson variational functional requires it.
        # ----------------------------------------------------

        loss_output = criterion(
            prediction=prediction,
            target=solution,
            reconstruction=reconstruction,
            input_field=forcing,
            dx=config.DX,
            dy=config.DY,
            spatial_dim=2,
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        if torch.is_tensor(loss_output):

            total_loss = loss_output

            pred_loss = torch.zeros_like(total_loss)
            recon_loss = torch.zeros_like(total_loss)
            energy_loss = torch.zeros_like(total_loss)
            grad_loss = torch.zeros_like(total_loss)
            bc_loss = torch.zeros_like(total_loss)

        else:

            total_loss = get_component(
                loss_output,
                [
                    "total_loss",
                    "loss",
                    "total",
                ],
                prediction,
            )

            pred_loss = get_component(
                loss_output,
                [
                    "prediction_loss",
                    "pred_loss",
                    "prediction",
                ],
                total_loss,
            )

            recon_loss = get_component(
                loss_output,
                [
                    "reconstruction_loss",
                    "recon_loss",
                    "reconstruction",
                ],
                total_loss,
            )

            energy_loss = get_component(
                loss_output,
                [
                    "energy_loss",
                    "variational_loss",
                    "physics_loss",
                    "energy",
                ],
                total_loss,
            )

            grad_loss = get_component(
                loss_output,
                [
                    "gradient_loss",
                    "grad_loss",
                    "gradient",
                ],
                total_loss,
            )

            bc_loss = get_component(
                loss_output,
                [
                    "boundary_loss",
                    "bc_loss",
                    "boundary",
                ],
                total_loss,
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        total_loss.backward()

        if (
            hasattr(config, "GRADIENT_CLIP")
            and config.GRADIENT_CLIP is not None
            and config.GRADIENT_CLIP > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.GRADIENT_CLIP,
            )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = forcing.shape[0]

        total_loss_sum += (
            total_loss.detach().item()
            * batch_size
        )

        pred_loss_sum += (
            pred_loss.detach().item()
            * batch_size
        )

        recon_loss_sum += (
            recon_loss.detach().item()
            * batch_size
        )

        energy_loss_sum += (
            energy_loss.detach().item()
            * batch_size
        )

        grad_loss_sum += (
            grad_loss.detach().item()
            * batch_size
        )

        bc_loss_sum += (
            bc_loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "total_loss":
            total_loss_sum / total_samples,

        "prediction_loss":
            pred_loss_sum / total_samples,

        "reconstruction_loss":
            recon_loss_sum / total_samples,

        "energy_loss":
            energy_loss_sum / total_samples,

        "gradient_loss":
            grad_loss_sum / total_samples,

        "boundary_loss":
            bc_loss_sum / total_samples,

        "energy_weight":
            warmup_weights["energy_weight"],

        "gradient_weight":
            warmup_weights["grad_weight"],

        "boundary_weight":
            warmup_weights["bc_weight"],
    }


# ============================================================
# SCHEDULER
# ============================================================

def build_scheduler(
    optimizer,
):
    """
    Build scheduler according to config.
    """

    scheduler_name = getattr(
        config,
        "SCHEDULER",
        "none",
    ).lower()

    if scheduler_name == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=config.MIN_LEARNING_RATE,
        )

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POISSON 2D - PROPOSED PEDVINO EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")

    # --------------------------------------------------------
    # Ensure PEDVINO directory exists
    # --------------------------------------------------------

    os.makedirs(
        config.PEDVINO_RESULTS_DIR,
        exist_ok=True,
    )

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
    # Model
    # --------------------------------------------------------

    model = build_pedvino_model().to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: PEDVINO")
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = build_loss().to(device)

    print("\nLoss configuration:")
    print(
        f"Prediction weight    : "
        f"{config.LAMBDA_PRED}"
    )
    print(
        f"Reconstruction weight: "
        f"{config.LAMBDA_RECON}"
    )
    print(
        f"Energy weight        : "
        f"{config.LAMBDA_ENERGY}"
    )
    print(
        f"Gradient weight      : "
        f"{config.LAMBDA_GRAD}"
    )
    print(
        f"Boundary weight      : "
        f"{config.LAMBDA_BC}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = build_scheduler(
        optimizer
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = initialize_history()

    best_val_l2 = float("inf")
    best_epoch = -1

    epochs_without_improvement = 0

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
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if scheduler is not None:
            scheduler.step()

        # ----------------------------------------------------
        # Store history
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

        current_val_l2 = val_metrics[
            "relative_l2"
        ]

        improved = (
            current_val_l2
            < best_val_l2
            - config.EARLY_STOPPING_MIN_DELTA
        )

        if improved:

            best_val_l2 = current_val_l2
            best_epoch = epoch

            epochs_without_improvement = 0

            if config.SAVE_CHECKPOINT:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        "train": train_metrics,
                        "validation": val_metrics,
                        "test": test_metrics,
                    },
                    path=config.PEDVINO_CHECKPOINT_PATH,
                )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Save history continuously
        #
        # This protects metrics if training stops unexpectedly.
        # ----------------------------------------------------

        if config.SAVE_HISTORY:

            save_json(
                history,
                config.PEDVINO_HISTORY_PATH,
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if (
            epoch % config.PRINT_EVERY == 0
        ):

            print(
                f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
                f"Train: "
                f"{train_metrics['total_loss']:.6e} | "
                f"Pred: "
                f"{train_metrics['prediction_loss']:.6e} | "
                f"Recon: "
                f"{train_metrics['reconstruction_loss']:.6e} | "
                f"Energy: "
                f"{train_metrics['energy_loss']:.6e} | "
                f"Grad: "
                f"{train_metrics['gradient_loss']:.6e} | "
                f"BC: "
                f"{train_metrics['boundary_loss']:.6e} | "
                f"Val L2: "
                f"{val_metrics['relative_l2']:.6e} | "
                f"Test L2: "
                f"{test_metrics['relative_l2']:.6e}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            config.EARLY_STOPPING
            and epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No validation improvement for "
                f"{config.EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            break

    # ========================================================
    # FINAL EVALUATION USING BEST CHECKPOINT
    # ========================================================

    if not os.path.exists(
        config.PEDVINO_CHECKPOINT_PATH
    ):

        raise RuntimeError(
            "PEDVINO best checkpoint was not created: "
            f"{config.PEDVINO_CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        config.PEDVINO_CHECKPOINT_PATH,
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
    # FINAL HISTORY SAVE
    # ========================================================

    if config.SAVE_HISTORY:

        save_json(
            history,
            config.PEDVINO_HISTORY_PATH,
        )

    # ========================================================
    # METRICS.JSON
    # ========================================================

    final_metrics = {

        "model": "PEDVINO",

        "experiment": "Poisson2D",

        "seed": config.SEED,

        "epochs_configured": config.EPOCHS,

        "epochs_completed": len(
            history.get("epoch", [])
        ),

        "batch_size": config.BATCH_SIZE,

        "learning_rate": config.LEARNING_RATE,

        "weight_decay": config.WEIGHT_DECAY,

        "scheduler": config.SCHEDULER,

        "trainable_parameters":
            num_parameters,

        "best_epoch":
            best_epoch,

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

        "loss_weights": {
            "prediction":
                config.LAMBDA_PRED,

            "reconstruction":
                config.LAMBDA_RECON,

            "energy":
                config.LAMBDA_ENERGY,

            "gradient":
                config.LAMBDA_GRAD,

            "boundary":
                config.LAMBDA_BC,
        },

        "warmup_epochs": {
            "energy":
                config.ENERGY_WARMUP_EPOCHS,

            "gradient":
                config.GRAD_WARMUP_EPOCHS,

            "boundary":
                config.BC_WARMUP_EPOCHS,
        },
    }

    metrics_path = os.path.join(
        config.PEDVINO_RESULTS_DIR,
        "metrics.json",
    )

    save_json(
        final_metrics,
        metrics_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "model": "PEDVINO",
        "best_epoch": best_epoch,
        "validation_relative_l2":
            final_val_metrics["relative_l2"],
        "test_relative_l2":
            final_test_metrics["relative_l2"],
        "test_mse":
            final_test_metrics["mse"],
        "trainable_parameters":
            num_parameters,
        "training_time_seconds":
            elapsed_time,
    }

    save_json(
        summary,
        config.PEDVINO_SUMMARY_PATH,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("PEDVINO TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best epoch                : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Relative L2: "
        f"{final_val_metrics['relative_l2']:.6e}"
    )

    print(
        f"Final Test Relative L2    : "
        f"{final_test_metrics['relative_l2']:.6e}"
    )

    print(
        f"Final Test MSE            : "
        f"{final_test_metrics['mse']:.6e}"
    )

    print(
        f"Trainable parameters      : "
        f"{num_parameters:,}"
    )

    print(
        f"Training time (seconds)   : "
        f"{elapsed_time:.2f}"
    )

    print("\nSaved files:")

    print(
        f"Checkpoint: "
        f"{config.PEDVINO_CHECKPOINT_PATH}"
    )

    print(
        f"History:    "
        f"{config.PEDVINO_HISTORY_PATH}"
    )

    print(
        f"Metrics:    "
        f"{metrics_path}"
    )

    print(
        f"Summary:    "
        f"{config.PEDVINO_SUMMARY_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
