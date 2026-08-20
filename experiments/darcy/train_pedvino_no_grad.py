import os
import torch

from koopmanlab.model_pedvino import PEDVINO
from koopmanlab.pde_functionals import DarcyFunctional
from koopmanlab.variational_loss import GeneralVariationalLoss
from koopmanlab.pedvino_loss import PEDVINOLoss

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
# ABLATION SETTINGS
# ============================================================

EXPERIMENT_NAME = "PEDVINO_NoGrad"

NUM_EPOCHS = 150

CHECKPOINT_PATH = os.path.join(
    config.PEDVINO_RESULTS_DIR,
    "no_grad_best_model.pt",
)

HISTORY_PATH = os.path.join(
    config.PEDVINO_RESULTS_DIR,
    "no_grad_history.json",
)

METRICS_PATH = os.path.join(
    config.PEDVINO_RESULTS_DIR,
    "no_grad_metrics.json",
)


# ============================================================
# MODEL
# ============================================================

def build_pedvino_model():
    """
    Same PEDVINO architecture as the main Darcy experiment.

    Operator mapping:

        coefficient a(x,y) -> solution u(x,y)

    The architecture is unchanged.

    This is a LOSS ABLATION only.
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
    """
    Darcy PEDVINO ablation loss.

    ONLY:

        prediction/data loss
        +
        variational energy loss

    REMOVED:

        reconstruction loss
        gradient loss
        boundary loss
    """

    functional = DarcyFunctional()

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

        # ----------------------------------------------------
        # KEEP
        # ----------------------------------------------------

        lambda_pred=config.LAMBDA_PRED,
        lambda_energy=config.LAMBDA_ENERGY,

        # ----------------------------------------------------
        # REMOVE
        # ----------------------------------------------------

        lambda_recon=0.0,
        lambda_grad=0.0,
        lambda_bc=0.0,

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
# LOSS COMPONENT HELPER
# ============================================================

def get_component(
    output,
    possible_names,
    reference_tensor,
):
    """
    Safely extract a loss component from PEDVINOLoss output.
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
):
    """
    Train one epoch for the Darcy no-gradient ablation.

    Dataset:

        coefficient, solution, forcing

    Neural operator:

        coefficient -> solution

    Training objective:

        L =
            lambda_pred   * prediction_loss
          + lambda_energy * variational_energy_loss

    Reconstruction, gradient and boundary losses are disabled.
    """

    model.train()

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    energy_loss_sum = 0.0

    total_samples = 0

    for coefficient, solution, forcing in data_loader:

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ZERO GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        prediction, reconstruction = model(
            coefficient
        )

        # ----------------------------------------------------
        # PDE PARAMETERS
        # ----------------------------------------------------

        pde_params = {
            "coefficient": coefficient,
            "forcing": forcing,
        }

        # ----------------------------------------------------
        # PEDVINO LOSS
        #
        # The criterion has:
        #
        # lambda_recon = 0
        # lambda_grad  = 0
        # lambda_bc    = 0
        # ----------------------------------------------------

        loss_output = criterion(
            prediction=prediction,
            target=solution,
            reconstruction=reconstruction,
            input_field=coefficient,
            params=pde_params,
            spatial_dim=2,
            dx=config.DX,
            dy=config.DY,
        )

        # ----------------------------------------------------
        # EXTRACT LOSSES
        # ----------------------------------------------------

        if torch.is_tensor(loss_output):

            total_loss = loss_output

            pred_loss = torch.zeros_like(
                total_loss
            )

            energy_loss = torch.zeros_like(
                total_loss
            )

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

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        total_loss.backward()

        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        gradient_clip = getattr(
            config,
            "GRADIENT_CLIP",
            None,
        )

        if (
            gradient_clip is not None
            and gradient_clip > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

        optimizer.step()

        # ----------------------------------------------------
        # ACCUMULATE
        # ----------------------------------------------------

        batch_size = coefficient.shape[0]

        total_loss_sum += (
            total_loss.detach().item()
            * batch_size
        )

        pred_loss_sum += (
            pred_loss.detach().item()
            * batch_size
        )

        energy_loss_sum += (
            energy_loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

    # --------------------------------------------------------
    # EPOCH METRICS
    # --------------------------------------------------------

    total_samples = max(
        total_samples,
        1,
    )

    return {
        "total_loss":
            total_loss_sum / total_samples,

        "prediction_loss":
            pred_loss_sum / total_samples,

        "energy_loss":
            energy_loss_sum / total_samples,

        # Explicitly record ablated components
        "reconstruction_loss": 0.0,
        "gradient_loss": 0.0,
        "boundary_loss": 0.0,

        "energy_weight":
            float(config.LAMBDA_ENERGY),

        "gradient_weight": 0.0,

        "boundary_weight": 0.0,
    }


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model):

    optimizer_name = getattr(
        config,
        "OPTIMIZER",
        "adamw",
    ).lower()

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


# ============================================================
# SCHEDULER
# ============================================================

def build_scheduler(
    optimizer,
    epochs,
):

    scheduler_name = getattr(
        config,
        "SCHEDULER",
        "none",
    ).lower()

    if scheduler_name == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=getattr(
                config,
                "MIN_LEARNING_RATE",
                1e-6,
            ),
        )

    if scheduler_name == "none":

        return None

    raise ValueError(
        f"Unsupported scheduler: {scheduler_name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY FLOW - PEDVINO NO-GRADIENT ABLATION")
    print("=" * 70)

    print(
        "\nObjective:"
    )

    print(
        "L = lambda_pred * L_prediction "
        "+ lambda_energy * L_variational"
    )

    print("\nDisabled losses:")
    print("  Reconstruction loss : DISABLED")
    print("  Gradient loss       : DISABLED")
    print("  Boundary loss       : DISABLED")

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(config.SEED)

    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Seed: {config.SEED}")
    print(f"Epochs: {NUM_EPOCHS}")

    # ========================================================
    # RESULTS DIRECTORY
    # ========================================================

    os.makedirs(
        config.PEDVINO_RESULTS_DIR,
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
    print(
        f"Train: {len(train_loader.dataset)}"
    )
    print(
        f"Val  : {len(val_loader.dataset)}"
    )
    print(
        f"Test : {len(test_loader.dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = build_pedvino_model().to(
        device
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: PEDVINO No-Gradient Ablation")
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = build_loss().to(
        device
    )

    print("\nLoss configuration:")
    print(
        f"Prediction weight    : "
        f"{config.LAMBDA_PRED}"
    )
    print(
        f"Variational weight   : "
        f"{config.LAMBDA_ENERGY}"
    )
    print(
        "Reconstruction weight: 0.0"
    )
    print(
        "Gradient weight      : 0.0"
    )
    print(
        "Boundary weight      : 0.0"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = build_optimizer(
        model
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
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
        NUM_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
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
        # ----------------------------------------------------

        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        append_history(
            history=history,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
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
                path=CHECKPOINT_PATH,
            )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        if scheduler is not None:

            scheduler.step()

        # ----------------------------------------------------
        # CURRENT LR
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        if (
            epoch % getattr(
                config,
                "PRINT_EVERY",
                1,
            ) == 0
        ):

            print(
                f"Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
                f"Train: {train_metrics['total_loss']:.6e} | "
                f"Pred: {train_metrics['prediction_loss']:.6e} | "
                f"Energy: {train_metrics['energy_loss']:.6e} | "
                f"Val L2: {val_metrics['relative_l2']:.6e} | "
                f"Test L2: {test_metrics['relative_l2']:.6e} | "
                f"LR: {current_lr:.2e}"
            )

    # ========================================================
    # CHECKPOINT SAFETY
    # ========================================================

    if best_epoch < 0:

        raise RuntimeError(
            "No best checkpoint was saved."
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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
            HISTORY_PATH,
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_metrics = {

        # ----------------------------------------------------
        # EXPERIMENT
        # ----------------------------------------------------

        "model": "PEDVINO",
        "experiment": "DarcyFlow",
        "variant": "no_gradient_ablation",

        "description": (
            "PEDVINO Darcy ablation using only "
            "prediction/data loss and variational energy loss. "
            "Reconstruction, gradient and boundary losses are disabled."
        ),

        "seed": config.SEED,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        "epochs_configured": NUM_EPOCHS,

        "epochs_completed":
            len(history.get("epoch", [])),

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
        # MODEL
        # ----------------------------------------------------

        "trainable_parameters":
            num_parameters,

        # ----------------------------------------------------
        # LOSS ABLATION
        # ----------------------------------------------------

        "loss_weights": {
            "prediction":
                float(config.LAMBDA_PRED),

            "variational_energy":
                float(config.LAMBDA_ENERGY),

            "reconstruction":
                0.0,

            "gradient":
                0.0,

            "boundary":
                0.0,
        },

        # ----------------------------------------------------
        # BEST MODEL
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
        # TEST
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
        # TIME
        # ----------------------------------------------------

        "training_time_seconds":
            float(elapsed_time),
    }

    # ========================================================
    # SAVE METRICS
    # ========================================================

    if getattr(
        config,
        "SAVE_METRICS",
        True,
    ):

        save_json(
            final_metrics,
            METRICS_PATH,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PEDVINO NO-GRADIENT ABLATION COMPLETED")
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

    print("\nAblation objective:")
    print(
        "Prediction/Data + Variational Energy ONLY"
    )

    print("\nDisabled:")
    print("  Reconstruction loss")
    print("  Gradient loss")
    print("  Boundary loss")

    print("\nSaved files:")

    print(
        f"Best checkpoint : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"History         : "
        f"{HISTORY_PATH}"
    )

    print(
        f"Metrics         : "
        f"{METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
