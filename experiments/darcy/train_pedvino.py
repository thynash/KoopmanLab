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
# MODEL
# ============================================================

def build_pedvino_model():
    """
    Build PEDVINO with the same one-channel operator input
    convention as the baseline KNO experiment.

        coefficient a(x, y) -> solution u(x, y)

    The forcing f(x, y) is supplied separately to the Darcy
    variational functional and is NOT concatenated to the
    neural operator input.
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
# WARM-UP
# ============================================================

def get_warmup_factor(epoch, warmup_epochs):
    """
    Linear warm-up from zero to one.
    """

    if warmup_epochs is None or warmup_epochs <= 0:
        return 1.0

    return min(
        1.0,
        float(epoch) / float(warmup_epochs),
    )


def update_physics_weights(criterion, epoch):
    """
    Update energy, gradient and boundary weights.

    Uses the exact configuration names:

        ENERGY_WARMUP_EPOCHS
        GRADIENT_WARMUP_EPOCHS
        BOUNDARY_WARMUP_EPOCHS
    """

    energy_factor = get_warmup_factor(
        epoch,
        config.ENERGY_WARMUP_EPOCHS,
    )

    gradient_factor = get_warmup_factor(
        epoch,
        config.GRADIENT_WARMUP_EPOCHS,
    )

    boundary_factor = get_warmup_factor(
        epoch,
        config.BOUNDARY_WARMUP_EPOCHS,
    )

    criterion.lambda_energy = (
        float(config.LAMBDA_ENERGY)
        * energy_factor
    )

    criterion.lambda_grad = (
        float(config.LAMBDA_GRAD)
        * gradient_factor
    )

    criterion.lambda_bc = (
        float(config.LAMBDA_BC)
        * boundary_factor
    )

    return {
        "energy_weight": criterion.lambda_energy,
        "gradient_weight": criterion.lambda_grad,
        "boundary_weight": criterion.lambda_bc,
    }


# ============================================================
# LOSS HELPER
# ============================================================

def get_component(
    output,
    possible_names,
    reference_tensor,
):
    """
    Safely obtain a loss component from a PEDVINOLoss output.
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

    Darcy dataset batch:

        coefficient, solution, forcing

    Neural operator mapping:

        coefficient -> solution

    Darcy physics parameters:

        coefficient = a(x,y)
        forcing     = f(x,y)
        solution    = u(x,y)
    """

    model.train()

    warmup_weights = update_physics_weights(
        criterion,
        epoch,
    )

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    recon_loss_sum = 0.0
    energy_loss_sum = 0.0
    grad_loss_sum = 0.0
    bc_loss_sum = 0.0

    total_samples = 0

    for coefficient, solution, forcing in data_loader:

        coefficient = coefficient.to(
            device,
            non_blocking=True,
        )

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
        # MODEL INPUT
        #
        # IMPORTANT:
        #
        # Keep exactly the same one-channel mapping as the
        # baseline KNO:
        #
        #     a(x,y) -> u(x,y)
        #
        # Do NOT concatenate forcing.
        # ----------------------------------------------------

        model_input = coefficient

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        prediction, reconstruction = model(
            model_input
        )

        # ----------------------------------------------------
        # DARCY PDE PARAMETERS
        #
        # Both fields are available to the variational
        # functional without changing the neural architecture.
        # ----------------------------------------------------

        pde_params = {
            "coefficient": coefficient,
            "forcing": forcing,
        }

        # ----------------------------------------------------
        # PEDVINO LOSS
        # ----------------------------------------------------

        loss_output = criterion(
            prediction=prediction,
            target=solution,
            reconstruction=reconstruction,
            input_field=model_input,
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

            recon_loss = torch.zeros_like(
                total_loss
            )

            energy_loss = torch.zeros_like(
                total_loss
            )

            grad_loss = torch.zeros_like(
                total_loss
            )

            bc_loss = torch.zeros_like(
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
        # BACKPROPAGATION
        # ----------------------------------------------------

        total_loss.backward()

        if hasattr(config, "GRADIENT_CLIP"):

            if config.GRADIENT_CLIP is not None:

                if config.GRADIENT_CLIP > 0:

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.GRADIENT_CLIP,
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
            warmup_weights["gradient_weight"],

        "boundary_weight":
            warmup_weights["boundary_weight"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DARCY FLOW - PROPOSED PEDVINO EXPERIMENT")
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

    train_loader, val_loader, test_loader = (
        get_darcy_loaders(
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
    # MODEL
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
    # LOSS
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
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer_name = getattr(
        config,
        "OPTIMIZER",
        "adamw",
    ).lower()

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    else:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = None

    scheduler_name = getattr(
        config,
        "SCHEDULER",
        None,
    )

    if scheduler_name is not None:

        scheduler_name = scheduler_name.lower()

        if scheduler_name == "cosine":

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS,
                eta_min=getattr(
                    config,
                    "MIN_LEARNING_RATE",
                    1e-6,
                ),
            )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = initialize_history()

    best_val_l2 = float("inf")
    best_epoch = -1

    timer = ExperimentTimer()
    timer.start()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        config.EPOCHS + 1,
    ):

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Validation and test use generic evaluation.
        #
        # train_utils.evaluate handles Darcy batches of:
        #
        # coefficient, forcing, solution
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

        if val_metrics["relative_l2"] < best_val_l2:

            best_val_l2 = (
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            checkpoint_path = (
                config.PEDVINO_CHECKPOINT_PATH
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
        # SCHEDULER
        # ----------------------------------------------------

        if scheduler is not None:

            scheduler.step()

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
            f"Train: {train_metrics['total_loss']:.6e} | "
            f"Pred: {train_metrics['prediction_loss']:.6e} | "
            f"Recon: {train_metrics['reconstruction_loss']:.6e} | "
            f"Energy: {train_metrics['energy_loss']:.6e} | "
            f"Grad: {train_metrics['gradient_loss']:.6e} | "
            f"BC: {train_metrics['boundary_loss']:.6e} | "
            f"Val L2: {val_metrics['relative_l2']:.6e} | "
            f"Test L2: {test_metrics['relative_l2']:.6e} | "
            f"LR: {current_lr:.2e}"
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        config.PEDVINO_CHECKPOINT_PATH,
        map_location=device,
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

    if getattr(config, "SAVE_HISTORY", True):

        save_json(
            history,
            config.PEDVINO_HISTORY_PATH,
        )

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    final_metrics = {
        "model": "PEDVINO",
        "experiment": "Darcy Flow",

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

        "loss_weights": {
            "prediction": config.LAMBDA_PRED,
            "reconstruction": config.LAMBDA_RECON,
            "energy": config.LAMBDA_ENERGY,
            "gradient": config.LAMBDA_GRAD,
            "boundary": config.LAMBDA_BC,
        },

        "warmup_epochs": {
            "energy": config.ENERGY_WARMUP_EPOCHS,
            "gradient": config.GRADIENT_WARMUP_EPOCHS,
            "boundary": config.BOUNDARY_WARMUP_EPOCHS,
        },
    }

    if getattr(config, "SAVE_METRICS", True):

        save_json(
            final_metrics,
            config.PEDVINO_METRICS_PATH,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PEDVINO TRAINING COMPLETED")
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
        "Training time (seconds): "
        f"{elapsed_time:.2f}"
    )

    print(
        f"\nBest checkpoint: "
        f"{config.PEDVINO_CHECKPOINT_PATH}"
    )

    print(
        f"History: "
        f"{config.PEDVINO_HISTORY_PATH}"
    )

    print(
        f"Metrics: "
        f"{config.PEDVINO_METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
