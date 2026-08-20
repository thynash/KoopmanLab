"""
PEDVINO experiment for the 1D Allen-Cahn equation.

Learning task:

    u(x, 0) -> u(x, T)

Allen-Cahn equation:

    u_t = epsilon * u_xx - (u^3 - u)

or equivalently:

    u_t - epsilon * u_xx + u^3 - u = 0

PEDVINO uses:

    1. Supervised prediction loss
    2. Reconstruction loss
    3. Physics / variational energy consistency loss
    4. Gradient loss
    5. Boundary loss

IMPORTANT FOR TIME-DEPENDENT PDE:

The AllenCahnFunctional requires u_t.

Since this experiment predicts the final state from the initial
state, we approximate:

    u_t ~= (u_pred - u_initial) / dt

The initial state is therefore passed to the variational functional
as:

    previous_state

and dt is passed as:

    dt
"""

import os
import torch

from koopmanlab.model_pedvino import PEDVINO
from koopmanlab.pde_functionals import AllenCahnFunctional
from koopmanlab.variational_loss import GeneralVariationalLoss
from koopmanlab.pedvino_loss import PEDVINOLoss

from experiments.allen_cahn import config

from experiments.allen_cahn.train_utils import (
    set_seed,
    get_device,
    get_allen_cahn_loaders,
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
    Build PEDVINO with KNO1d backbone.

    Operator learning task:

        u(x, 0) -> u(x, T)

    The architecture remains directly comparable with the
    baseline KNO1d experiment.
    """

    model = PEDVINO(
        backbone="KNO1d",

        t_len=config.T_LEN,

        operator_size=config.OPERATOR_SIZE,

        modes_x=config.MODES_X,

        decompose=config.DECOMPOSE,

        linear_type=config.LINEAR_TYPE,

        normalization=config.NORMALIZATION,

        hidden_size=config.PHYSICS_HIDDEN_SIZE,

        dx=config.DX,
    )

    return model


# ============================================================
# TIME STEP
# ============================================================

def get_time_step():
    """
    Obtain the effective temporal interval used for the
    Allen-Cahn functional.

    Priority:

        1. config.DT
        2. config.TIME_STEP
        3. config.FINAL_TIME / config.T_LEN
        4. config.FINAL_TIME
        5. 1.0

    IMPORTANT:

    If your generated dataset maps:

        u(x,0) -> u(x,T)

    directly in one operator step, the correct dt should be the
    total interval T represented by the input/output pair.

    If your dataset/config already defines DT explicitly, that
    value is used.
    """

    if hasattr(config, "DT"):

        dt = float(config.DT)

        if dt > 0:
            return dt

    if hasattr(config, "TIME_STEP"):

        dt = float(config.TIME_STEP)

        if dt > 0:
            return dt

    if (
        hasattr(config, "FINAL_TIME")
        and hasattr(config, "T_LEN")
    ):

        final_time = float(config.FINAL_TIME)
        t_len = float(config.T_LEN)

        if final_time > 0 and t_len > 0:
            return final_time / t_len

    if hasattr(config, "FINAL_TIME"):

        final_time = float(config.FINAL_TIME)

        if final_time > 0:
            return final_time

    return 1.0


# ============================================================
# PDE PARAMETERS
# ============================================================

def build_pde_params(
    initial_state,
):
    """
    Build parameters required by AllenCahnFunctional.

    The functional needs the previous state to compute:

        u_t = (u - previous_state) / dt

    The Allen-Cahn diffusion coefficient is supplied if the
    config defines one.

    We provide both common parameter names where available.
    The functional will use the one implemented in its code.
    """

    pde_params = {}

    # --------------------------------------------------------
    # Required for temporal derivative
    # --------------------------------------------------------

    pde_params["previous_state"] = initial_state

    # --------------------------------------------------------
    # Effective time interval
    # --------------------------------------------------------

    pde_params["dt"] = get_time_step()

    # --------------------------------------------------------
    # Allen-Cahn diffusion parameter
    # --------------------------------------------------------

    if hasattr(config, "EPSILON"):

        pde_params["epsilon"] = float(
            config.EPSILON
        )

    elif hasattr(config, "NU"):

        pde_params["nu"] = float(
            config.NU
        )

    elif hasattr(config, "DIFFUSION"):

        pde_params["diffusion"] = float(
            config.DIFFUSION
        )

    return pde_params


# ============================================================
# LOSS
# ============================================================

def build_loss():
    """
    Build PEDVINO loss for Allen-Cahn.

    The GeneralVariationalLoss uses the AllenCahnFunctional.

    The temporal derivative is computed by the functional using:

        previous_state
        dt

    passed through the PDE parameters.
    """

    functional = AllenCahnFunctional()

    variational_loss = GeneralVariationalLoss(
        functional=functional,

        spatial_dim=1,

        dx=config.DX,

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

def get_warmup_factor(
    epoch,
    warmup_epochs,
):
    """
    Linear warm-up from zero to one.
    """

    if (
        warmup_epochs is None
        or warmup_epochs <= 0
    ):
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
    Update physics weights using linear warm-up.
    """

    energy_warmup = getattr(
        config,
        "ENERGY_WARMUP_EPOCHS",
        0,
    )

    gradient_warmup = getattr(
        config,
        "GRADIENT_WARMUP_EPOCHS",
        getattr(
            config,
            "GRAD_WARMUP_EPOCHS",
            0,
        ),
    )

    boundary_warmup = getattr(
        config,
        "BOUNDARY_WARMUP_EPOCHS",
        getattr(
            config,
            "BC_WARMUP_EPOCHS",
            0,
        ),
    )

    energy_factor = get_warmup_factor(
        epoch,
        energy_warmup,
    )

    gradient_factor = get_warmup_factor(
        epoch,
        gradient_warmup,
    )

    boundary_factor = get_warmup_factor(
        epoch,
        boundary_warmup,
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
        "energy_weight":
            criterion.lambda_energy,

        "gradient_weight":
            criterion.lambda_grad,

        "boundary_weight":
            criterion.lambda_bc,
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
    Safely obtain a loss component.
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

    Dataset format:

        initial_state, solution

    Operator:

        u(x,0) -> u(x,T)

    Physics:

        u_t ~= (u_pred - u_initial) / dt
    """

    model.train()

    # --------------------------------------------------------
    # Physics warm-up
    # --------------------------------------------------------

    warmup_weights = update_physics_weights(
        criterion,
        epoch,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    recon_loss_sum = 0.0
    energy_loss_sum = 0.0
    grad_loss_sum = 0.0
    bc_loss_sum = 0.0

    total_samples = 0

    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch in data_loader:

        # ----------------------------------------------------
        # Dataset:
        #
        # u0 = u(x,0)
        # uT = u(x,T)
        # ----------------------------------------------------

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
        # Zero gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ----------------------------------------------------
        # Model input
        # ----------------------------------------------------

        model_input = initial_state

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction, reconstruction = model(
            model_input
        )

        # ----------------------------------------------------
        # PDE parameters
        #
        # This is the critical fix.
        #
        # AllenCahnFunctional requires previous_state to
        # construct u_t.
        # ----------------------------------------------------

        pde_params = build_pde_params(
            initial_state=model_input,
        )

        # ----------------------------------------------------
        # PEDVINO loss
        # ----------------------------------------------------

        loss_output = criterion(
            prediction=prediction,

            target=solution,

            reconstruction=reconstruction,

            input_field=model_input,

            params=pde_params,

            spatial_dim=1,

            dx=config.DX,
        )

        # ----------------------------------------------------
        # Extract loss components
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
        # Safety check
        # ----------------------------------------------------

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                f"Non-finite PEDVINO loss at epoch "
                f"{epoch}: {total_loss.item()}"
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
        # Accumulate statistics
        # ----------------------------------------------------

        batch_size = initial_state.shape[0]

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

    # ========================================================
    # EPOCH METRICS
    # ========================================================

    total_samples = max(
        total_samples,
        1,
    )

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
# OPTIMIZER
# ============================================================

def build_optimizer(
    model,
):
    """
    Build optimizer.
    """

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
):
    """
    Build learning-rate scheduler.
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
    print("ALLEN-CAHN - PROPOSED PEDVINO EXPERIMENT")
    print("=" * 70)

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(
        config.SEED
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Seed: {config.SEED}"
    )

    # ========================================================
    # RESULT DIRECTORY
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
    ) = get_allen_cahn_loaders(
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

    print(
        "\nModel: PEDVINO-KNO1d"
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # ========================================================
    # TIME STEP
    # ========================================================

    effective_dt = get_time_step()

    print(
        f"\nAllen-Cahn physics dt: "
        f"{effective_dt:.6e}"
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
        # Training
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
        # Save history
        # ----------------------------------------------------

        append_history(
            history,
            epoch,
            train_metrics,
            val_metrics,
            test_metrics,
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
                path=config.PEDVINO_CHECKPOINT_PATH,
            )

        # ----------------------------------------------------
        # Scheduler
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

        if epoch % config.PRINT_EVERY == 0:

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
                f"LR: {current_lr:.3e}"
            )

    # ========================================================
    # CHECKPOINT SAFETY
    # ========================================================

    if best_epoch < 0:

        raise RuntimeError(
            "No best checkpoint was saved."
        )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        config.PEDVINO_CHECKPOINT_PATH,
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
            config.PEDVINO_HISTORY_PATH,
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_metrics = {

        "model":
            "PEDVINO-KNO1d",

        "experiment":
            "Allen-Cahn",

        "seed":
            config.SEED,

        "epochs_configured":
            config.EPOCHS,

        "epochs_completed":
            len(
                history.get(
                    "epoch",
                    [],
                )
            ),

        "batch_size":
            config.BATCH_SIZE,

        "learning_rate":
            config.LEARNING_RATE,

        "weight_decay":
            config.WEIGHT_DECAY,

        "optimizer":
            getattr(
                config,
                "OPTIMIZER",
                "adamw",
            ),

        "scheduler":
            getattr(
                config,
                "SCHEDULER",
                "none",
            ),

        "trainable_parameters":
            num_parameters,

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

        "training_time_seconds":
            float(elapsed_time),

        "allen_cahn_dt":
            float(effective_dt),

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
            config.PEDVINO_METRICS_PATH,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "ALLEN-CAHN PEDVINO TRAINING COMPLETED"
    )

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
        f"{config.PEDVINO_CHECKPOINT_PATH}"
    )

    print(
        f"History         : "
        f"{config.PEDVINO_HISTORY_PATH}"
    )

    print(
        f"Metrics         : "
        f"{config.PEDVINO_METRICS_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
