import torch

from koopmanlab.model_pedvino import PEDVINO
from koopmanlab.pde_functionals import BurgersFunctional
from koopmanlab.variational_loss import GeneralVariationalLoss
from koopmanlab.pedvino_loss import PEDVINOLoss

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

def build_pedvino_model():
    """
    Build the full PEDVINO-KNO1d model.

        u(x, 0)
            ↓
        Physics Encoder
            ↓
        KNO1d Koopman Core
            ↓
        Physics Decoder
            ↓
        u_hat(x, T)
    """

    return PEDVINO(
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


# ============================================================
# LOSS
# ============================================================

def build_loss():
    """
    Build the PEDVINO loss for 1D viscous Burgers.

        u_t + u u_x - nu u_xx = 0

    Functional density:

        1/2 (u_t + u u_x - nu u_xx)^2
    """

    functional = BurgersFunctional()

    variational_loss = GeneralVariationalLoss(
        functional=functional,
        spatial_dim=1,
        dx=config.DX,
        dt=config.DT,
        reduction="none",
    )

    # IMPORTANT:
    # Do NOT pass eps here.
    # The PEDVINOLoss currently installed in the runtime
    # does not accept it.
    criterion = PEDVINOLoss(
        variational_loss=variational_loss,
        lambda_pred=config.LAMBDA_PRED,
        lambda_recon=config.LAMBDA_RECON,
        lambda_energy=config.LAMBDA_ENERGY,
        lambda_grad=config.LAMBDA_GRAD,
        lambda_bc=config.LAMBDA_BC,
        use_relative_prediction_loss=(
            config.USE_RELATIVE_PREDICTION_LOSS
        ),
    )

    return criterion


# ============================================================
# PHYSICS WARM-UP
# ============================================================

def get_warmup_factor(epoch, warmup_epochs):

    if warmup_epochs is None or warmup_epochs <= 0:
        return 1.0

    return min(
        1.0,
        float(epoch) / float(warmup_epochs),
    )


def get_grad_warmup_epochs():
    """
    Support both possible config naming conventions.
    """

    if hasattr(config, "GRADIENT_WARMUP_EPOCHS"):
        return config.GRADIENT_WARMUP_EPOCHS

    if hasattr(config, "GRAD_WARMUP_EPOCHS"):
        return config.GRAD_WARMUP_EPOCHS

    if hasattr(config, "PHYSICS_WARMUP_EPOCHS"):
        return config.PHYSICS_WARMUP_EPOCHS

    return 0


def get_bc_warmup_epochs():

    if hasattr(config, "BOUNDARY_WARMUP_EPOCHS"):
        return config.BOUNDARY_WARMUP_EPOCHS

    if hasattr(config, "BC_WARMUP_EPOCHS"):
        return config.BC_WARMUP_EPOCHS

    if hasattr(config, "PHYSICS_WARMUP_EPOCHS"):
        return config.PHYSICS_WARMUP_EPOCHS

    return 0


def get_energy_warmup_epochs():

    if hasattr(config, "ENERGY_WARMUP_EPOCHS"):
        return config.ENERGY_WARMUP_EPOCHS

    if hasattr(config, "PHYSICS_WARMUP_EPOCHS"):
        return config.PHYSICS_WARMUP_EPOCHS

    return 0


def update_physics_weights(criterion, epoch):

    energy_factor = get_warmup_factor(
        epoch,
        get_energy_warmup_epochs(),
    )

    grad_factor = get_warmup_factor(
        epoch,
        get_grad_warmup_epochs(),
    )

    bc_factor = get_warmup_factor(
        epoch,
        get_bc_warmup_epochs(),
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
        "gradient_weight": (
            float(config.LAMBDA_GRAD)
            * grad_factor
        ),
        "boundary_weight": (
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

    return reference_tensor.new_zeros(())


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

    model.train()

    warmup_weights = update_physics_weights(
        criterion,
        epoch,
    )

    total_loss_sum = 0.0
    prediction_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    energy_loss_sum = 0.0
    gradient_loss_sum = 0.0
    boundary_loss_sum = 0.0
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

        # ====================================================
        # FORWARD
        # ====================================================

        output = model(initial_state)

        if isinstance(output, tuple):

            if len(output) < 2:
                raise RuntimeError(
                    "PEDVINO model tuple output must contain "
                    "prediction and reconstruction."
                )

            prediction = output[0]
            reconstruction = output[1]

        else:
            raise RuntimeError(
                "PEDVINO model is expected to return "
                "(prediction, reconstruction)."
            )

        if prediction.shape != solution.shape:
            raise RuntimeError(
                "Prediction and solution shapes do not match. "
                f"Prediction={tuple(prediction.shape)}, "
                f"Solution={tuple(solution.shape)}"
            )

        # ====================================================
        # FULL PEDVINO LOSS
        #
        # u_t ≈ [u_hat(x,T) - u(x,0)] / DT
        #
        # previous_state is the initial condition.
        # ====================================================

        loss_output = criterion(
            prediction=prediction,
            target=solution,
            reconstruction=reconstruction,
            input_field=initial_state,
            spatial_dim=1,
            previous_state=initial_state,
            params={
                "nu": config.VISCOSITY,
            },
            dx=config.DX,
        )

        # ====================================================
        # EXTRACT COMPONENTS
        # ====================================================

        total_loss = get_component(
            loss_output,
            ["total_loss", "total"],
            prediction,
        )

        prediction_loss = get_component(
            loss_output,
            ["prediction_loss", "pred_loss"],
            prediction,
        )

        reconstruction_loss = get_component(
            loss_output,
            ["reconstruction_loss", "recon_loss"],
            prediction,
        )

        energy_loss = get_component(
            loss_output,
            ["energy_loss", "variational_loss"],
            prediction,
        )

        gradient_loss = get_component(
            loss_output,
            ["gradient_loss", "grad_loss"],
            prediction,
        )

        boundary_loss = get_component(
            loss_output,
            ["boundary_loss", "bc_loss"],
            prediction,
        )

        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite PEDVINO loss encountered.\n"
                f"Epoch={epoch}\n"
                f"Total={total_loss.detach().item()}\n"
                f"Prediction={prediction_loss.detach().item()}\n"
                f"Reconstruction={reconstruction_loss.detach().item()}\n"
                f"Energy={energy_loss.detach().item()}\n"
                f"Gradient={gradient_loss.detach().item()}\n"
                f"Boundary={boundary_loss.detach().item()}"
            )

        # ====================================================
        # BACKPROP
        # ====================================================

        total_loss.backward()

        if (
            config.GRADIENT_CLIP is not None
            and config.GRADIENT_CLIP > 0.0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.GRADIENT_CLIP,
            )

        optimizer.step()

        # ====================================================
        # METRICS
        # ====================================================

        batch_size = initial_state.shape[0]

        batch_relative_l2 = relative_l2_error(
            prediction.detach(),
            solution,
            eps=getattr(config, "LOSS_EPS", 1e-8),
        )

        total_loss_sum += (
            total_loss.detach().item() * batch_size
        )

        prediction_loss_sum += (
            prediction_loss.detach().item() * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.detach().item() * batch_size
        )

        energy_loss_sum += (
            energy_loss.detach().item() * batch_size
        )

        gradient_loss_sum += (
            gradient_loss.detach().item() * batch_size
        )

        boundary_loss_sum += (
            boundary_loss.detach().item() * batch_size
        )

        relative_l2_sum += (
            batch_relative_l2.detach().item() * batch_size
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

        "energy_loss":
            energy_loss_sum / total_samples,

        "gradient_loss":
            gradient_loss_sum / total_samples,

        "boundary_loss":
            boundary_loss_sum / total_samples,

        "relative_l2":
            relative_l2_sum / total_samples,

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
    print("BURGERS 1D - FULL PEDVINO EXPERIMENT")
    print("=" * 70)

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")

    # ========================================================
    # DATA
    # ========================================================

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
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val  : {len(val_loader.dataset)}")
    print(f"Test : {len(test_loader.dataset)}")

    # ========================================================
    # MODEL
    # ========================================================

    model = build_pedvino_model().to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel: Full PEDVINO-KNO1d")
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = build_loss().to(device)

    print("\nLoss configuration:")
    print(f"Prediction     : {config.LAMBDA_PRED}")
    print(f"Reconstruction : {config.LAMBDA_RECON}")
    print(f"Energy         : {config.LAMBDA_ENERGY}")
    print(f"Gradient       : {config.LAMBDA_GRAD}")
    print(f"Boundary       : {config.LAMBDA_BC}")

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = None

    if getattr(
        config,
        "SCHEDULER",
        ""
    ).lower() == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=config.MIN_LEARNING_RATE,
        )

    # ========================================================
    # HISTORY
    # ========================================================

    history = initialize_history()

    history["energy_loss"] = []
    history["gradient_loss"] = []
    history["boundary_loss"] = []

    history["energy_weight"] = []
    history["gradient_weight"] = []
    history["boundary_weight"] = []

    history["test_relative_l2"] = []
    history["test_mse"] = []

    # ========================================================
    # CHECKPOINT TRACKING
    # ========================================================

    best_val_l2 = float("inf")
    best_epoch = -1

    early_stopping = None

    if config.EARLY_STOPPING:

        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
        )

    # ========================================================
    # TIMER
    # ========================================================

    timer = ExperimentTimer()
    timer.start()

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(1, config.EPOCHS + 1):

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
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

        current_learning_rate = get_learning_rate(
            optimizer
        )

        # ====================================================
        # STANDARD HISTORY
        # ====================================================

        append_history(
            history=history,
            epoch=epoch,
            train_loss=train_metrics["total_loss"],
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

        # ====================================================
        # PEDVINO HISTORY
        # ====================================================

        history["energy_loss"].append(
            float(train_metrics["energy_loss"])
        )

        history["gradient_loss"].append(
            float(train_metrics["gradient_loss"])
        )

        history["boundary_loss"].append(
            float(train_metrics["boundary_loss"])
        )

        history["energy_weight"].append(
            float(train_metrics["energy_weight"])
        )

        history["gradient_weight"].append(
            float(train_metrics["gradient_weight"])
        )

        history["boundary_weight"].append(
            float(train_metrics["boundary_weight"])
        )

        history["test_relative_l2"].append(
            float(test_metrics["relative_l2"])
        )

        history["test_mse"].append(
            float(test_metrics["mse"])
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_metrics["relative_l2"] < best_val_l2:

            best_val_l2 = float(
                val_metrics["relative_l2"]
            )

            best_epoch = epoch

            save_checkpoint(
                path=config.PEDVINO_CHECKPOINT_PATH,
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
                    "experiment": "burgers_pedvino",
                    "model": "PEDVINO-KNO1d",
                    "functional": "BurgersFunctional",
                    "viscosity": float(config.VISCOSITY),
                },
            )

        # ====================================================
        # SCHEDULER
        # ====================================================

        if scheduler is not None:
            scheduler.step()

        # ====================================================
        # OUTPUT
        # ====================================================

        print(
            f"Epoch [{epoch:03d}/{config.EPOCHS}] | "
            f"Loss: {train_metrics['total_loss']:.6e} | "
            f"Pred: {train_metrics['prediction_loss']:.6e} | "
            f"Recon: {train_metrics['reconstruction_loss']:.6e} | "
            f"Energy: {train_metrics['energy_loss']:.6e} | "
            f"Grad: {train_metrics['gradient_loss']:.6e} | "
            f"BC: {train_metrics['boundary_loss']:.6e} | "
            f"Train L2: {train_metrics['relative_l2']:.6e} | "
            f"Val L2: {val_metrics['relative_l2']:.6e} | "
            f"Test L2: {test_metrics['relative_l2']:.6e} | "
            f"LR: {current_learning_rate:.3e}"
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if early_stopping is not None:

            early_stopping.step(
                val_metrics["relative_l2"]
            )

            if early_stopping.should_stop:

                print("\nEarly stopping triggered.")
                break

    # ========================================================
    # END TRAINING
    # ========================================================

    timer.stop()

    print("\n" + "=" * 70)
    print("PEDVINO TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Training time: "
        f"{timer.elapsed_minutes:.2f} minutes"
    )

    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation L2: "
        f"{best_val_l2:.6e}"
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    load_checkpoint(
        path=config.PEDVINO_CHECKPOINT_PATH,
        model=model,
        device=device,
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_train_metrics = evaluate(
        model,
        train_loader,
        device,
    )

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    final_test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    print("\n" + "=" * 70)
    print("BEST PEDVINO CHECKPOINT - FINAL RESULTS")
    print("=" * 70)

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
            config.PEDVINO_HISTORY_PATH,
        )

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    final_metrics = {
        "experiment": "Burgers 1D",
        "model": "PEDVINO-KNO1d",
        "functional": "BurgersFunctional",
        "pde": "u_t + u*u_x - nu*u_xx = 0",
        "viscosity": float(config.VISCOSITY),
        "best_epoch": int(best_epoch),
        "training_time_minutes": float(
            timer.elapsed_minutes
        ),
        "train": {
            key: float(value)
            for key, value in final_train_metrics.items()
            if key != "num_samples"
        },
        "validation": {
            key: float(value)
            for key, value in final_val_metrics.items()
            if key != "num_samples"
        },
        "test": {
            key: float(value)
            for key, value in final_test_metrics.items()
            if key != "num_samples"
        },
    }

    if hasattr(config, "SAVE_METRICS"):

        if config.SAVE_METRICS:

            save_json(
                final_metrics,
                config.PEDVINO_METRICS_PATH,
            )

    print("\nSaved:")
    print(
        f"Checkpoint: "
        f"{config.PEDVINO_CHECKPOINT_PATH}"
    )

    print(
        f"History: "
        f"{config.PEDVINO_HISTORY_PATH}"
    )

    if hasattr(config, "PEDVINO_METRICS_PATH"):
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
