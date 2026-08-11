# ============================================================
# demo_variational.py
#
# Variational KNO experiment for 2D Navier-Stokes
# viscosity: nu = 1e-5
#
# No results are saved.
# Everything is displayed directly.
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

from model_variational import koopman


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VARIATIONAL KNO - NAVIER-STOKES")
print("=" * 70)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. DATASET
# ============================================================
#
# IMPORTANT:
# Replace ONLY this section with the SAME data-loading code
# used by your current working demo.py.
#
# It must produce:
#
#     trainloader
#     testloader
#
# Expected format:
#
#     xx : [B, Nx, Ny, T_in]
#     yy : [B, Nx, Ny, T_out]
#
# Do not change your existing preprocessing here.
# ============================================================

# ------------------------------------------------------------
# Example:
#
# from dataset import ...
# from loader import ...
#
# trainloader, testloader = ...
# ------------------------------------------------------------


# Safety check
if "trainloader" not in globals():
    raise RuntimeError(
        "\ntrainloader was not created.\n"
        "Copy the dataset-loading section from your existing "
        "working demo.py into Section 2 of this file."
    )

if "testloader" not in globals():
    raise RuntimeError(
        "\ntestloader was not created.\n"
        "Copy the dataset-loading section from your existing "
        "working demo.py into Section 2 of this file."
    )


# ============================================================
# 3. INSPECT DATA
# ============================================================

sample_x, sample_y = next(iter(trainloader))

print("\n" + "-" * 70)
print("DATASET")
print("-" * 70)

print("Input shape :", tuple(sample_x.shape))
print("Target shape:", tuple(sample_y.shape))

print(
    "Input range :",
    float(sample_x.min()),
    "to",
    float(sample_x.max())
)

print(
    "Target range:",
    float(sample_y.min()),
    "to",
    float(sample_y.max())
)


# ============================================================
# 4. AUTOMATICALLY DETERMINE SPATIAL GRID
# ============================================================

# Expected:
#
# [B, Nx, Ny, T]

if sample_x.ndim != 4:

    raise RuntimeError(
        f"Expected input with 4 dimensions "
        f"[B, Nx, Ny, T], got {sample_x.shape}"
    )

Nx = sample_x.shape[1]
Ny = sample_x.shape[2]

T_in = sample_x.shape[-1]
T_out = sample_y.shape[-1]

print("\nGrid:")
print("Nx =", Nx)
print("Ny =", Ny)
print("T_in =", T_in)
print("T_out =", T_out)


# ============================================================
# 5. PHYSICAL PARAMETERS
# ============================================================

NU = 1e-5

DX = 1.0 / Nx
DY = 1.0 / Ny

# IMPORTANT:
# Change this if your dataset uses a different temporal spacing.
DT = 1.0


# ============================================================
# 6. EXPERIMENT PARAMETERS
# ============================================================

EPOCHS = 25

# Use the complete rollout contained in the dataset.
ROLLOUT_STEPS = T_out

LEARNING_RATE = 1e-3

STEP_SIZE = 10
GAMMA = 0.5


# ============================================================
# 7. VARIATIONAL LOSS WEIGHTS
# ============================================================

LAMBDA_PRED = 5.0
LAMBDA_RECON = 0.5

LAMBDA_VAR = 0.1

LAMBDA_WEAK = 1.0
LAMBDA_ENERGY = 0.1


# ============================================================
# 8. BUILD MODEL
# ============================================================

print("\n" + "-" * 70)
print("BUILDING VARIATIONAL KNO")
print("-" * 70)

model = koopman(
    backbone="KNO2d",
    autoencoder="Conv2d",

    # Same KNO architecture as baseline
    o=16,
    m=16,
    r=8,

    t_in=T_in,

    device=device,

    # Supervised prediction loss
    lambda_pred=LAMBDA_PRED,

    # Autoencoder reconstruction loss
    lambda_recon=LAMBDA_RECON,

    # Variational physics
    lambda_var=LAMBDA_VAR,

    # Navier-Stokes viscosity
    nu=NU,

    # Grid
    dx=DX,
    dy=DY,

    # Temporal spacing
    dt=DT,

    # Weak formulation
    n_test=8,
    lambda_weak=LAMBDA_WEAK,

    # Energy balance
    lambda_energy=LAMBDA_ENERGY,
)


# ============================================================
# 9. COMPILE
# ============================================================

model.compile()


# ============================================================
# 10. OPTIMIZER
# ============================================================

model.opt_init(
    opt="Adam",
    lr=LEARNING_RATE,
    step_size=STEP_SIZE,
    gamma=GAMMA,
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

print(f"Epochs          : {EPOCHS}")
print(f"Rollout steps   : {ROLLOUT_STEPS}")
print(f"Viscosity       : {NU}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Lambda var      : {LAMBDA_VAR}")
print(f"Lambda weak     : {LAMBDA_WEAK}")
print(f"Lambda energy   : {LAMBDA_ENERGY}")
print("=" * 70)


history = model.train(
    epochs=EPOCHS,
    trainloader=trainloader,
    step=1,
    T_out=ROLLOUT_STEPS,
    evalloader=testloader,
)


# ============================================================
# 12. TRAINING CURVES
# ============================================================

print("\nDisplaying training curves...")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9)
)


# ------------------------------------------------------------
# Prediction loss
# ------------------------------------------------------------

axes[0, 0].plot(
    history["train_pred"],
    label="Train"
)

axes[0, 0].plot(
    history["eval_pred"],
    label="Test"
)

axes[0, 0].set_title(
    "Prediction MSE"
)

axes[0, 0].set_xlabel(
    "Epoch"
)

axes[0, 0].set_ylabel(
    "MSE"
)

axes[0, 0].grid(True)
axes[0, 0].legend()


# ------------------------------------------------------------
# Variational loss
# ------------------------------------------------------------

axes[0, 1].plot(
    history["train_var"],
    label="Train Variational"
)

axes[0, 1].plot(
    history["eval_var"],
    label="Test Variational"
)

axes[0, 1].set_title(
    "Variational Loss"
)

axes[0, 1].set_xlabel(
    "Epoch"
)

axes[0, 1].set_ylabel(
    "Loss"
)

axes[0, 1].grid(True)
axes[0, 1].legend()


# ------------------------------------------------------------
# Weak loss
# ------------------------------------------------------------

axes[1, 0].plot(
    history["train_weak"],
    label="Train Weak"
)

axes[1, 0].plot(
    history["eval_weak"],
    label="Test Weak"
)

axes[1, 0].set_title(
    "Weak-Form Loss"
)

axes[1, 0].set_xlabel(
    "Epoch"
)

axes[1, 0].set_ylabel(
    "Loss"
)

axes[1, 0].grid(True)
axes[1, 0].legend()


# ------------------------------------------------------------
# Energy balance
# ------------------------------------------------------------

axes[1, 1].plot(
    history["train_energy"],
    label="Train Energy"
)

axes[1, 1].plot(
    history["eval_energy"],
    label="Test Energy"
)

axes[1, 1].set_title(
    "Energy-Balance Loss"
)

axes[1, 1].set_xlabel(
    "Epoch"
)

axes[1, 1].set_ylabel(
    "Loss"
)

axes[1, 1].grid(True)
axes[1, 1].legend()


plt.suptitle(
    "Variational KNO Training"
)

plt.tight_layout()
plt.show()


# ============================================================
# 13. AUTOREGRESSIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("AUTOREGRESSIVE TEST")
print("=" * 70)

model.kernel.eval()


with torch.no_grad():

    # Take one test example
    xx, yy = next(iter(testloader))

    xx = xx.to(device)
    yy = yy.to(device)

    # --------------------------------------------------------
    # Keep only one example for visualization
    # --------------------------------------------------------

    xx_single = xx[0:1].clone()
    yy_single = yy[0:1].clone()

    predictions = []

    # --------------------------------------------------------
    # Roll forward
    # --------------------------------------------------------

    for t in range(ROLLOUT_STEPS):

        pred, _ = model.kernel(
            xx_single
        )

        predicted = pred[..., -1:]

        predictions.append(
            predicted.clone()
        )

        # Shift temporal window
        xx_single = torch.cat(
            (
                xx_single[..., 1:],
                predicted
            ),
            dim=-1
        )

    prediction = torch.cat(
        predictions,
        dim=-1
    )


# ============================================================
# 14. CONVERT TO CPU
# ============================================================

prediction_cpu = (
    prediction[0]
    .detach()
    .cpu()
)

truth_cpu = (
    yy_single[0]
    .detach()
    .cpu()
)


# ============================================================
# 15. ERROR AT EVERY TIME STEP
# ============================================================

mse_t = []
relative_l2_t = []

for t in range(ROLLOUT_STEPS):

    pred_t = prediction_cpu[..., t]

    true_t = truth_cpu[..., t]

    mse = torch.mean(
        (pred_t - true_t) ** 2
    )

    rel_l2 = (
        torch.norm(pred_t - true_t)
        /
        (
            torch.norm(true_t)
            + 1e-12
        )
    )

    mse_t.append(
        mse.item()
    )

    relative_l2_t.append(
        rel_l2.item()
    )


# ============================================================
# 16. ERROR VS TIME
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    range(1, ROLLOUT_STEPS + 1),
    mse_t,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Prediction time step"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "Autoregressive Prediction Error"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. RELATIVE L2 VS TIME
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    range(1, ROLLOUT_STEPS + 1),
    relative_l2_t,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Prediction time step"
)

plt.ylabel(
    "Relative L2 Error"
)

plt.title(
    "Relative L2 Error During Rollout"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 18. FIELD VISUALIZATION
# ============================================================

# Select representative time steps.
#
# Beginning, early rollout, middle, late rollout, final.

plot_times = sorted(
    set(
        [
            0,
            min(9, ROLLOUT_STEPS - 1),
            min(19, ROLLOUT_STEPS - 1),
            min(29, ROLLOUT_STEPS - 1),
            ROLLOUT_STEPS - 1,
        ]
    )
)


fig, axes = plt.subplots(
    len(plot_times),
    3,
    figsize=(13, 3.5 * len(plot_times))
)


# Handle case where only one row exists
if len(plot_times) == 1:

    axes = np.expand_dims(
        axes,
        axis=0
    )


for row, t in enumerate(plot_times):

    true_field = (
        truth_cpu[..., t]
        .numpy()
    )

    pred_field = (
        prediction_cpu[..., t]
        .numpy()
    )

    error_field = np.abs(
        true_field - pred_field
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    im0 = axes[row, 0].imshow(
        true_field,
        aspect="auto"
    )

    axes[row, 0].set_title(
        f"Ground Truth — t={t + 1}"
    )

    axes[row, 0].set_xlabel("y")
    axes[row, 0].set_ylabel("x")

    plt.colorbar(
        im0,
        ax=axes[row, 0],
        fraction=0.046,
        pad=0.04
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    im1 = axes[row, 1].imshow(
        pred_field,
        aspect="auto"
    )

    axes[row, 1].set_title(
        f"Prediction — t={t + 1}"
    )

    axes[row, 1].set_xlabel("y")
    axes[row, 1].set_ylabel("x")

    plt.colorbar(
        im1,
        ax=axes[row, 1],
        fraction=0.046,
        pad=0.04
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    im2 = axes[row, 2].imshow(
        error_field,
        aspect="auto"
    )

    axes[row, 2].set_title(
        f"Absolute Error — t={t + 1}"
    )

    axes[row, 2].set_xlabel("y")
    axes[row, 2].set_ylabel("x")

    plt.colorbar(
        im2,
        ax=axes[row, 2],
        fraction=0.046,
        pad=0.04
    )


plt.suptitle(
    "Variational KNO — Navier-Stokes Rollout",
    fontsize=15
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. FINAL METRICS
# ============================================================

overall_mse = torch.mean(
    (
        prediction_cpu
        -
        truth_cpu
    ) ** 2
).item()


overall_relative_l2 = (
    torch.norm(
        prediction_cpu
        -
        truth_cpu
    )
    /
    (
        torch.norm(truth_cpu)
        +
        1e-12
    )
).item()


final_mse = mse_t[-1]
final_relative_l2 = relative_l2_t[-1]


print("\n")
print("=" * 70)
print("FINAL VARIATIONAL KNO RESULTS")
print("=" * 70)

print(
    f"Overall rollout MSE       : "
    f"{overall_mse:.6e}"
)

print(
    f"Overall relative L2       : "
    f"{overall_relative_l2:.6e}"
)

print(
    f"First-step MSE             : "
    f"{mse_t[0]:.6e}"
)

print(
    f"Final-step MSE             : "
    f"{final_mse:.6e}"
)

print(
    f"First-step relative L2     : "
    f"{relative_l2_t[0]:.6e}"
)

print(
    f"Final-step relative L2     : "
    f"{final_relative_l2:.6e}"
)

print("=" * 70)


# ============================================================
# 20. LOSS SUMMARY
# ============================================================

print("\nLOSS SUMMARY")
print("-" * 70)

print(
    f"Initial train prediction : "
    f"{history['train_pred'][0]:.6e}"
)

print(
    f"Final train prediction   : "
    f"{history['train_pred'][-1]:.6e}"
)

print(
    f"Initial train variational: "
    f"{history['train_var'][0]:.6e}"
)

print(
    f"Final train variational  : "
    f"{history['train_var'][-1]:.6e}"
)

print(
    f"Initial train weak       : "
    f"{history['train_weak'][0]:.6e}"
)

print(
    f"Final train weak         : "
    f"{history['train_weak'][-1]:.6e}"
)

print(
    f"Initial train energy     : "
    f"{history['train_energy'][0]:.6e}"
)

print(
    f"Final train energy       : "
    f"{history['train_energy'][-1]:.6e}"
)

print("=" * 70)

print("\nExperiment finished.")
print("No model, prediction, or figure was saved to disk.")
