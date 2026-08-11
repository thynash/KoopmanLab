# ============================================================
# demo_variational.py
#
# Variational KNO for 2D Navier-Stokes
#
# Dataset:
#   NavierStokes_V1e-5_N1200_T20.mat
#
# Viscosity:
#   nu = 1e-5
#
# This demo:
#   1. Loads the .mat dataset directly
#   2. Creates train/test loaders
#   3. Trains Variational KNO for 25 epochs
#   4. Displays training curves
#   5. Displays autoregressive prediction errors
#   6. Displays GT / Prediction / Error fields
#   7. Prints final metrics
#
# NOTHING IS SAVED TO DISK.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import torch
import koopmanlab as kp

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")

import matplotlib.pyplot as plt

import numpy as np

from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader

from model_variational import koopman


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "/content/NavierStokes_V1e-5_N1200_T20.mat"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------

NU = 1e-5

# Dataset is normally on [0, 1] x [0, 1]
# These will be corrected automatically from the grid size.

# IMPORTANT:
# Set this to the actual temporal spacing used by your
# Navier-Stokes dataset if your original demo specifies it.
DT = 1.0

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

TRAIN_RATIO = 0.8

BATCH_SIZE = 20

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 25

LEARNING_RATE = 1e-3

STEP_SIZE = 10

GAMMA = 0.5

# ------------------------------------------------------------
# KNO architecture
# ------------------------------------------------------------

OPERATOR_SIZE = 16

MODES = 16

DECOMPOSE = 8

T_IN = 1

# ------------------------------------------------------------
# Variational loss
# ------------------------------------------------------------

LAMBDA_PRED = 5.0

LAMBDA_RECON = 0.5

LAMBDA_VAR = 0.1

LAMBDA_WEAK = 1.0

LAMBDA_ENERGY = 0.1

N_TEST_FUNCTIONS = 8


# ============================================================
# 2. DEVICE INFORMATION
# ============================================================

print("=" * 75)
print("VARIATIONAL KNO — NAVIER-STOKES")
print("=" * 75)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. LOAD MATLAB DATASET
# ============================================================

print("\n" + "-" * 75)
print("LOADING DATASET")
print("-" * 75)

print("Path:")
print(DATA_PATH)


mat_data = loadmat(
    DATA_PATH
)


# ------------------------------------------------------------
# Display available MATLAB variables
# ------------------------------------------------------------

print("\nMATLAB variables:")

for key, value in mat_data.items():

    if not key.startswith("__"):

        if isinstance(value, np.ndarray):

            print(
                f"  {key:20s} "
                f"shape={value.shape} "
                f"dtype={value.dtype}"
            )


# ============================================================
# 4. FIND THE SOLUTION FIELD
# ============================================================
#
# The standard Navier-Stokes MATLAB dataset stores the
# solution under the key "u".
#
# We first try "u".
#
# If it is not present, automatically search for a suitable
# numerical array.
# ============================================================

if "u" in mat_data:

    data = mat_data["u"]

else:

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if not isinstance(value, np.ndarray):
            continue

        if value.ndim >= 3:

            candidates.append(
                (key, value)
            )

    if len(candidates) == 0:

        raise RuntimeError(
            "Could not find a solution field in the .mat file."
        )

    print(
        "\nWARNING: 'u' was not found."
    )

    print(
        "Using:",
        candidates[0][0]
    )

    data = candidates[0][1]


# ============================================================
# 5. CONVERT DATA TO FLOAT32
# ============================================================

data = np.asarray(
    data,
    dtype=np.float32
)

print("\nSolution field shape:")
print(data.shape)

print(
    "Minimum:",
    data.min()
)

print(
    "Maximum:",
    data.max()
)

print(
    "Mean:",
    data.mean()
)

print(
    "Std:",
    data.std()
)


# ============================================================
# 6. CHECK DATA DIMENSIONS
# ============================================================

if data.ndim != 4:

    raise RuntimeError(
        "\nExpected Navier-Stokes data to have four dimensions "
        "[N, Nx, Ny, T].\n"
        f"Received shape: {data.shape}\n"
        "If the dataset uses a different layout, inspect the "
        "printed MATLAB variables above."
    )


# ------------------------------------------------------------
# Expected:
#
# [N, Nx, Ny, T]
# ------------------------------------------------------------

N_SAMPLES = data.shape[0]

NX = data.shape[1]

NY = data.shape[2]

TOTAL_TIME = data.shape[3]


print("\n" + "-" * 75)
print("DATASET INFORMATION")
print("-" * 75)

print(
    "Number of samples:",
    N_SAMPLES
)

print(
    "Spatial grid:",
    NX,
    "x",
    NY
)

print(
    "Number of time steps:",
    TOTAL_TIME
)


# ============================================================
# 7. INPUT / OUTPUT CONSTRUCTION
# ============================================================
#
# KNO receives the previous state as input:
#
#     [B, Nx, Ny, 1]
#
# and predicts future states:
#
#     [B, Nx, Ny, T_out]
#
# We use the first time slice as the initial condition and
# the remaining time slices as the prediction target.
# ============================================================

if TOTAL_TIME < 2:

    raise RuntimeError(
        "Dataset must contain at least two time steps."
    )


x = data[..., :1]

y = data[..., 1:]


print("\nInput shape:")
print(x.shape)

print("Target shape:")
print(y.shape)


T_IN = x.shape[-1]

T_OUT = y.shape[-1]


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

N_TRAIN = int(
    TRAIN_RATIO * N_SAMPLES
)

N_TEST = N_SAMPLES - N_TRAIN


x_train = x[:N_TRAIN]

y_train = y[:N_TRAIN]

x_test = x[N_TRAIN:]

y_test = y[N_TRAIN:]


print("\n" + "-" * 75)
print("TRAIN / TEST SPLIT")
print("-" * 75)

print(
    "Training samples:",
    N_TRAIN
)

print(
    "Testing samples:",
    N_TEST
)

print(
    "Training input:",
    x_train.shape
)

print(
    "Training target:",
    y_train.shape
)

print(
    "Testing input:",
    x_test.shape
)

print(
    "Testing target:",
    y_test.shape
)


# ============================================================
# 9. CONVERT TO PYTORCH
# ============================================================

x_train = torch.from_numpy(
    x_train
)

y_train = torch.from_numpy(
    y_train
)

x_test = torch.from_numpy(
    x_test
)

y_test = torch.from_numpy(
    y_test
)


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    x_train,
    y_train
)

test_dataset = TensorDataset(
    x_test,
    y_test
)


trainloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)


testloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 11. SPATIAL / TEMPORAL PARAMETERS
# ============================================================

DX = 1.0 / NX

DY = 1.0 / NY


print("\n" + "-" * 75)
print("PHYSICAL PARAMETERS")
print("-" * 75)

print(
    "Viscosity nu:",
    NU
)

print(
    "dx:",
    DX
)

print(
    "dy:",
    DY
)

print(
    "dt:",
    DT
)

print(
    "Input time steps:",
    T_IN
)

print(
    "Output time steps:",
    T_OUT
)


# ============================================================
# 12. CREATE VARIATIONAL KNO
# ============================================================

print("\n" + "-" * 75)
print("BUILDING VARIATIONAL KNO")
print("-" * 75)


model = koopman(

    # --------------------------------------------------------
    # KNO backbone
    # --------------------------------------------------------

    backbone="KNO2d",

    autoencoder="Conv2d",

    o=OPERATOR_SIZE,

    m=MODES,

    r=DECOMPOSE,

    t_in=T_IN,

    device=DEVICE,

    # --------------------------------------------------------
    # Existing prediction / reconstruction losses
    # --------------------------------------------------------

    lambda_pred=LAMBDA_PRED,

    lambda_recon=LAMBDA_RECON,

    # --------------------------------------------------------
    # Variational loss
    # --------------------------------------------------------

    lambda_var=LAMBDA_VAR,

    # --------------------------------------------------------
    # Navier-Stokes
    # --------------------------------------------------------

    nu=NU,

    dx=DX,

    dy=DY,

    dt=DT,

    # --------------------------------------------------------
    # Weak formulation
    # --------------------------------------------------------

    n_test=N_TEST_FUNCTIONS,

    lambda_weak=LAMBDA_WEAK,

    # --------------------------------------------------------
    # Energy balance
    # --------------------------------------------------------

    lambda_energy=LAMBDA_ENERGY,
)


# ============================================================
# 13. COMPILE
# ============================================================

model.compile()


# ============================================================
# 14. OPTIMIZER
# ============================================================

model.opt_init(

    opt="Adam",

    lr=LEARNING_RATE,

    step_size=STEP_SIZE,

    gamma=GAMMA,
)


# ============================================================
# 15. TRAINING INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("TRAINING CONFIGURATION")
print("=" * 75)

print(
    f"Epochs              : {EPOCHS}"
)

print(
    f"Batch size          : {BATCH_SIZE}"
)

print(
    f"Rollout steps       : {T_OUT}"
)

print(
    f"Learning rate       : {LEARNING_RATE}"
)

print(
    f"Viscosity           : {NU}"
)

print(
    f"Lambda prediction   : {LAMBDA_PRED}"
)

print(
    f"Lambda reconstruction: {LAMBDA_RECON}"
)

print(
    f"Lambda variational  : {LAMBDA_VAR}"
)

print(
    f"Lambda weak         : {LAMBDA_WEAK}"
)

print(
    f"Lambda energy       : {LAMBDA_ENERGY}"
)

print("=" * 75)


# ============================================================
# 16. TRAIN
# ============================================================

history = model.train(

    epochs=EPOCHS,

    trainloader=trainloader,

    step=1,

    T_out=T_OUT,

    evalloader=testloader,
)


# ============================================================
# 17. TRAINING CURVES
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
    label="Train"
)

axes[0, 1].plot(
    history["eval_var"],
    label="Test"
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
# Weak-form loss
# ------------------------------------------------------------

axes[1, 0].plot(
    history["train_weak"],
    label="Train"
)

axes[1, 0].plot(
    history["eval_weak"],
    label="Test"
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
# Energy loss
# ------------------------------------------------------------

axes[1, 1].plot(
    history["train_energy"],
    label="Train"
)

axes[1, 1].plot(
    history["eval_energy"],
    label="Test"
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
    "Variational KNO — Navier-Stokes"
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. AUTOREGRESSIVE TEST
# ============================================================

print("\n" + "=" * 75)
print("AUTOREGRESSIVE TEST")
print("=" * 75)


model.kernel.eval()


# ------------------------------------------------------------
# Take ONE complete test example
# ------------------------------------------------------------

xx, yy = next(
    iter(testloader)
)


xx = xx.to(
    DEVICE
)

yy = yy.to(
    DEVICE
)


# ------------------------------------------------------------
# Use first test sample
# ------------------------------------------------------------

xx_single = xx[0:1].clone()

yy_single = yy[0:1].clone()


predictions = []


# ============================================================
# 19. ROLLOUT
# ============================================================

with torch.no_grad():

    for t in range(T_OUT):

        pred, _ = model.kernel(
            xx_single
        )

        predicted = pred[..., -1:]

        predictions.append(
            predicted.clone()
        )

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
# 20. CPU ARRAYS
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
# 21. ERROR OVER TIME
# ============================================================

mse_t = []

relative_l2_t = []


for t in range(T_OUT):

    pred_t = prediction_cpu[..., t]

    true_t = truth_cpu[..., t]


    mse = torch.mean(
        (
            pred_t
            -
            true_t
        ) ** 2
    )


    rel_l2 = (
        torch.norm(
            pred_t
            -
            true_t
        )
        /
        (
            torch.norm(true_t)
            +
            1e-12
        )
    )


    mse_t.append(
        mse.item()
    )

    relative_l2_t.append(
        rel_l2.item()
    )


# ============================================================
# 22. MSE VS TIME
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    range(1, T_OUT + 1),
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
    "Autoregressive MSE vs Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 23. RELATIVE L2 VS TIME
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    range(1, T_OUT + 1),
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
    "Autoregressive Relative L2 Error"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 24. SELECT VISUALIZATION TIMES
# ============================================================

plot_times = sorted(
    set(
        [
            0,
            min(4, T_OUT - 1),
            min(9, T_OUT - 1),
            min(14, T_OUT - 1),
            T_OUT - 1,
        ]
    )
)


# ============================================================
# 25. GT / PREDICTION / ERROR
# ============================================================

fig, axes = plt.subplots(
    len(plot_times),
    3,
    figsize=(
        13,
        3.5 * len(plot_times)
    )
)


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
        true_field
        -
        pred_field
    )


    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    im0 = axes[row, 0].imshow(
        true_field,
        aspect="auto"
    )

    axes[row, 0].set_title(
        f"Ground Truth — t={t + 1}"
    )

    axes[row, 0].set_xlabel(
        "y"
    )

    axes[row, 0].set_ylabel(
        "x"
    )

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

    axes[row, 1].set_xlabel(
        "y"
    )

    axes[row, 1].set_ylabel(
        "x"
    )

    plt.colorbar(
        im1,
        ax=axes[row, 1],
        fraction=0.046,
        pad=0.04
    )


    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    im2 = axes[row, 2].imshow(
        error_field,
        aspect="auto"
    )

    axes[row, 2].set_title(
        f"Absolute Error — t={t + 1}"
    )

    axes[row, 2].set_xlabel(
        "y"
    )

    axes[row, 2].set_ylabel(
        "x"
    )

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
# 26. FINAL METRICS
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


print("\n" + "=" * 75)
print("FINAL VARIATIONAL KNO RESULTS")
print("=" * 75)


print(
    f"Overall rollout MSE       : "
    f"{overall_mse:.6e}"
)


print(
    f"Overall relative L2       : "
    f"{overall_relative_l2:.6e}"
)


print(
    f"First-step MSE            : "
    f"{mse_t[0]:.6e}"
)


print(
    f"Final-step MSE            : "
    f"{mse_t[-1]:.6e}"
)


print(
    f"First-step relative L2    : "
    f"{relative_l2_t[0]:.6e}"
)


print(
    f"Final-step relative L2    : "
    f"{relative_l2_t[-1]:.6e}"
)


print("=" * 75)


# ============================================================
# 27. LOSS SUMMARY
# ============================================================

print("\nLOSS SUMMARY")
print("-" * 75)


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


print("=" * 75)

print(
    "\nExperiment completed."
)

print(
    "No models, predictions, or figures were saved."
)
