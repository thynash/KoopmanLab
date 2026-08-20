import os
import torch


# ============================================================
# EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "allen_cahn"

SEED = 42


# ============================================================
# DIRECTORIES
# ============================================================

EXPERIMENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    EXPERIMENT_DIR,
    "data",
)

RESULTS_DIR = os.path.join(
    EXPERIMENT_DIR,
    "results",
)

BASELINE_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "baseline",
)

PEDVINO_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "pedvino",
)

COMPARISON_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "comparison",
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "allen_cahn_dataset.pt",
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)

os.makedirs(
    BASELINE_RESULTS_DIR,
    exist_ok=True,
)

os.makedirs(
    PEDVINO_RESULTS_DIR,
    exist_ok=True,
)

os.makedirs(
    COMPARISON_RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SPATIAL DOMAIN
# ============================================================

# Periodic domain:
#
#     x in [0, 1)
#
# The endpoint x = 1 is not explicitly duplicated because
# x = 1 and x = 0 represent the same periodic location.

DOMAIN_X = (
    0.0,
    1.0,
)

DOMAIN_LENGTH = (
    DOMAIN_X[1]
    - DOMAIN_X[0]
)


# ============================================================
# SPATIAL GRID
# ============================================================

GRID_SIZE = 128

GRID_SIZE_X = GRID_SIZE

DX = (
    DOMAIN_LENGTH
    / GRID_SIZE_X
)


# ============================================================
# ALLEN-CAHN PDE
# ============================================================

# PDE:
#
#     u_t = epsilon^2 u_xx + u - u^3
#
# equivalently:
#
#     u_t - epsilon^2 u_xx - u + u^3 = 0

EPSILON = 0.01


# ============================================================
# TIME DOMAIN
# ============================================================

INITIAL_TIME = 0.0

FINAL_TIME = 0.10

TIME_HORIZON = (
    FINAL_TIME
    - INITIAL_TIME
)


# Numerical solver time step used only while generating
# the ground-truth dataset.

SOLVER_DT = 1e-4

SOLVER_NUM_STEPS = int(
    round(
        TIME_HORIZON
        / SOLVER_DT
    )
)


# Actual PDE time difference used by PEDVINO when computing:
#
#     u_t ≈ (u_next - u_previous) / DT
#
# Since the operator learns:
#
#     u(x, 0) -> u(x, FINAL_TIME)
#
# previous_state is the input state and prediction is the
# state at FINAL_TIME.

DT = TIME_HORIZON


# ============================================================
# DATASET
# ============================================================

NUM_SAMPLES = 2000

TRAIN_FRACTION = 0.70

VAL_FRACTION = 0.15

TEST_FRACTION = 0.15


TRAIN_SIZE = int(
    NUM_SAMPLES
    * TRAIN_FRACTION
)

VAL_SIZE = int(
    NUM_SAMPLES
    * VAL_FRACTION
)

TEST_SIZE = (
    NUM_SAMPLES
    - TRAIN_SIZE
    - VAL_SIZE
)


BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

PERSISTENT_WORKERS = (
    NUM_WORKERS > 0
)


# ============================================================
# INITIAL CONDITION GENERATION
# ============================================================

# Each initial condition is generated as a smooth random
# Fourier series:
#
#     u_0(x)
#       =
#       c_0
#       +
#       sum_k [
#           a_k sin(2 pi k x / L)
#           +
#           b_k cos(2 pi k x / L)
#       ]
#
# with spectral decay to ensure smooth initial states.

NUM_INITIAL_MODES = 8

INITIAL_AMPLITUDE = 0.35

INITIAL_MEAN_AMPLITUDE = 0.15

SPECTRAL_DECAY_POWER = 1.5


# ============================================================
# NUMERICAL DATA GENERATION
# ============================================================

# The Allen-Cahn dataset is generated using a
# semi-implicit Fourier spectral solver:
#
#     u_t = epsilon^2 u_xx + u - u^3
#
# Diffusion is treated implicitly and the reaction term
# explicitly.

SOLVER_METHOD = "semi_implicit_spectral"

# Generate the dataset in batches so generation does not
# unnecessarily consume memory.

DATA_GENERATION_BATCH_SIZE = 100


# ============================================================
# INPUT / OUTPUT DIMENSIONS
# ============================================================

# Allen-Cahn operator:
#
#     u(x, 0) -> u(x, T)

T_LEN = 1

INPUT_CHANNELS = 1

OUTPUT_CHANNELS = 1


# ============================================================
# KNO / PEDVINO LATENT DIMENSION
# ============================================================

# Keep the same architecture as the Burgers experiment
# for a consistent experimental setup.

OPERATOR_SIZE = 24


# ============================================================
# KOOPMAN SPECTRAL MODES
# ============================================================

MODES_X = 16


# ============================================================
# KOOPMAN DECOMPOSITION
# ============================================================

DECOMPOSE = 4

LINEAR_TYPE = True

NORMALIZATION = False


# ============================================================
# PHYSICS ENCODER / DECODER
# ============================================================

PHYSICS_HIDDEN_SIZE = 32


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 150

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5


# ============================================================
# OPTIMIZATION
# ============================================================

OPTIMIZER = "adamw"


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

SCHEDULER = "cosine"

MIN_LEARNING_RATE = 1e-5


# ============================================================
# GRADIENT STABILITY
# ============================================================

GRADIENT_CLIP = 1.0


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 30

EARLY_STOPPING_MIN_DELTA = 1e-4


# ============================================================
# BASELINE LOSS
# ============================================================

LAMBDA_PRED = 1.0

LAMBDA_RECON = 0.02


# ============================================================
# PEDVINO LOSS
# ============================================================

LAMBDA_ENERGY = 0.05

LAMBDA_GRAD = 0.02


# Allen-Cahn dataset uses periodic boundary conditions.
#
# The periodic boundary handling will be implemented in the
# PEDVINO experiment rather than imposing Dirichlet values.

LAMBDA_BC = 0.05


# ============================================================
# LOSS STABILITY
# ============================================================

LOSS_EPS = 1e-8

USE_RELATIVE_PREDICTION_LOSS = True


# ============================================================
# PHYSICS WARM-UP
# ============================================================

ENERGY_WARMUP_EPOCHS = 15

GRADIENT_WARMUP_EPOCHS = 15

BOUNDARY_WARMUP_EPOCHS = 5


# ============================================================
# PHYSICS RAMP
# ============================================================

PHYSICS_RAMP_EPOCHS = 20


# ============================================================
# EVALUATION
# ============================================================

EVALUATE_EVERY = 1

CHECKPOINT_METRIC = "relative_l2"

PRINT_EVERY = 1


# ============================================================
# SAVING
# ============================================================

SAVE_HISTORY = True

SAVE_METRICS = True

SAVE_CHECKPOINT = True


# ============================================================
# FAIR BASELINE COMPARISON
# ============================================================

FAIR_COMPARISON = True


# ============================================================
# CHECKPOINT PATHS
# ============================================================

BASELINE_CHECKPOINT_PATH = os.path.join(
    BASELINE_RESULTS_DIR,
    "best_model.pt",
)

PEDVINO_CHECKPOINT_PATH = os.path.join(
    PEDVINO_RESULTS_DIR,
    "best_model.pt",
)


# ============================================================
# HISTORY PATHS
# ============================================================

BASELINE_HISTORY_PATH = os.path.join(
    BASELINE_RESULTS_DIR,
    "history.json",
)

PEDVINO_HISTORY_PATH = os.path.join(
    PEDVINO_RESULTS_DIR,
    "history.json",
)


# ============================================================
# METRICS PATHS
# ============================================================

BASELINE_METRICS_PATH = os.path.join(
    BASELINE_RESULTS_DIR,
    "metrics.json",
)

PEDVINO_METRICS_PATH = os.path.join(
    PEDVINO_RESULTS_DIR,
    "metrics.json",
)


# ============================================================
# CONFIG VALIDATION
# ============================================================

assert GRID_SIZE_X >= 8

assert NUM_SAMPLES > 0

assert BATCH_SIZE > 0

assert OPERATOR_SIZE > 0

assert MODES_X > 0

assert MODES_X <= GRID_SIZE_X // 2

assert NUM_INITIAL_MODES > 0

assert NUM_INITIAL_MODES < GRID_SIZE_X // 2

assert EPSILON > 0.0

assert TIME_HORIZON > 0.0

assert SOLVER_DT > 0.0

assert SOLVER_NUM_STEPS > 0

assert abs(
    TRAIN_FRACTION
    + VAL_FRACTION
    + TEST_FRACTION
    - 1.0
) < 1e-6

assert TRAIN_SIZE > 0

assert VAL_SIZE > 0

assert TEST_SIZE > 0


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ALLEN-CAHN EXPERIMENT CONFIGURATION")
    print("=" * 60)

    print(f"Dataset path         : {DATASET_PATH}")
    print(f"Device               : {DEVICE}")
    print(f"Grid size            : {GRID_SIZE_X}")
    print(f"Domain               : {DOMAIN_X}")
    print(f"DX                   : {DX}")
    print(f"Epsilon              : {EPSILON}")
    print(f"Final time           : {FINAL_TIME}")
    print(f"Solver dt            : {SOLVER_DT}")
    print(f"Solver steps         : {SOLVER_NUM_STEPS}")
    print(f"Number of samples    : {NUM_SAMPLES}")
    print(
        f"Train / Val / Test   : "
        f"{TRAIN_SIZE} / {VAL_SIZE} / {TEST_SIZE}"
    )
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Operator size        : {OPERATOR_SIZE}")
    print(f"KNO modes            : {MODES_X}")
    print(f"Epochs               : {EPOCHS}")

    print("-" * 60)
    print("PEDVINO LOSS WEIGHTS")
    print("-" * 60)

    print(f"Prediction           : {LAMBDA_PRED}")
    print(f"Reconstruction       : {LAMBDA_RECON}")
    print(f"Energy               : {LAMBDA_ENERGY}")
    print(f"Gradient             : {LAMBDA_GRAD}")
    print(f"Boundary             : {LAMBDA_BC}")

    print("=" * 60)
