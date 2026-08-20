import os
import torch


# ============================================================
# EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "darcy"

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
    "darcy_dataset.pt",
)

VINO_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "vino",
)
# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)

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

os.makedirs(
    VINO_RESULTS_DIR,
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

DOMAIN_X = (0.0, 1.0)

DOMAIN_Y = (0.0, 1.0)


# ============================================================
# GRID
# ============================================================

GRID_SIZE = 64

GRID_SIZE_X = GRID_SIZE

GRID_SIZE_Y = GRID_SIZE


DX = (
    DOMAIN_X[1] - DOMAIN_X[0]
) / (GRID_SIZE_X - 1)

DY = (
    DOMAIN_Y[1] - DOMAIN_Y[0]
) / (GRID_SIZE_Y - 1)


# ============================================================
# DATASET
# ============================================================

NUM_SAMPLES = 2000

TRAIN_FRACTION = 0.70

VAL_FRACTION = 0.15

TEST_FRACTION = 0.15


TRAIN_SIZE = int(
    NUM_SAMPLES * TRAIN_FRACTION
)

VAL_SIZE = int(
    NUM_SAMPLES * VAL_FRACTION
)

TEST_SIZE = (
    NUM_SAMPLES
    - TRAIN_SIZE
    - VAL_SIZE
)


BATCH_SIZE = 20

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

PERSISTENT_WORKERS = (
    NUM_WORKERS > 0
)


# ============================================================
# DARCY PDE
# ============================================================

# PDE:
#
#     - div(a(x,y) grad(u(x,y))) = f(x,y)
#
# where:
#
#     a(x,y) = permeability / coefficient
#     u(x,y) = pressure / solution
#     f(x,y) = forcing



FORCING_TYPE='constant'


FORCING_VALUE = 1.0


# ============================================================
# COEFFICIENT FIELD
# ============================================================

# Keep one canonical naming convention.

COEFFICIENT_MIN = 0.1

COEFFICIENT_MAX = 1.0


# Compatibility aliases used by existing code.

MIN_COEFFICIENT = COEFFICIENT_MIN

MAX_COEFFICIENT = COEFFICIENT_MAX


# ============================================================
# COEFFICIENT FIELD GENERATION
# ============================================================

# Number of Fourier / random modes used by dataset.py
# when generating Darcy coefficient fields.

NUM_MODES_X = 8

NUM_MODES_Y = 8


# Optional amplitude controls.

COEFFICIENT_AMPLITUDE = 0.5

COEFFICIENT_MEAN = 1.0


# ============================================================
# NUMERICAL DARCY SOLVER
# ============================================================

# dataset.py explicitly expects these variables.

SOLVER_MAX_ITERATIONS = 5000

SOLVER_TOLERANCE = 1e-8


# Compatibility aliases.

MAX_ITERATIONS = SOLVER_MAX_ITERATIONS

TOLERANCE = SOLVER_TOLERANCE


# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

# Current dataset / experiment assumes homogeneous
# Dirichlet boundary conditions.

BOUNDARY_VALUE = 0.0

DIRICHLET_BOUNDARY_VALUE = BOUNDARY_VALUE


# ============================================================
# INPUT / OUTPUT DIMENSIONS
# ============================================================

# Darcy operator:
#
#     a(x,y) -> u(x,y)
#
# coefficient/permeability : 1 channel
# solution/pressure        : 1 channel

T_LEN = 1

INPUT_CHANNELS = 1

OUTPUT_CHANNELS = 1


# ============================================================
# KNO / PEDVINO LATENT DIMENSION
# ============================================================

OPERATOR_SIZE = 24


# ============================================================
# KOOPMAN SPECTRAL MODES
# ============================================================

# These are KNO spectral modes.
#
# They are NOT the same as NUM_MODES_X/Y, which are used
# for generating coefficient fields.

MODES_X = 12

MODES_Y = 12


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

# Original KNO objective:
#
# L =
#     lambda_pred  * L_prediction
#   + lambda_recon * L_reconstruction

LAMBDA_PRED = 1.0

LAMBDA_RECON = 0.02


# ============================================================
# PEDVINO LOSS
# ============================================================

# Full PEDVINO objective:
#
# L =
#     lambda_pred   * L_prediction
#   + lambda_recon  * L_reconstruction
#   + lambda_energy * L_energy
#   + lambda_grad   * L_gradient
#   + lambda_bc     * L_boundary

LAMBDA_ENERGY = 0.05

LAMBDA_GRAD = 0.02

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
# ORIGINAL VINO EXPERIMENT
# ============================================================

VINO_EPOCHS = 150

VINO_BATCH_SIZE = 20

VINO_LEARNING_RATE = 1e-3

VINO_WEIGHT_DECAY = 1e-5

VINO_MODES_X = 12

VINO_MODES_Y = 12

# Selected to keep the model close to the requested
# ~84k trainable parameter budget.
VINO_WIDTH = 8

VINO_DEPTH = 4

VINO_GAMMA = 0.5

VINO_PATIENCE = 30

# Data-enhanced VINO:
# total loss = data loss + lambda_vino * variational loss
VINO_USE_DATA = True

VINO_LAMBDA_DATA = 1.0

VINO_LAMBDA_PHYSICS = 0.05

VINO_GRADIENT_CLIP = 1.0
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

VINO_CHECKPOINT_PATH = os.path.join(
    VINO_RESULTS_DIR,
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

VINO_HISTORY_PATH = os.path.join(
    VINO_RESULTS_DIR,
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

VINO_METRICS_PATH = os.path.join(
    VINO_RESULTS_DIR,
    "metrics.json",
)
# ============================================================
# CONFIG VALIDATION
# ============================================================

assert GRID_SIZE_X > 2
assert GRID_SIZE_Y > 2

assert NUM_SAMPLES > 0

assert BATCH_SIZE > 0

assert OPERATOR_SIZE > 0

assert MODES_X > 0
assert MODES_Y > 0

assert MODES_X <= GRID_SIZE_X // 2
assert MODES_Y <= GRID_SIZE_Y // 2

assert NUM_MODES_X > 0
assert NUM_MODES_Y > 0

assert COEFFICIENT_MIN > 0.0

assert COEFFICIENT_MAX > COEFFICIENT_MIN

assert SOLVER_MAX_ITERATIONS > 0

assert SOLVER_TOLERANCE > 0.0

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
# PRINT CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DARCY FLOW EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print("\nDIRECTORIES")
    print(
        f"Experiment directory : {EXPERIMENT_DIR}"
    )
    print(
        f"Data directory       : {DATA_DIR}"
    )
    print(
        f"Dataset path         : {DATASET_PATH}"
    )
    print(
        f"Baseline results     : {BASELINE_RESULTS_DIR}"
    )
    print(
        f"PEDVINO results      : {PEDVINO_RESULTS_DIR}"
    )

    print("\nDOMAIN")
    print(f"Domain X             : {DOMAIN_X}")
    print(f"Domain Y             : {DOMAIN_Y}")
    print(
        f"Grid                 : "
        f"{GRID_SIZE_X} x {GRID_SIZE_Y}"
    )
    print(f"dx                   : {DX:.6f}")
    print(f"dy                   : {DY:.6f}")

    print("\nDARCY PDE")
    print(
        "Equation             : "
        "-div(a grad(u)) = f"
    )
    print(
        f"Forcing              : {FORCING_VALUE}"
    )
    print(
        f"Coefficient range    : "
        f"[{COEFFICIENT_MIN}, "
        f"{COEFFICIENT_MAX}]"
    )
    print(
        f"Solver iterations    : "
        f"{SOLVER_MAX_ITERATIONS}"
    )
    print(
        f"Solver tolerance     : "
        f"{SOLVER_TOLERANCE}"
    )

    print("\nDATASET")
    print(f"Total samples        : {NUM_SAMPLES}")
    print(f"Train samples        : {TRAIN_SIZE}")
    print(f"Validation samples   : {VAL_SIZE}")
    print(f"Test samples         : {TEST_SIZE}")
    print(
        f"Coefficient modes    : "
        f"{NUM_MODES_X} x {NUM_MODES_Y}"
    )

    print("\nMODEL")
    print(f"T_LEN                : {T_LEN}")
    print(f"Operator size        : {OPERATOR_SIZE}")
    print(
        f"Koopman modes        : "
        f"{MODES_X} x {MODES_Y}"
    )
    print(f"Decompose            : {DECOMPOSE}")
    print(
        f"Physics hidden       : "
        f"{PHYSICS_HIDDEN_SIZE}"
    )

    print("\nTRAINING")
    print(f"Epochs               : {EPOCHS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Weight decay         : {WEIGHT_DECAY}")
    print(f"Scheduler            : {SCHEDULER}")

    print("\nLOSS")
    print(f"Prediction           : {LAMBDA_PRED}")
    print(f"Reconstruction       : {LAMBDA_RECON}")
    print(f"Energy               : {LAMBDA_ENERGY}")
    print(f"Gradient             : {LAMBDA_GRAD}")
    print(f"Boundary             : {LAMBDA_BC}")

    print("\nPHYSICS TRAINING")
    print(
        f"Energy warm-up       : "
        f"{ENERGY_WARMUP_EPOCHS}"
    )
    print(
        f"Gradient warm-up     : "
        f"{GRADIENT_WARMUP_EPOCHS}"
    )
    print(
        f"Boundary warm-up     : "
        f"{BOUNDARY_WARMUP_EPOCHS}"
    )
    print(
        f"Physics ramp         : "
        f"{PHYSICS_RAMP_EPOCHS}"
    )

    print("=" * 70)
