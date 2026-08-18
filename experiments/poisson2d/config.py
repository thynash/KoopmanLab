import os


# ============================================================
# EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "poisson2d"

EXPERIMENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        EXPERIMENT_DIR,
        "..",
        "..",
    )
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# DIRECTORIES
# ============================================================

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


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_PATH = os.path.join(
    DATA_DIR,
    "poisson2d_dataset.pt",
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)

os.makedirs(
    RESULTS_DIR,
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
# DOMAIN
# ============================================================

DOMAIN_X = (
    0.0,
    1.0,
)

DOMAIN_Y = (
    0.0,
    1.0,
)


# ============================================================
# SPATIAL GRID
# ============================================================

GRID_SIZE_X = 64

GRID_SIZE_Y = 64

# Backward compatibility
GRID_SIZE = GRID_SIZE_X


# ============================================================
# SPATIAL DISCRETIZATION
# ============================================================

DX = (
    DOMAIN_X[1] - DOMAIN_X[0]
) / (
    GRID_SIZE_X - 1
)

DY = (
    DOMAIN_Y[1] - DOMAIN_Y[0]
) / (
    GRID_SIZE_Y - 1
)


# ============================================================
# DATASET SIZE
# ============================================================

NUM_SAMPLES = 2000


# ============================================================
# DATASET SPLIT
# ============================================================

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


# ============================================================
# POISSON FORCING GENERATION
#
# These modes are ONLY for generating the dataset.
#
# They are NOT Koopman Fourier modes.
# ============================================================

NUM_MODES_X = 4

NUM_MODES_Y = 4


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 4

PIN_MEMORY = True

PERSISTENT_WORKERS = (
    NUM_WORKERS > 0
)


# ============================================================
# INPUT / OUTPUT DIMENSIONS
#
# Poisson:
#
#     -Δu = f
#
# Input:
#     forcing f
#
# Output:
#     solution u
# ============================================================

T_LEN = 1

INPUT_CHANNELS = 1

OUTPUT_CHANNELS = 1


# ============================================================
# KNO / PEDVINO LATENT DIMENSION
#
# Reduced from the previous larger architecture to keep
# PEDVINO efficient and prevent parameter count alone from
# becoming the reason for performance degradation.
# ============================================================

OPERATOR_SIZE = 24


# ============================================================
# KOOPMAN SPECTRAL MODES
#
# These are used by KNO2d.
#
# Different from NUM_MODES_X/Y used for dataset generation.
# ============================================================

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

EPOCHS = 120

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5


# ============================================================
# OPTIMIZER
# ============================================================

OPTIMIZER = "adamw"


# ============================================================
# SCHEDULER
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

EARLY_STOPPING_PATIENCE = 25

EARLY_STOPPING_MIN_DELTA = 1e-4


# ============================================================
# PEDVINO LOSS
#
# Updated objective:
#
# L =
#
#   lambda_pred   * L_prediction
# + lambda_recon  * L_reconstruction
# + lambda_energy * L_energy
# + lambda_grad   * L_gradient
# + lambda_bc     * L_boundary
#
#
# IMPORTANT:
#
# The old variables:
#
#     LAMBDA_VAR
#     LAMBDA_CONSISTENCY
#
# are intentionally removed because the updated PEDVINOLoss
# no longer uses those arguments.
# ============================================================


# ------------------------------------------------------------
# 1. DATA SUPERVISION
# ------------------------------------------------------------

LAMBDA_PRED = 1.0


# ------------------------------------------------------------
# 2. AUTOENCODER RECONSTRUCTION
# ------------------------------------------------------------

LAMBDA_RECON = 0.02


# ------------------------------------------------------------
# 3. VARIATIONAL / ENERGY MATCHING
#
# This is deliberately moderate.
#
# It should regularize the operator rather than dominate
# the supervised prediction objective.
# ------------------------------------------------------------

LAMBDA_ENERGY = 0.05


# ------------------------------------------------------------
# 4. GRADIENT / DIFFERENTIAL CONSISTENCY
#
# Encourages the predicted solution to preserve spatial
# differential structure.
# ------------------------------------------------------------

LAMBDA_GRAD = 0.02


# ------------------------------------------------------------
# 5. BOUNDARY CONDITION LOSS
# ------------------------------------------------------------

LAMBDA_BC = 0.05


# ============================================================
# PHYSICS WARM-UP
#
# The physics terms should not dominate at initialization.
#
# During early training, the model first learns the mapping:
#
#       f -> u
#
# Then the physics regularization gradually becomes active.
# ============================================================

PHYSICS_WARMUP_EPOCHS = 15


# ============================================================
# INDIVIDUAL LOSS WARM-UP
#
# Useful if train_pedvino.py controls the loss weights
# independently.
# ============================================================

ENERGY_WARMUP_EPOCHS = 15

GRAD_WARMUP_EPOCHS = 15

BC_WARMUP_EPOCHS = 10


# ============================================================
# LOSS SETTINGS
# ============================================================

USE_RELATIVE_PREDICTION_LOSS = True

LOSS_EPS = 1e-8


# ============================================================
# ENERGY STABILITY
#
# Energy itself may mathematically be negative for some PDE
# formulations, but the optimization penalty should remain
# non-negative.
#
# These settings are consumed only if implemented in the
# current PEDVINOLoss / variational engine.
# ============================================================

ENERGY_LOSS_TYPE = "relative"

ENERGY_EPS = 1e-8


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

SAVE_CHECKPOINT = True


# ============================================================
# FAIR KNO vs PEDVINO COMPARISON
#
# Both models should use:
#
# - same dataset
# - same split
# - same batch size
# - same optimizer
# - same learning rate
# - same epochs
#
# PEDVINO differs only through the proposed architecture
# and physics-aware objective.
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
# SUMMARY PATHS
# ============================================================

BASELINE_SUMMARY_PATH = os.path.join(
    BASELINE_RESULTS_DIR,
    "summary.json",
)

PEDVINO_SUMMARY_PATH = os.path.join(
    PEDVINO_RESULTS_DIR,
    "summary.json",
)

COMPARISON_SUMMARY_PATH = os.path.join(
    COMPARISON_RESULTS_DIR,
    "comparison_summary.json",
)


# ============================================================
# CONFIG VALIDATION
# ============================================================

assert GRID_SIZE_X > 2
assert GRID_SIZE_Y > 2

assert NUM_SAMPLES > 0

assert BATCH_SIZE > 0

assert OPERATOR_SIZE > 0

assert PHYSICS_HIDDEN_SIZE > 0

assert EPOCHS > 0

assert LEARNING_RATE > 0

assert DX > 0
assert DY > 0

assert (
    abs(
        TRAIN_FRACTION
        + VAL_FRACTION
        + TEST_FRACTION
        - 1.0
    )
    < 1e-6
)

assert MODES_X <= GRID_SIZE_X // 2

assert MODES_Y <= (
    GRID_SIZE_Y // 2 + 1
)

assert NUM_MODES_X > 0

assert NUM_MODES_Y > 0

assert LAMBDA_PRED >= 0

assert LAMBDA_RECON >= 0

assert LAMBDA_ENERGY >= 0

assert LAMBDA_GRAD >= 0

assert LAMBDA_BC >= 0


# ============================================================
# PRINT CONFIG
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("POISSON2D EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print("\nDIRECTORIES")
    print(f"Experiment directory : {EXPERIMENT_DIR}")
    print(f"Data directory       : {DATA_DIR}")
    print(f"Dataset path         : {DATASET_PATH}")
    print(f"KNO results          : {KNO_RESULTS_DIR}")
    print(f"PEDVINO results      : {PEDVINO_RESULTS_DIR}")

    print("\nDOMAIN")
    print(f"Domain X             : {DOMAIN_X}")
    print(f"Domain Y             : {DOMAIN_Y}")
    print(f"Grid                 : {GRID_SIZE_X} x {GRID_SIZE_Y}")
    print(f"dx                   : {DX}")
    print(f"dy                   : {DY}")

    print("\nDATASET")
    print(f"Total samples        : {NUM_SAMPLES}")
    print(f"Train samples        : {TRAIN_SIZE}")
    print(f"Validation samples   : {VAL_SIZE}")
    print(f"Test samples         : {TEST_SIZE}")
    print(f"Forcing modes        : {NUM_MODES_X} x {NUM_MODES_Y}")

    print("\nMODEL")
    print(f"T_LEN                : {T_LEN}")
    print(f"Operator size        : {OPERATOR_SIZE}")
    print(f"Koopman modes        : {MODES_X} x {MODES_Y}")
    print(f"Decompose            : {DECOMPOSE}")
    print(f"Physics hidden       : {PHYSICS_HIDDEN_SIZE}")

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

    print("\nWARM-UP")
    print(f"Physics warm-up      : {PHYSICS_WARMUP_EPOCHS}")
    print(f"Energy warm-up       : {ENERGY_WARMUP_EPOCHS}")
    print(f"Gradient warm-up     : {GRAD_WARMUP_EPOCHS}")
    print(f"Boundary warm-up     : {BC_WARMUP_EPOCHS}")

    print("=" * 70)
