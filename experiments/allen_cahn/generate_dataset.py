# ============================================================
# Allen-Cahn 1D Dataset Generator
#
# PDE:
#
#     u_t = epsilon^2 u_xx + u - u^3
#
# Domain:
#
#     x in [0, 1), periodic
#     t in [0, T]
#
# Operator learning task:
#
#     u_0(x)  --->  u(x, T)
#
# The dataset is generated ONCE and saved. All experiments
# (Baseline KNO and PEDVINO) subsequently use this same file.
# ============================================================

import os
import math
import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_SAMPLES = 2000
NX = 128

LENGTH = 1.0

# Allen-Cahn interface parameter
EPSILON = 0.01

# Final solution time
FINAL_TIME = 0.10

# Stable semi-implicit time step
DT = 1.0e-4

# Number of saved PDE time steps
NUM_STEPS = int(round(FINAL_TIME / DT))

# Initial-condition generation
NUM_MODES = 8
AMPLITUDE_SCALE = 0.35
MEAN_SCALE = 0.15

# Output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DATASET_PATH = os.path.join(
    DATA_DIR,
    "allen_cahn_1d_n2000_nx128.pt",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# INITIAL CONDITIONS
# ============================================================

def generate_initial_conditions(
    num_samples: int,
    nx: int,
    length: float,
    num_modes: int,
):
    """
    Generate smooth random periodic initial conditions.

    Each sample is constructed as:

        u0(x)
        =
        c0
        +
        sum_k [
            a_k sin(2*pi*k*x/L)
            +
            b_k cos(2*pi*k*x/L)
        ]

    Higher-frequency coefficients decay with k so the initial
    conditions remain smooth.

    Returns:
        u0 : float32 array, shape [num_samples, nx]
    """

    x = np.arange(nx, dtype=np.float64) * length / nx

    u0 = np.zeros(
        (num_samples, nx),
        dtype=np.float64,
    )

    # Random mean component
    means = np.random.uniform(
        low=-MEAN_SCALE,
        high=MEAN_SCALE,
        size=(num_samples, 1),
    )

    u0 += means

    for k in range(1, num_modes + 1):

        # Decaying spectral amplitudes
        scale = AMPLITUDE_SCALE / (k ** 1.5)

        a = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(num_samples, 1),
        )

        b = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(num_samples, 1),
        )

        phase = 2.0 * math.pi * k * x / length

        u0 += (
            a * np.sin(phase)[None, :]
            +
            b * np.cos(phase)[None, :]
        )

    # Keep initial conditions in a reasonable Allen-Cahn range.
    # The PDE naturally has stable phases near +/-1.
    max_abs = np.max(
        np.abs(u0),
        axis=1,
        keepdims=True,
    )

    scale = np.maximum(max_abs, 1.0)

    u0 = 0.9 * u0 / scale

    return u0.astype(np.float32)


# ============================================================
# FOURIER WAVENUMBERS
# ============================================================

def build_wavenumber_squared(
    nx: int,
    length: float,
):
    """
    Construct k^2 corresponding to NumPy's FFT ordering.

    For the periodic Laplacian:

        FFT(u_xx) = -(k^2) FFT(u)
    """

    dx = length / nx

    frequencies = np.fft.fftfreq(
        nx,
        d=dx,
    )

    wave_numbers = (
        2.0
        * math.pi
        * frequencies
    )

    return wave_numbers ** 2


# ============================================================
# ALLEN-CAHN SOLVER
# ============================================================

def solve_allen_cahn(
    initial_conditions: np.ndarray,
    epsilon: float,
    final_time: float,
    dt: float,
    length: float,
):
    """
    Solve the periodic 1D Allen-Cahn equation:

        u_t = epsilon^2 u_xx + u - u^3

    using a semi-implicit spectral scheme.

    The diffusion term is treated implicitly:

        u_hat^{n+1}
        =
        [u_hat^n + dt * FFT(u^n - (u^n)^3)]
        /
        [1 + dt * epsilon^2 * k^2]

    This is substantially more stable than explicit Euler for
    the diffusion term.

    Args:
        initial_conditions:
            Array of shape [num_samples, nx]

    Returns:
        solutions:
            Array of shape [num_samples, nx]
            containing u(x, FINAL_TIME).
    """

    u = initial_conditions.astype(
        np.float64,
        copy=True,
    )

    num_samples, nx = u.shape

    num_steps = int(
        round(final_time / dt)
    )

    k_squared = build_wavenumber_squared(
        nx=nx,
        length=length,
    )

    # Implicit denominator for:
    #
    # (I - dt * epsilon^2 * d_xx) u^{n+1}
    #
    # Since Fourier(d_xx u) = -k^2 u_hat:
    #
    # denominator = 1 + dt * epsilon^2 * k^2
    denominator = (
        1.0
        +
        dt
        * (epsilon ** 2)
        * k_squared
    )

    print()
    print("=" * 70)
    print("SOLVING ALLEN-CAHN EQUATION")
    print("=" * 70)
    print(f"Samples       : {num_samples}")
    print(f"Spatial points: {nx}")
    print(f"Domain length : {length}")
    print(f"Epsilon       : {epsilon}")
    print(f"Final time    : {final_time}")
    print(f"Time step     : {dt}")
    print(f"Time steps    : {num_steps}")
    print("=" * 70)

    for step in range(num_steps):

        # Explicit nonlinear/reaction contribution
        reaction = (
            u
            -
            u ** 3
        )

        # Fourier transform of current state
        u_hat = np.fft.fft(
            u,
            axis=1,
        )

        reaction_hat = np.fft.fft(
            reaction,
            axis=1,
        )

        # Semi-implicit update
        u_hat_next = (
            u_hat
            +
            dt * reaction_hat
        ) / denominator[None, :]

        # Return to physical space
        u = np.fft.ifft(
            u_hat_next,
            axis=1,
        ).real

        if (
            (step + 1) % 100 == 0
            or step == 0
            or step + 1 == num_steps
        ):
            print(
                f"Step {step + 1:5d}/{num_steps} | "
                f"u min: {u.min(): .6f} | "
                f"u max: {u.max(): .6f}"
            )

    return u.astype(np.float32)


# ============================================================
# DATASET CREATION
# ============================================================

def main():

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Avoid accidental regeneration.
    # --------------------------------------------------------

    if os.path.exists(DATASET_PATH):

        print()
        print("=" * 70)
        print("ALLEN-CAHN DATASET ALREADY EXISTS")
        print("=" * 70)
        print(DATASET_PATH)
        print()
        print(
            "Delete the file explicitly if you want to "
            "regenerate the dataset."
        )

        loaded = torch.load(
            DATASET_PATH,
            map_location="cpu",
        )

        print()
        print("Stored dataset information:")

        if "initial_state" in loaded:
            print(
                "Initial state shape:",
                tuple(
                    loaded["initial_state"].shape
                ),
            )

        if "solution" in loaded:
            print(
                "Solution shape:",
                tuple(
                    loaded["solution"].shape
                ),
            )

        return

    # --------------------------------------------------------
    # Generate initial conditions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING ALLEN-CAHN DATASET")
    print("=" * 70)

    initial_state = generate_initial_conditions(
        num_samples=NUM_SAMPLES,
        nx=NX,
        length=LENGTH,
        num_modes=NUM_MODES,
    )

    print()
    print(
        "Initial conditions generated:",
        initial_state.shape,
    )

    # --------------------------------------------------------
    # Solve PDE
    # --------------------------------------------------------

    solution = solve_allen_cahn(
        initial_conditions=initial_state,
        epsilon=EPSILON,
        final_time=FINAL_TIME,
        dt=DT,
        length=LENGTH,
    )

    # --------------------------------------------------------
    # Convert to operator-learning format
    #
    # [N, NX] -> [N, NX, 1]
    # --------------------------------------------------------

    initial_tensor = torch.from_numpy(
        initial_state
    ).unsqueeze(-1)

    solution_tensor = torch.from_numpy(
        solution
    ).unsqueeze(-1)

    # Spatial grid
    grid = torch.linspace(
        0.0,
        LENGTH,
        NX + 1,
        dtype=torch.float32,
    )[:-1]

    # --------------------------------------------------------
    # Save complete metadata so all experiments use exactly
    # the same PDE definition.
    # --------------------------------------------------------

    dataset = {
        "initial_state": initial_tensor,
        "solution": solution_tensor,
        "grid": grid,

        "equation": (
            "u_t = epsilon^2 u_xx + u - u^3"
        ),

        "epsilon": EPSILON,
        "length": LENGTH,
        "final_time": FINAL_TIME,
        "dt": DT,

        "num_samples": NUM_SAMPLES,
        "nx": NX,

        "boundary_condition": "periodic",
        "seed": SEED,
    }

    torch.save(
        dataset,
        DATASET_PATH,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALLEN-CAHN DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Saved to: {DATASET_PATH}")
    print()
    print(
        "Initial state shape:",
        tuple(initial_tensor.shape),
    )
    print(
        "Solution shape:",
        tuple(solution_tensor.shape),
    )
    print()
    print(
        "Initial state range:",
        f"[{initial_tensor.min().item():.6f}, "
        f"{initial_tensor.max().item():.6f}]",
    )
    print(
        "Solution range     :",
        f"[{solution_tensor.min().item():.6f}, "
        f"{solution_tensor.max().item():.6f}]",
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
