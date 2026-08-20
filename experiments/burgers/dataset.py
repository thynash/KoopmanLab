"""
Burgers Equation Dataset
========================

Generates a reproducible dataset for the 1D viscous Burgers equation:

    u_t + u u_x = nu u_xx

on the periodic domain:

    x in [0, 1)

with:

    u(x + 1, t) = u(x, t)

The operator-learning task is:

    u(x, 0)  --->  u(x, T)

Each sample contains:

    initial_state : [N, 1]
    solution      : [N, 1]

The dataset is generated once using a spectral spatial
discretization and RK4 time integration, then saved as:

    experiments/burgers/data/burgers_dataset.pt

All later experiments must load this same fixed dataset.
"""


import math
import os

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from experiments.burgers import config


# ============================================================
# CREATE PERIODIC GRID
# ============================================================

def create_periodic_grid(
    grid_size,
    domain_x,
    dtype=torch.float32,
):
    """
    Create a periodic 1D grid.

    The endpoint is excluded:

        x_j = x_min + j * dx

    for:

        j = 0, ..., N - 1

    Returns
    -------
    x  : [N]
    dx : float
    """

    x_min = float(domain_x[0])
    x_max = float(domain_x[1])

    length = x_max - x_min

    dx = length / grid_size

    x = (
        torch.arange(
            grid_size,
            dtype=dtype,
        )
        * dx
        + x_min
    )

    return x, dx


# ============================================================
# FOURIER WAVE NUMBERS
# ============================================================

def create_wave_numbers(
    grid_size,
    dx,
    device="cpu",
    dtype=torch.float32,
):
    """
    Create Fourier wave numbers for periodic derivatives.

    If:

        L = N * dx

    then:

        k = 2 pi * fftfreq(N, d=dx)
    """

    frequencies = torch.fft.fftfreq(
        grid_size,
        d=dx,
        device=device,
        dtype=dtype,
    )

    wave_numbers = (
        2.0
        * math.pi
        * frequencies
    )

    return wave_numbers


# ============================================================
# GENERATE SMOOTH INITIAL CONDITIONS
# ============================================================

def generate_initial_conditions(
    num_samples,
    x,
    domain_x,
    num_modes,
    amplitude,
    spectral_decay_power,
    generator=None,
):
    """
    Generate smooth random periodic initial conditions.

    Each sample is:

        u_0(x)
        =
        sum_{k=1}^{K}
        [
            a_k sin(2 pi k x / L)
            +
            b_k cos(2 pi k x / L)
        ]

    Random coefficients decay with frequency:

        scale_k = 1 / k^p

    which prevents excessively oscillatory initial fields.

    Each sample is normalized independently so that its
    maximum absolute value is approximately `amplitude`.

    Returns
    -------
    initial_states : [B, N]
    """

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if num_modes <= 0:
        raise ValueError(
            "num_modes must be positive."
        )

    grid_size = x.numel()

    x_min = float(domain_x[0])
    x_max = float(domain_x[1])

    length = x_max - x_min

    normalized_x = (
        (x - x_min)
        / length
    )

    initial_states = torch.zeros(
        num_samples,
        grid_size,
        dtype=x.dtype,
    )

    for mode in range(
        1,
        num_modes + 1,
    ):

        scale = (
            1.0
            / (mode ** spectral_decay_power)
        )

        sine_coefficients = (
            torch.randn(
                num_samples,
                1,
                generator=generator,
                dtype=x.dtype,
            )
            * scale
        )

        cosine_coefficients = (
            torch.randn(
                num_samples,
                1,
                generator=generator,
                dtype=x.dtype,
            )
            * scale
        )

        phase = (
            2.0
            * math.pi
            * mode
            * normalized_x
        )

        sine_basis = torch.sin(
            phase
        ).view(
            1,
            grid_size,
        )

        cosine_basis = torch.cos(
            phase
        ).view(
            1,
            grid_size,
        )

        initial_states = (
            initial_states
            + sine_coefficients
            * sine_basis
            + cosine_coefficients
            * cosine_basis
        )

    # --------------------------------------------------------
    # Remove sample-wise mean.
    #
    # This keeps the generated fields centered and avoids
    # arbitrary constant offsets dominating the dataset.
    # --------------------------------------------------------

    initial_states = (
        initial_states
        - initial_states.mean(
            dim=1,
            keepdim=True,
        )
    )

    # --------------------------------------------------------
    # Normalize each sample independently.
    # --------------------------------------------------------

    max_abs = (
        initial_states.abs()
        .amax(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1e-8)
    )

    initial_states = (
        float(amplitude)
        * initial_states
        / max_abs
    )

    return initial_states.contiguous()


# ============================================================
# BURGERS RIGHT-HAND SIDE
# ============================================================

def burgers_rhs(
    u,
    wave_numbers,
    viscosity,
):
    """
    Compute:

        u_t = -u u_x + nu u_xx

    using Fourier spectral differentiation.

    Parameters
    ----------
    u : [B, N]
        Current solution state.

    wave_numbers : [N]
        Fourier wave numbers.

    viscosity : float

    Returns
    -------
    rhs : [B, N]
    """

    if u.ndim != 2:
        raise ValueError(
            "u must have shape [B, N]."
        )

    u_hat = torch.fft.fft(
        u,
        dim=-1,
    )

    ik = (
        1j
        * wave_numbers
    )

    ux = torch.fft.ifft(
        ik
        * u_hat,
        dim=-1,
    ).real

    uxx = torch.fft.ifft(
        -(wave_numbers ** 2)
        * u_hat,
        dim=-1,
    ).real

    nonlinear_term = (
        -u * ux
    )

    diffusion_term = (
        float(viscosity)
        * uxx
    )

    return (
        nonlinear_term
        + diffusion_term
    )


# ============================================================
# RK4 BURGERS SOLVER
# ============================================================

def solve_burgers(
    initial_state,
    wave_numbers,
    viscosity,
    dt,
    num_steps,
):
    """
    Solve the periodic viscous Burgers equation:

        u_t + u u_x = nu u_xx

    using classical fourth-order Runge-Kutta time integration.

    Parameters
    ----------
    initial_state : [B, N]
    wave_numbers  : [N]
    viscosity     : float
    dt            : float
    num_steps     : int

    Returns
    -------
    solution : [B, N]
    """

    if initial_state.ndim != 2:
        raise ValueError(
            "initial_state must have shape [B, N]."
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if num_steps <= 0:
        raise ValueError(
            "num_steps must be positive."
        )

    u = initial_state.clone()

    for _ in range(num_steps):

        k1 = burgers_rhs(
            u,
            wave_numbers,
            viscosity,
        )

        k2 = burgers_rhs(
            u + 0.5 * dt * k1,
            wave_numbers,
            viscosity,
        )

        k3 = burgers_rhs(
            u + 0.5 * dt * k2,
            wave_numbers,
            viscosity,
        )

        k4 = burgers_rhs(
            u + dt * k3,
            wave_numbers,
            viscosity,
        )

        u = (
            u
            + (
                dt
                / 6.0
            )
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        if not torch.isfinite(
            u
        ).all():
            raise RuntimeError(
                "Non-finite values encountered while "
                "solving Burgers equation."
            )

    return u.contiguous()


# ============================================================
# MAIN DATASET
# ============================================================

class BurgersDataset(Dataset):
    """
    Dataset for learning:

        u(x, 0) -> u(x, T)

    Returned tensors use channel-last format:

        initial_state : [N, 1]
        solution      : [N, 1]
    """

    def __init__(
        self,
        num_samples=None,
        grid_size=None,
        domain_x=None,
        viscosity=None,
        final_time=None,
        solver_dt=None,
        num_initial_modes=None,
        initial_amplitude=None,
        spectral_decay_power=None,
        generation_batch_size=None,
        seed=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        if num_samples is None:
            num_samples = config.NUM_SAMPLES

        if grid_size is None:
            grid_size = config.GRID_SIZE_X

        if domain_x is None:
            domain_x = config.DOMAIN_X

        if viscosity is None:
            viscosity = config.VISCOSITY

        if final_time is None:
            final_time = config.FINAL_TIME

        if solver_dt is None:
            solver_dt = config.SOLVER_DT

        if num_initial_modes is None:
            num_initial_modes = (
                config.NUM_INITIAL_MODES
            )

        if initial_amplitude is None:
            initial_amplitude = (
                config.INITIAL_AMPLITUDE
            )

        if spectral_decay_power is None:
            spectral_decay_power = (
                config.SPECTRAL_DECAY_POWER
            )

        if generation_batch_size is None:
            generation_batch_size = (
                config.DATA_GENERATION_BATCH_SIZE
            )

        if seed is None:
            seed = config.SEED

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive."
            )

        if grid_size < 8:
            raise ValueError(
                "grid_size must be at least 8."
            )

        if viscosity <= 0.0:
            raise ValueError(
                "viscosity must be positive."
            )

        if final_time <= 0.0:
            raise ValueError(
                "final_time must be positive."
            )

        if solver_dt <= 0.0:
            raise ValueError(
                "solver_dt must be positive."
            )

        if generation_batch_size <= 0:
            raise ValueError(
                "generation_batch_size must be positive."
            )

        # ----------------------------------------------------
        # STORE METADATA
        # ----------------------------------------------------

        self.num_samples = int(
            num_samples
        )

        self.grid_size = int(
            grid_size
        )

        self.domain_x = tuple(
            domain_x
        )

        self.viscosity = float(
            viscosity
        )

        self.initial_time = float(
            config.INITIAL_TIME
        )

        self.final_time = float(
            final_time
        )

        self.time_horizon = (
            self.final_time
            - self.initial_time
        )

        self.solver_dt = float(
            solver_dt
        )

        self.num_initial_modes = int(
            num_initial_modes
        )

        self.initial_amplitude = float(
            initial_amplitude
        )

        self.spectral_decay_power = float(
            spectral_decay_power
        )

        self.generation_batch_size = int(
            generation_batch_size
        )

        self.seed = int(
            seed
        )

        # ----------------------------------------------------
        # RESOLVE NUMBER OF TIME STEPS
        # ----------------------------------------------------

        self.num_steps = int(
            round(
                self.time_horizon
                / self.solver_dt
            )
        )

        if self.num_steps <= 0:
            raise ValueError(
                "Computed number of solver steps must "
                "be positive."
            )

        self.actual_solver_dt = (
            self.time_horizon
            / self.num_steps
        )

        # ----------------------------------------------------
        # REPRODUCIBLE GENERATOR
        # ----------------------------------------------------

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            self.seed
        )

        # ----------------------------------------------------
        # GRID
        # ----------------------------------------------------

        self.x, self.dx = (
            create_periodic_grid(
                grid_size=self.grid_size,
                domain_x=self.domain_x,
            )
        )

        self.wave_numbers = (
            create_wave_numbers(
                grid_size=self.grid_size,
                dx=self.dx,
            )
        )

        # ----------------------------------------------------
        # GENERATE INITIAL CONDITIONS
        # ----------------------------------------------------

        print(
            f"Generating {self.num_samples} "
            "Burgers initial conditions..."
        )

        initial_states = (
            generate_initial_conditions(
                num_samples=self.num_samples,
                x=self.x,
                domain_x=self.domain_x,
                num_modes=self.num_initial_modes,
                amplitude=self.initial_amplitude,
                spectral_decay_power=(
                    self.spectral_decay_power
                ),
                generator=generator,
            )
        )

        # ----------------------------------------------------
        # SOLVE BURGERS EQUATION
        # ----------------------------------------------------

        solutions = []

        print(
            f"Solving Burgers equation for "
            f"{self.num_samples} samples..."
        )

        print(
            f"Grid size: {self.grid_size}"
        )

        print(
            f"Viscosity: {self.viscosity}"
        )

        print(
            f"Final time: {self.final_time}"
        )

        print(
            f"Solver time step: "
            f"{self.actual_solver_dt}"
        )

        print(
            f"Solver steps: {self.num_steps}"
        )

        for start in range(
            0,
            self.num_samples,
            self.generation_batch_size,
        ):

            end = min(
                start
                + self.generation_batch_size,
                self.num_samples,
            )

            batch_initial_state = (
                initial_states[start:end]
            )

            batch_solution = (
                solve_burgers(
                    initial_state=batch_initial_state,
                    wave_numbers=self.wave_numbers,
                    viscosity=self.viscosity,
                    dt=self.actual_solver_dt,
                    num_steps=self.num_steps,
                )
            )

            solutions.append(
                batch_solution
            )

            print(
                f"Solved {end}/{self.num_samples}"
            )

        solutions = torch.cat(
            solutions,
            dim=0,
        ).contiguous()

        # ----------------------------------------------------
        # CHANNEL-LAST FORMAT
        # ----------------------------------------------------

        self.initial_states = (
            initial_states
            .unsqueeze(-1)
            .contiguous()
        )

        self.solutions = (
            solutions
            .unsqueeze(-1)
            .contiguous()
        )

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        if not torch.isfinite(
            self.initial_states
        ).all():
            raise RuntimeError(
                "Non-finite values detected in "
                "Burgers initial states."
            )

        if not torch.isfinite(
            self.solutions
        ).all():
            raise RuntimeError(
                "Non-finite values detected in "
                "Burgers solutions."
            )

        print(
            "=" * 60
        )

        print(
            "Burgers dataset generation completed."
        )

        print(
            "Initial state shape:",
            tuple(
                self.initial_states.shape
            ),
        )

        print(
            "Solution shape:",
            tuple(
                self.solutions.shape
            ),
        )

        print(
            "=" * 60
        )

    # ========================================================
    # DATASET INTERFACE
    # ========================================================

    def __len__(self):

        return self.num_samples

    def __getitem__(
        self,
        index,
    ):

        return (
            self.initial_states[index],
            self.solutions[index],
        )


# ============================================================
# DATASET CREATION
# ============================================================

def create_burgers_dataset():
    """
    Generate the complete Burgers dataset.

    This function generates the dataset from scratch.

    It should normally be called only when creating the
    stored dataset for the first time.
    """

    return BurgersDataset()


# ============================================================
# DATASET SAVE
# ============================================================

def save_dataset(
    dataset,
    path=None,
):
    """
    Save the generated Burgers dataset.

    Stored:

        initial_states
        solutions

    together with:

        x
        dx
        viscosity
        initial_time
        final_time
        time_horizon
        solver_dt
        num_steps
        domain_x
        grid_size
    """

    if path is None:
        path = config.DATASET_PATH

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    payload = {
        "initial_states":
            dataset.initial_states,

        "solutions":
            dataset.solutions,

        "x":
            dataset.x,

        "dx":
            dataset.dx,

        "viscosity":
            dataset.viscosity,

        "initial_time":
            dataset.initial_time,

        "final_time":
            dataset.final_time,

        "time_horizon":
            dataset.time_horizon,

        "solver_dt":
            dataset.actual_solver_dt,

        "num_steps":
            dataset.num_steps,

        "domain_x":
            dataset.domain_x,

        "grid_size":
            dataset.grid_size,
    }

    torch.save(
        payload,
        path,
    )

    print(
        f"\nDataset saved to:\n{path}"
    )


# ============================================================
# DATASET LOAD
# ============================================================

def load_dataset(
    path=None,
):
    """
    Load a previously generated Burgers dataset.

    Important:
    ---------
    Training scripts should use this function rather than
    regenerating the dataset.
    """

    if path is None:
        path = config.DATASET_PATH

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"Burgers dataset not found: {path}\n"
            "Generate it first by running:\n"
            "python -m experiments.burgers.dataset"
        )

    payload = torch.load(
        path,
        map_location="cpu",
    )

    dataset = BurgersDataset.__new__(
        BurgersDataset
    )

    Dataset.__init__(
        dataset
    )

    dataset.initial_states = (
        payload["initial_states"]
        .contiguous()
    )

    dataset.solutions = (
        payload["solutions"]
        .contiguous()
    )

    dataset.x = (
        payload["x"]
        .contiguous()
    )

    dataset.num_samples = (
        dataset.initial_states.shape[0]
    )

    dataset.grid_size = payload.get(
        "grid_size",
        dataset.initial_states.shape[1],
    )

    dataset.dx = float(
        payload["dx"]
    )

    dataset.viscosity = float(
        payload["viscosity"]
    )

    dataset.initial_time = float(
        payload["initial_time"]
    )

    dataset.final_time = float(
        payload["final_time"]
    )

    dataset.time_horizon = float(
        payload["time_horizon"]
    )

    dataset.solver_dt = float(
        payload["solver_dt"]
    )

    dataset.num_steps = int(
        payload["num_steps"]
    )

    dataset.domain_x = tuple(
        payload["domain_x"]
    )

    print(
        f"Loaded Burgers dataset from:\n{path}"
    )

    print(
        f"Samples: {dataset.num_samples}"
    )

    print(
        "Initial state shape:",
        tuple(
            dataset.initial_states.shape
        ),
    )

    print(
        "Solution shape:",
        tuple(
            dataset.solutions.shape
        ),
    )

    print(
        f"Viscosity: {dataset.viscosity}"
    )

    print(
        f"DX: {dataset.dx}"
    )

    print(
        f"Time horizon: "
        f"{dataset.time_horizon}"
    )

    return dataset


# ============================================================
# DATASET SPLIT
# ============================================================

def split_burgers_dataset(
    dataset,
    train_fraction=None,
    val_fraction=None,
    test_fraction=None,
    seed=None,
):
    """
    Reproducibly split the fixed stored dataset into:

        train
        validation
        test
    """

    if train_fraction is None:
        train_fraction = (
            config.TRAIN_FRACTION
        )

    if val_fraction is None:
        val_fraction = (
            config.VAL_FRACTION
        )

    if test_fraction is None:
        test_fraction = (
            config.TEST_FRACTION
        )

    if seed is None:
        seed = config.SEED

    total_fraction = (
        train_fraction
        + val_fraction
        + test_fraction
    )

    if abs(
        total_fraction - 1.0
    ) > 1e-6:
        raise ValueError(
            "Train, validation and test fractions "
            "must sum to 1.0."
        )

    total_samples = len(
        dataset
    )

    train_size = int(
        train_fraction
        * total_samples
    )

    val_size = int(
        val_fraction
        * total_samples
    )

    test_size = (
        total_samples
        - train_size
        - val_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )

    return random_split(
        dataset,
        [
            train_size,
            val_size,
            test_size,
        ],
        generator=generator,
    )


# ============================================================
# DATA LOADERS
# ============================================================

def get_burgers_loaders(
    batch_size=None,
    num_workers=None,
    seed=None,
    dataset_path=None,
):
    """
    Load the already-generated Burgers dataset and create
    reproducible DataLoaders.

    Important:
    ---------
    This function never regenerates the dataset.
    """

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if num_workers is None:
        num_workers = config.NUM_WORKERS

    if seed is None:
        seed = config.SEED

    dataset = load_dataset(
        path=dataset_path
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = split_burgers_dataset(
        dataset=dataset,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
        ),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# MAIN: GENERATE AND SAVE ONCE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BURGERS DATASET GENERATION")
    print("=" * 60)

    print(
        f"Samples        : {config.NUM_SAMPLES}"
    )

    print(
        f"Grid size      : {config.GRID_SIZE_X}"
    )

    print(
        f"Viscosity      : {config.VISCOSITY}"
    )

    print(
        f"Final time     : {config.FINAL_TIME}"
    )

    print(
        f"Solver dt      : {config.SOLVER_DT}"
    )

    print(
        f"Solver steps   : {config.SOLVER_NUM_STEPS}"
    )

    print("=" * 60)

    dataset = create_burgers_dataset()

    save_dataset(
        dataset
    )

    print("\nDataset generation completed successfully.")
