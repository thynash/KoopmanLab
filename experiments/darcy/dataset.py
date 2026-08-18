"""
Darcy Flow Dataset
==================

Generates a dataset for the variable-coefficient elliptic PDE:

    -div(a(x, y) grad(u(x, y))) = f(x, y)

on a rectangular domain with homogeneous Dirichlet boundary
conditions:

    u = 0 on the boundary.

The operator-learning task is:

    coefficient a(x, y)  --->  solution u(x, y)

Each dataset sample returns:

    coefficient : [H, W, 1]
    solution    : [H, W, 1]
    forcing     : [H, W, 1]

The forcing is returned explicitly because the general variational
framework requires it for the Darcy energy functional:

    Pi[u] = Integral[
        1/2 a |grad(u)|^2 - f u
    ] dOmega

This file expects the final config.py to define the configuration
variables referenced below.
"""


import math
import os
import torch

from torch.utils.data import Dataset, DataLoader, random_split

from experiments.darcy import config


# ============================================================
# HELPER: CREATE GRID
# ============================================================

def create_grid(
    grid_size_x,
    grid_size_y,
    domain_x,
    domain_y,
    dtype=torch.float32,
):
    """
    Create a 2D Cartesian grid.

    Returns
    -------
    x : [H, W]
    y : [H, W]
    dx : float
    dy : float
    """

    x_values = torch.linspace(
        domain_x[0],
        domain_x[1],
        grid_size_x,
        dtype=dtype,
    )

    y_values = torch.linspace(
        domain_y[0],
        domain_y[1],
        grid_size_y,
        dtype=dtype,
    )

    # indexing="ij" gives:
    #
    # x.shape = [H, W]
    # y.shape = [H, W]
    #
    x, y = torch.meshgrid(
        x_values,
        y_values,
        indexing="ij",
    )

    dx = (
        domain_x[1] - domain_x[0]
    ) / (
        grid_size_x - 1
    )

    dy = (
        domain_y[1] - domain_y[0]
    ) / (
        grid_size_y - 1
    )

    return x, y, dx, dy


# ============================================================
# HELPER: SMOOTH RANDOM COEFFICIENT FIELD
# ============================================================

def generate_coefficient_field(
    x,
    y,
    num_modes_x,
    num_modes_y,
    min_coefficient,
    max_coefficient,
    generator=None,
):
    """
    Generate a smooth strictly-positive permeability field.

    The field is constructed from a random truncated Fourier series:

        raw(x, y)
            =
            sum c_mn sin(...) sin(...)
            +
            sum d_mn cos(...) cos(...)

    followed by exponentiation / normalization so that:

        min_coefficient <= a(x,y) <= max_coefficient

    Positivity is essential for the elliptic Darcy problem.
    """

    height, width = x.shape

    device = x.device
    dtype = x.dtype

    domain_x_length = (
        x.max() - x.min()
    )

    domain_y_length = (
        y.max() - y.min()
    )

    raw_field = torch.zeros(
        height,
        width,
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # Random smooth Fourier modes
    # --------------------------------------------------------

    for mode_x in range(1, num_modes_x + 1):

        for mode_y in range(1, num_modes_y + 1):

            coefficient_scale = (
                1.0
                /
                (
                    mode_x ** 2
                    +
                    mode_y ** 2
                )
            )

            sine_coefficient = torch.randn(
                (),
                generator=generator,
                device=device,
                dtype=dtype,
            ) * coefficient_scale

            cosine_coefficient = torch.randn(
                (),
                generator=generator,
                device=device,
                dtype=dtype,
            ) * coefficient_scale

            phase_x = (
                2.0
                * math.pi
                * mode_x
                * (
                    x - x.min()
                )
                / domain_x_length
            )

            phase_y = (
                2.0
                * math.pi
                * mode_y
                * (
                    y - y.min()
                )
                / domain_y_length
            )

            raw_field = (
                raw_field
                + sine_coefficient
                * torch.sin(phase_x)
                * torch.sin(phase_y)
                + cosine_coefficient
                * torch.cos(phase_x)
                * torch.cos(phase_y)
            )

    # --------------------------------------------------------
    # Normalize sample to approximately [-1, 1]
    # --------------------------------------------------------

    raw_min = raw_field.min()
    raw_max = raw_field.max()

    denominator = (
        raw_max - raw_min
    ).clamp_min(1e-8)

    normalized = (
        2.0
        * (
            raw_field - raw_min
        )
        / denominator
        - 1.0
    )

    # --------------------------------------------------------
    # Map to strictly positive permeability interval
    # --------------------------------------------------------

    coefficient = (
        min_coefficient
        +
        0.5
        * (
            normalized + 1.0
        )
        * (
            max_coefficient
            - min_coefficient
        )
    )

    return coefficient


# ============================================================
# HELPER: GENERATE FORCING
# ============================================================

def generate_forcing_field(
    x,
    y,
    forcing_type="constant",
    forcing_value=1.0,
):
    """
    Generate the forcing f(x,y).

    Supported forcing types:

        "constant"
            f(x,y) = forcing_value

        "sine"
            f(x,y) =
                forcing_value *
                sin(pi * x_normalized)
                sin(pi * y_normalized)

    Returning forcing explicitly keeps the dataset compatible with
    the general PDE functional framework.
    """

    forcing_type = forcing_type.lower()

    if forcing_type == "constant":

        forcing = torch.full_like(
            x,
            float(forcing_value),
        )

    elif forcing_type == "sine":

        x_normalized = (
            x - x.min()
        ) / (
            x.max() - x.min()
        ).clamp_min(1e-8)

        y_normalized = (
            y - y.min()
        ) / (
            y.max() - y.min()
        ).clamp_min(1e-8)

        forcing = (
            float(forcing_value)
            * torch.sin(
                math.pi * x_normalized
            )
            * torch.sin(
                math.pi * y_normalized
            )
        )

    else:
        raise ValueError(
            f"Unsupported forcing_type: {forcing_type}. "
            "Supported values are 'constant' and 'sine'."
        )

    return forcing


# ============================================================
# DARCY FINITE-DIFFERENCE SOLVER
# ============================================================

def solve_darcy(
    coefficient,
    forcing,
    dx,
    dy,
    max_iterations=500,
    tolerance=1e-6,
):
    """
    Solve:

        -div(a grad(u)) = f

    using a matrix-free iterative finite-difference solver.

    Boundary condition:

        u = 0 on boundary.

    Discretization
    --------------

    At each interior point:

        -(a u_x)_x - (a u_y)_y = f

    Face coefficients are approximated using arithmetic averages:

        a_{i+1/2,j}
        =
        1/2 (a_{i,j} + a_{i+1,j})

    The resulting discrete equation is solved using Jacobi
    iterations.

    Parameters
    ----------
    coefficient : [H, W]
    forcing     : [H, W]
    dx, dy      : spatial grid spacings
    max_iterations : maximum Jacobi iterations
    tolerance      : stopping tolerance

    Returns
    -------
    solution : [H, W]
    """

    if coefficient.ndim != 2:
        raise ValueError(
            "coefficient must have shape [H, W]."
        )

    if forcing.shape != coefficient.shape:
        raise ValueError(
            "forcing and coefficient must have identical shapes."
        )

    height, width = coefficient.shape

    if height < 3 or width < 3:
        raise ValueError(
            "Grid dimensions must both be at least 3."
        )

    device = coefficient.device
    dtype = coefficient.dtype

    # Homogeneous Dirichlet boundary condition.
    solution = torch.zeros(
        height,
        width,
        device=device,
        dtype=dtype,
    )

    dx2 = dx * dx
    dy2 = dy * dy

    # --------------------------------------------------------
    # Face coefficients
    # --------------------------------------------------------
    #
    # ax_plus  corresponds to a_{i+1/2,j}
    # ax_minus corresponds to a_{i-1/2,j}
    #
    # ay_plus  corresponds to a_{i,j+1/2}
    # ay_minus corresponds to a_{i,j-1/2}
    #
    # All are defined on interior points.
    # --------------------------------------------------------

    center = coefficient[1:-1, 1:-1]

    ax_plus = 0.5 * (
        center
        + coefficient[2:, 1:-1]
    )

    ax_minus = 0.5 * (
        center
        + coefficient[:-2, 1:-1]
    )

    ay_plus = 0.5 * (
        center
        + coefficient[1:-1, 2:]
    )

    ay_minus = 0.5 * (
        center
        + coefficient[1:-1, :-2]
    )

    diagonal = (
        (ax_plus + ax_minus) / dx2
        +
        (ay_plus + ay_minus) / dy2
    ).clamp_min(1e-12)

    interior_forcing = forcing[
        1:-1,
        1:-1,
    ]

    # ========================================================
    # JACOBI ITERATION
    # ========================================================

    for _ in range(max_iterations):

        previous_solution = solution

        updated_solution = solution.clone()

        numerator = (
            ax_plus
            * previous_solution[
                2:,
                1:-1,
            ]
            / dx2
            +
            ax_minus
            * previous_solution[
                :-2,
                1:-1,
            ]
            / dx2
            +
            ay_plus
            * previous_solution[
                1:-1,
                2:,
            ]
            / dy2
            +
            ay_minus
            * previous_solution[
                1:-1,
                :-2,
            ]
            / dy2
            +
            interior_forcing
        )

        updated_solution[
            1:-1,
            1:-1,
        ] = numerator / diagonal

        difference = torch.max(
            torch.abs(
                updated_solution - previous_solution
            )
        )

        solution = updated_solution

        if difference.item() < tolerance:
            break

    return solution


# ============================================================
# MAIN DATASET
# ============================================================

class DarcyFlowDataset(Dataset):
    """
    Dataset for learning the Darcy solution operator:

        a(x,y) ---> u(x,y)

    Internally each sample also stores f(x,y), allowing the
    variational physics loss to evaluate:

        1/2 a |grad(u)|^2 - f u

    Returned tensors use channel-last format:

        coefficient : [H, W, 1]
        solution    : [H, W, 1]
        forcing     : [H, W, 1]
    """

    def __init__(
        self,
        num_samples=None,
        grid_size_x=None,
        grid_size_y=None,
        domain_x=None,
        domain_y=None,
        num_modes_x=None,
        num_modes_y=None,
        min_coefficient=None,
        max_coefficient=None,
        forcing_type=None,
        forcing_value=None,
        solver_iterations=None,
        solver_tolerance=None,
        seed=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Read configuration lazily.
        #
        # This allows config.py to be written last.
        # ----------------------------------------------------

        if num_samples is None:
            num_samples = config.NUM_SAMPLES

        if grid_size_x is None:
            grid_size_x = config.GRID_SIZE_X

        if grid_size_y is None:
            grid_size_y = config.GRID_SIZE_Y

        if domain_x is None:
            domain_x = config.DOMAIN_X

        if domain_y is None:
            domain_y = config.DOMAIN_Y

        if num_modes_x is None:
            num_modes_x = config.NUM_MODES_X

        if num_modes_y is None:
            num_modes_y = config.NUM_MODES_Y

        if min_coefficient is None:
            min_coefficient = config.MIN_COEFFICIENT

        if max_coefficient is None:
            max_coefficient = config.MAX_COEFFICIENT

        if forcing_type is None:
            forcing_type = config.FORCING_TYPE

        if forcing_value is None:
            forcing_value = config.FORCING_VALUE

        if solver_iterations is None:
            solver_iterations = config.SOLVER_MAX_ITERATIONS

        if solver_tolerance is None:
            solver_tolerance = config.SOLVER_TOLERANCE

        if seed is None:
            seed = config.SEED

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive."
            )

        if grid_size_x < 3 or grid_size_y < 3:
            raise ValueError(
                "Both grid dimensions must be >= 3."
            )

        if min_coefficient <= 0:
            raise ValueError(
                "min_coefficient must be strictly positive."
            )

        if max_coefficient <= min_coefficient:
            raise ValueError(
                "max_coefficient must be greater than "
                "min_coefficient."
            )

        self.num_samples = int(num_samples)

        self.grid_size_x = int(grid_size_x)
        self.grid_size_y = int(grid_size_y)

        self.domain_x = tuple(domain_x)
        self.domain_y = tuple(domain_y)

        self.num_modes_x = int(num_modes_x)
        self.num_modes_y = int(num_modes_y)

        self.min_coefficient = float(min_coefficient)
        self.max_coefficient = float(max_coefficient)

        self.forcing_type = forcing_type
        self.forcing_value = float(forcing_value)

        self.solver_iterations = int(
            solver_iterations
        )

        self.solver_tolerance = float(
            solver_tolerance
        )

        self.seed = int(seed)

        # ----------------------------------------------------
        # Reproducible CPU random generator
        # ----------------------------------------------------

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            self.seed
        )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        (
            x,
            y,
            self.dx,
            self.dy,
        ) = create_grid(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            domain_x=self.domain_x,
            domain_y=self.domain_y,
        )

        # Fixed forcing for all samples.
        #
        # The coefficient changes sample-to-sample.
        # This creates the operator-learning mapping:
        #
        #     a(x,y) -> u(x,y)
        #
        forcing = generate_forcing_field(
            x=x,
            y=y,
            forcing_type=self.forcing_type,
            forcing_value=self.forcing_value,
        )

        # ----------------------------------------------------
        # Storage
        # ----------------------------------------------------

        coefficients = []
        solutions = []
        forcings = []

        print(
            f"Generating {self.num_samples} Darcy samples..."
        )

        # ====================================================
        # GENERATE SAMPLES
        # ====================================================

        for sample_index in range(
            self.num_samples
        ):

            coefficient = (
                generate_coefficient_field(
                    x=x,
                    y=y,
                    num_modes_x=self.num_modes_x,
                    num_modes_y=self.num_modes_y,
                    min_coefficient=self.min_coefficient,
                    max_coefficient=self.max_coefficient,
                    generator=generator,
                )
            )

            solution = solve_darcy(
                coefficient=coefficient,
                forcing=forcing,
                dx=self.dx,
                dy=self.dy,
                max_iterations=self.solver_iterations,
                tolerance=self.solver_tolerance,
            )

            # Channel-last representation expected by
            # the current operator-learning pipeline.
            coefficients.append(
                coefficient.unsqueeze(-1)
            )

            solutions.append(
                solution.unsqueeze(-1)
            )

            forcings.append(
                forcing.unsqueeze(-1)
            )

            if (
                (sample_index + 1) % 100 == 0
                or sample_index == 0
                or sample_index + 1
                == self.num_samples
            ):

                print(
                    f"Generated "
                    f"{sample_index + 1}/"
                    f"{self.num_samples}"
                )

        # ----------------------------------------------------
        # Convert to contiguous tensors
        # ----------------------------------------------------

        self.coefficients = torch.stack(
            coefficients,
            dim=0,
        ).contiguous()

        self.solutions = torch.stack(
            solutions,
            dim=0,
        ).contiguous()

        self.forcings = torch.stack(
            forcings,
            dim=0,
        ).contiguous()

        # ----------------------------------------------------
        # Basic safety checks
        # ----------------------------------------------------

        if not torch.isfinite(
            self.coefficients
        ).all():
            raise RuntimeError(
                "Non-finite coefficient values detected."
            )

        if not torch.isfinite(
            self.solutions
        ).all():
            raise RuntimeError(
                "Non-finite Darcy solution values detected."
            )

        if not torch.isfinite(
            self.forcings
        ).all():
            raise RuntimeError(
                "Non-finite forcing values detected."
            )

        print(
            "Darcy dataset generation completed."
        )

        print(
            "Coefficient shape:",
            tuple(self.coefficients.shape),
        )

        print(
            "Solution shape:",
            tuple(self.solutions.shape),
        )

        print(
            "Forcing shape:",
            tuple(self.forcings.shape),
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
            self.coefficients[index],
            self.solutions[index],
            self.forcings[index],
        )


# ============================================================
# DATASET CREATION
# ============================================================

def create_darcy_dataset():
    """
    Create the complete Darcy dataset.

    The final config.py must provide:

        NUM_SAMPLES
        GRID_SIZE_X
        GRID_SIZE_Y
        DOMAIN_X
        DOMAIN_Y
        NUM_MODES_X
        NUM_MODES_Y
        MIN_COEFFICIENT
        MAX_COEFFICIENT
        FORCING_TYPE
        FORCING_VALUE
        SOLVER_MAX_ITERATIONS
        SOLVER_TOLERANCE
        SEED
    """

    return DarcyFlowDataset()


# ============================================================
# DATASET SAVE
# ============================================================

def save_dataset(
    dataset,
    path=None,
):
    """
    Save the generated Darcy dataset.

    Stored tensors:

        coefficients
        solutions
        forcings

    together with spatial metadata.
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
        "coefficients":
            dataset.coefficients,

        "solutions":
            dataset.solutions,

        "forcings":
            dataset.forcings,

        "dx":
            dataset.dx,

        "dy":
            dataset.dy,

        "domain_x":
            dataset.domain_x,

        "domain_y":
            dataset.domain_y,

        "grid_size_x":
            dataset.grid_size_x,

        "grid_size_y":
            dataset.grid_size_y,
    }

    torch.save(
        payload,
        path,
    )

    print(
        f"Dataset saved to: {path}"
    )


# ============================================================
# DATASET LOAD
# ============================================================

def load_dataset(
    path=None,
):
    """
    Load a previously generated Darcy dataset.

    Returns a DarcyFlowDataset-like object with all required
    tensors and metadata restored.
    """

    if path is None:
        path = config.DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Darcy dataset not found: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
    )

    dataset = DarcyFlowDataset.__new__(
        DarcyFlowDataset
    )

    Dataset.__init__(
        dataset
    )

    dataset.coefficients = (
        payload["coefficients"].contiguous()
    )

    dataset.solutions = (
        payload["solutions"].contiguous()
    )

    dataset.forcings = (
        payload["forcings"].contiguous()
    )

    dataset.num_samples = (
        dataset.coefficients.shape[0]
    )

    dataset.grid_size_x = payload.get(
        "grid_size_x",
        dataset.coefficients.shape[1],
    )

    dataset.grid_size_y = payload.get(
        "grid_size_y",
        dataset.coefficients.shape[2],
    )

    dataset.dx = payload["dx"]
    dataset.dy = payload["dy"]

    dataset.domain_x = tuple(
        payload["domain_x"]
    )

    dataset.domain_y = tuple(
        payload["domain_y"]
    )

    print(
        f"Loaded Darcy dataset from: {path}"
    )

    print(
        f"Samples: {dataset.num_samples}"
    )

    print(
        "Coefficient shape:",
        tuple(dataset.coefficients.shape),
    )

    print(
        "Solution shape:",
        tuple(dataset.solutions.shape),
    )

    return dataset


# ============================================================
# SPLIT DATASET
# ============================================================

def split_darcy_dataset(
    dataset,
    train_fraction=None,
    val_fraction=None,
    test_fraction=None,
    seed=None,
):
    """
    Split the Darcy dataset reproducibly into:

        train
        validation
        test
    """

    if train_fraction is None:
        train_fraction = config.TRAIN_FRACTION

    if val_fraction is None:
        val_fraction = config.VAL_FRACTION

    if test_fraction is None:
        test_fraction = config.TEST_FRACTION

    if seed is None:
        seed = config.SEED

    total_fraction = (
        train_fraction
        + val_fraction
        + test_fraction
    )

    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation and test fractions "
            "must sum to 1.0."
        )

    total_samples = len(dataset)

    train_size = int(
        train_fraction * total_samples
    )

    val_size = int(
        val_fraction * total_samples
    )

    test_size = (
        total_samples
        - train_size
        - val_size
    )

    if min(
        train_size,
        val_size,
        test_size,
    ) <= 0:
        raise ValueError(
            "Dataset split produced an empty subset."
        )

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )

    train_dataset, val_dataset, test_dataset = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
                test_size,
            ],
            generator=generator,
        )
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DATA LOADERS
# ============================================================

def get_darcy_loaders(
    batch_size=None,
    num_workers=None,
    seed=None,
):
    """
    Return:

        train_loader
        val_loader
        test_loader

    Dataset behaviour:

    If config.REGENERATE_DATASET is True:
        generate and save a fresh dataset.

    Otherwise:
        load DATASET_PATH if it exists.

    If the dataset does not exist:
        generate and save it automatically.
    """

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if num_workers is None:
        num_workers = config.NUM_WORKERS

    if seed is None:
        seed = config.SEED

    regenerate = getattr(
        config,
        "REGENERATE_DATASET",
        False,
    )

    dataset_exists = os.path.exists(
        config.DATASET_PATH
    )

    # --------------------------------------------------------
    # Generate or load
    # --------------------------------------------------------

    if regenerate or not dataset_exists:

        dataset = create_darcy_dataset()

        save_dataset(
            dataset,
            config.DATASET_PATH,
        )

    else:

        dataset = load_dataset(
            config.DATASET_PATH
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = split_darcy_dataset(
        dataset=dataset,
        seed=seed,
    )

    # --------------------------------------------------------
    # Loader settings
    # --------------------------------------------------------

    pin_memory = getattr(
        config,
        "PIN_MEMORY",
        torch.cuda.is_available(),
    )

    persistent_workers = (
        getattr(
            config,
            "PERSISTENT_WORKERS",
            False,
        )
        and num_workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DARCY FLOW DATASET TEST")
    print("=" * 70)

    dataset = create_darcy_dataset()

    print()
    print(
        f"Number of samples: {len(dataset)}"
    )

    coefficient, solution, forcing = dataset[0]

    print(
        f"Coefficient shape: {tuple(coefficient.shape)}"
    )

    print(
        f"Solution shape   : {tuple(solution.shape)}"
    )

    print(
        f"Forcing shape    : {tuple(forcing.shape)}"
    )

    print()
    print(
        f"Coefficient min/max: "
        f"{coefficient.min().item():.6f} / "
        f"{coefficient.max().item():.6f}"
    )

    print(
        f"Solution min/max   : "
        f"{solution.min().item():.6f} / "
        f"{solution.max().item():.6f}"
    )

    print(
        f"Forcing min/max    : "
        f"{forcing.min().item():.6f} / "
        f"{forcing.max().item():.6f}"
    )

    print("=" * 70)
    print("DATASET TEST COMPLETED")
    print("=" * 70)
