import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from experiments.poisson2d import config


class Poisson2DDataset(Dataset):
    """
    Exact manufactured dataset for:

        -Delta(u) = f

    on [0, 1]^2 with:

        u = 0 on the boundary.

    Operator learned:

        f(x, y) -> u(x, y)
    """

    def __init__(
        self,
        num_samples=config.NUM_SAMPLES,
        grid_size_x=config.GRID_SIZE_X,
        grid_size_y=config.GRID_SIZE_Y,
        num_modes_x=config.NUM_MODES_X,
        num_modes_y=config.NUM_MODES_Y,
        seed=config.SEED,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_modes_x = num_modes_x
        self.num_modes_y = num_modes_y

        generator = torch.Generator()
        generator.manual_seed(seed)

        # Spatial grid
        x = torch.linspace(
            config.DOMAIN_X[0],
            config.DOMAIN_X[1],
            grid_size_x,
        )

        y = torch.linspace(
            config.DOMAIN_Y[0],
            config.DOMAIN_Y[1],
            grid_size_y,
        )

        X, Y = torch.meshgrid(
            x,
            y,
            indexing="ij",
        )

        self.x = x
        self.y = y
        self.X = X
        self.Y = Y

        # Output tensors:
        #
        # [samples, H, W, 1]
        #
        self.forcing = torch.zeros(
            num_samples,
            grid_size_x,
            grid_size_y,
            1,
            dtype=torch.float32,
        )

        self.solution = torch.zeros_like(
            self.forcing
        )

        # Random coefficients a_mn
        coefficients = torch.randn(
            num_samples,
            num_modes_x,
            num_modes_y,
            generator=generator,
            dtype=torch.float32,
        )

        # Mild spectral decay prevents unnecessarily
        # high-frequency-dominated solutions.
        for m in range(1, num_modes_x + 1):
            for n in range(1, num_modes_y + 1):

                decay = 1.0 / (
                    m * m + n * n
                )

                coefficients[:, m - 1, n - 1] *= decay

        # Construct exact u and f = -Delta(u)
        for m in range(1, num_modes_x + 1):
            for n in range(1, num_modes_y + 1):

                basis = (
                    torch.sin(m * torch.pi * X)
                    * torch.sin(n * torch.pi * Y)
                )

                eigenvalue = (
                    torch.pi ** 2
                    * (m * m + n * n)
                )

                coeff = coefficients[
                    :,
                    m - 1,
                    n - 1,
                ].view(num_samples, 1, 1)

                self.solution[..., 0] += (
                    coeff * basis
                )

                self.forcing[..., 0] += (
                    eigenvalue * coeff * basis
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return (
            self.forcing[index],
            self.solution[index],
        )


def create_poisson2d_dataset():
    """
    Create the full reproducible Poisson 2D dataset.
    """

    return Poisson2DDataset()


def save_dataset(dataset, path=config.DATASET_PATH):
    """
    Save generated tensors.
    """

    torch.save(
        {
            "forcing": dataset.forcing,
            "solution": dataset.solution,
            "x": dataset.x,
            "y": dataset.y,
        },
        path,
    )


if __name__ == "__main__":

    dataset = create_poisson2d_dataset()

    print("=" * 60)
    print("POISSON 2D DATASET")
    print("=" * 60)

    print("Number of samples:", len(dataset))
    print("Forcing shape    :", dataset.forcing.shape)
    print("Solution shape   :", dataset.solution.shape)

    save_dataset(dataset)

    print("\nSaved dataset to:")
    print(config.DATASET_PATH)


def get_poisson2d_loaders(
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    seed=config.SEED,
):
    """
    Create reproducible train/validation/test DataLoaders.

    Returns
    -------
    train_loader
    val_loader
    test_loader
    """

    dataset = create_poisson2d_dataset()

    total_samples = len(dataset)

    train_size = int(
        config.TRAIN_FRACTION * total_samples
    )

    val_size = int(
        config.VAL_FRACTION * total_samples
    )

    test_size = total_samples - train_size - val_size

    assert (
        train_size + val_size + test_size
        == total_samples
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [
            train_size,
            val_size,
            test_size,
        ],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
