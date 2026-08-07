import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import koopmanlab as kp

# ============================================================
# Reproducibility
# ============================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# ============================================================
# Dataset
# ============================================================

data_path = "./data/ns_V1e-5_N1200_T20.mat"

train_loader, test_loader = kp.data.navier_stokes(
    path=data_path,
    batch_size=10,
    T_in=10,
    T_out=10,
    type="1e-5",
    sub=1
)

# ============================================================
# Label fractions
# ============================================================

fractions = [
    1.0,
    0.5,
    0.25,
    0.10,
]

physics_weights = [
    0.0,
    1e-4,
]

results = []

epochs = 10

o = 32
m = 16
r = 4

# ============================================================
# Loop
# ============================================================

for frac in fractions:

    dataset = train_loader.dataset

    n = len(dataset)

    keep = int(frac * n)

    indices = np.random.permutation(n)[:keep]

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=10,
        shuffle=True,
    )

    print("=" * 70)
    print(f"Training Fraction = {frac}")
    print("=" * 70)

    for lam in physics_weights:

        print(f"\nPhysics Weight = {lam}\n")

        model = kp.model_physics.koopman(
            backbone="KNO2d",
            autoencoder="MLP",
            o=o,
            m=m,
            r=r,
            t_in=10,
            device=device,

            lambda_phy=lam,

            nu=1e-5,
            dx=1/64,
            dy=1/64,
            dt=1.0,
        )

        model.compile()

        model.opt_init(
            opt="Adam",
            lr=1e-3,
            step_size=5,
            gamma=0.5,
        )

        model.train(
            epochs=epochs,
            trainloader=loader,
            evalloader=test_loader,
            T_out=10,
        )

        error = model.test(
            test_loader,
            T_out=10,
            is_save=False,
            is_plot=False,
        )

        results.append({
            "fraction": frac,
            "lambda_phy": lam,
            "time_error": error.mean().item(),
        })

# ============================================================
# Print
# ============================================================

print("\n\nRESULTS\n")

for r in results:

    print(
        f"Fraction={r['fraction']:4.2f} "
        f"Lambda={r['lambda_phy']:.0e} "
        f"TimeError={r['time_error']:.6f}"
    )
