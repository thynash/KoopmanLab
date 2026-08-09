"""
=========================================================
Physics-Informed Koopman Neural Operator
Semi-Supervised Label Ablation Study
=========================================================

This experiment studies how performance changes as the
percentage of labeled trajectories decreases while the
remaining trajectories are used through the physics loss.

Author: Nityansh Pant
=========================================================
"""

import random
import numpy as np
import torch
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import (
    DataLoader,
    Subset,
    TensorDataset
)

import koopmanlab as kp

###############################################################
# Reproducibility
###############################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)
print("=" * 60)

###############################################################
# Dataset
###############################################################

DATA_PATH = "/content/NavierStokes_V1e-5_N1200_T20.mat"

train_loader, test_loader = kp.data.navier_stokes(
    path=DATA_PATH,
    batch_size=10,
    T_in=10,
    T_out=10,
    type="1e-5",
    sub=1,
)

dataset = train_loader.dataset

print("\nTotal Training Samples :", len(dataset))
print("Total Test Batches     :", len(test_loader))

###############################################################
# Label Percentages
###############################################################

LABEL_FRACTIONS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.10,
    0.05,
    0.01,
    0.00,
]

###############################################################
# Hyperparameters
###############################################################

EPOCHS = 10
BATCH_SIZE = 10
OPERATOR_SIZE = 32
MODES = 16
DECOMPOSE = 4
LAMBDA_PHY = 1e-4

###############################################################
# Containers for Results
###############################################################

results = []
histories = {}

###############################################################
# Helper Function
###############################################################

def create_semisupervised_loaders(dataset, fraction):
    """
    Split the training dataset into labeled + unlabeled while using ALL trajectories.
    """
    total = len(dataset)
    num_labeled = int(total * fraction)
    indices = np.random.permutation(total)
    
    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    # labeled loader
    if num_labeled > 0:
        labeled_dataset = Subset(dataset, labeled_idx)
        labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        labeled_loader = None

    # unlabeled loader
    if len(unlabeled_idx) > 0:
        x_only = []
        for idx in unlabeled_idx:
            x, _ = dataset[idx]
            x_only.append(x)
        x_only = torch.stack(x_only)
        unlabeled_dataset = TensorDataset(x_only)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        unlabeled_loader = None

    return labeled_loader, unlabeled_loader, num_labeled, len(unlabeled_idx)

###############################################################
# Start Experiments
###############################################################

print("\nStarting Semi-Supervised Label Ablation...\n")

for fraction in LABEL_FRACTIONS:
    labeled_loader, unlabeled_loader, n_label, n_unlabel = create_semisupervised_loaders(dataset, fraction)

    print("=" * 70)
    print(f"Labeled Data   : {fraction*100:.0f}%")
    print(f"Labeled Samples: {n_label}")
    print(f"Unlabeled      : {n_unlabel}")
    print("=" * 70)

    model = kp.model_physics.koopman(
        backbone="KNO2d",
        autoencoder="MLP",
        o=OPERATOR_SIZE,
        m=MODES,
        r=DECOMPOSE,
        t_in=10,
        device=device,
        lambda_phy=LAMBDA_PHY,
        nu=1e-5,
        dx=1/64,
        dy=1/64,
        dt=1.0,
    )

    model.compile()
    model.opt_init(opt="Adam", lr=1e-3, step_size=5, gamma=0.5)

    history = model.train_semisupervised(
        epochs=EPOCHS,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        evalloader=test_loader,
        step=1,
        T_out=10,
    )

    histories[fraction] = history

    time_error = model.test(
        testloader=test_loader,
        step=1,
        T_out=10,
        is_save=False,
        is_plot=False,
    )

    pred_mse = history["eval_pred"][-1] if history["eval_pred"] else np.nan
    recon_mse = history["eval_recon"][-1] if history["eval_recon"] else np.nan
    phy_loss = history["eval_phy"][-1] if history["eval_phy"] else np.nan

    results.append({
        "Labeled (%)": int(fraction*100),
        "Unlabeled (%)": 100-int(fraction*100),
        "Labeled Samples": n_label,
        "Unlabeled Samples": n_unlabel,
        "Prediction MSE": pred_mse,
        "Reconstruction MSE": recon_mse,
        "Physics Loss": phy_loss,
        "Test Time MSE": time_error.mean().item(),
    })

###############################################################
# Results Analysis
###############################################################

table = pd.DataFrame(results).sort_values(by="Labeled (%)", ascending=False).round(6)

print("\n" + "="*100)
print("SEMI-SUPERVISED LABEL ABLATION RESULTS")
print("="*100)
print(table.to_string(index=False))

# Plot 1: Performance
label_percent = table["Labeled (%)"].values
test_mse = table["Test Time MSE"].values

plt.figure(figsize=(7,5))
plt.plot(label_percent, test_mse, marker='o', linewidth=2.5)
plt.gca().invert_xaxis()
plt.grid(True)
plt.xlabel("Labeled Data (%)")
plt.ylabel("Test MSE")
plt.title("PIKNO Performance vs Labeled Data")
plt.tight_layout()
plt.show()

# Plot 2: Physics Residual
physics_loss = table["Physics Loss"].values

plt.figure(figsize=(7,5))
plt.plot(label_percent, physics_loss, marker='s', linewidth=2.5)
plt.gca().invert_xaxis()
plt.grid(True)
plt.xlabel("Labeled Data (%)")
plt.ylabel("Physics Loss")
plt.title("Physics Residual vs Labeled Data")
plt.tight_layout()
plt.show()
