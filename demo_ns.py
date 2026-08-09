import torch
import koopmanlab as kp
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Dataset
# ============================================================

data_path = "/content/NavierStokes_V1e-5_N1200_T20.mat"

train_loader, test_loader = kp.data.navier_stokes(
    data_path,
    batch_size=10,
    T_in=10,
    T_out=10,
    type="1e-5",
    sub=1,
)

# ============================================================
# Hyperparameters
# ============================================================

ep = 20
o = 32
m = 16
r = 8

# ============================================================
# Model
# ============================================================

koopman_model = kp.model.koopman(
    backbone="KNO2d",
    autoencoder="MLP",
    o=o,
    m=m,
    r=r,
    t_in=10,
    device=device,
)

koopman_model.compile()

koopman_model.opt_init(
    "Adam",
    lr=0.005,
    step_size=100,
    gamma=0.5,
)

# ============================================================
# Training
# ============================================================

koopman_model.train(
    epochs=ep,
    trainloader=train_loader,
    evalloader=test_loader,
    T_out=10,
)

# ============================================================
# Evaluation
# ============================================================

time_error = koopman_model.test(
    test_loader,
    T_out=10,
    is_save=False,
    is_plot=False,     # disable repository plotting
)

print("Average Time Error:", time_error.mean().item())

# ============================================================
# Prediction Visualization
# ============================================================

koopman_model.kernel.eval()

with torch.no_grad():

    xx, yy = next(iter(test_loader))

    xx = xx.to(device)
    yy = yy.to(device)

    current = xx.clone()

    predictions = []

    for t in range(10):

        out, _ = koopman_model.kernel(current)

        next_frame = out[..., -1:]

        predictions.append(next_frame)

        current = torch.cat(
            (current[..., 1:], next_frame),
            dim=-1,
        )

    predictions = torch.cat(predictions, dim=-1)

# ============================================================
# Display Results
# ============================================================

sample = 0

display_steps = [0, 3, 6, 9]

for t in display_steps:

    plt.figure(figsize=(12,4))

    # Prediction
    plt.subplot(1,3,1)
    plt.imshow(predictions[sample,:,:,t].cpu(), cmap="jet")
    plt.title(f"Prediction (t={t+1})")
    plt.colorbar()

    # Ground Truth
    plt.subplot(1,3,2)
    plt.imshow(yy[sample,:,:,t].cpu(), cmap="jet")
    plt.title(f"Ground Truth (t={t+1})")
    plt.colorbar()

    # Absolute Error
    plt.subplot(1,3,3)
    plt.imshow(
        torch.abs(
            predictions[sample,:,:,t] -
            yy[sample,:,:,t]
        ).cpu(),
        cmap="hot"
    )
    plt.title("Absolute Error")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
