import torch
import koopmanlab as kp
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# Dataset
# ==========================================================

data_path = "/content/NavierStokes_V1e-5_N1200_T20.mat"

train_loader, test_loader = kp.data.navier_stokes(
    path=data_path,
    batch_size=10,
    T_in=10,
    T_out=10,
    type="1e-5",
    sub=1,
)

# ==========================================================
# Hyperparameters
# ==========================================================

epochs = 20

o = 32
m = 16
r = 4

# ==========================================================
# Physics weight
# ==========================================================

lambda_phy = 1e-4

# ==========================================================
# Build PIKNO
# ==========================================================

model = kp.model_physics.koopman(
    backbone="KNO2d",
    autoencoder="MLP",
    o=o,
    m=m,
    r=r,
    t_in=10,
    device=device,
    lambda_phy=lambda_phy,
    nu=1e-5,
    dx=1/64,
    dy=1/64,
    dt=1.0,
)

model.compile()

model.opt_init(
    opt="Adam",
    lr=1e-3,
    step_size=50,
    gamma=0.5,
)

# ==========================================================
# Train
# ==========================================================

model.train(
    epochs=epochs,
    trainloader=train_loader,
    evalloader=test_loader,
    step=1,
    T_out=10,
)

# ==========================================================
# Test
# ==========================================================

time_error = model.test(
    test_loader,
    step=1,
    T_out=10,
    is_save=False,
    is_plot=False,
)

print("\nAverage Time Error:", time_error.mean().item())

# ==========================================================
# Qualitative Visualization
# ==========================================================

print("\nGenerating prediction plots...")

model.kernel.eval()

with torch.no_grad():

    xx, yy = next(iter(test_loader))

    xx = xx.to(device)
    yy = yy.to(device)

    current = xx.clone()

    predictions = []

    for t in range(10):

        out, _ = model.kernel(current)

        next_frame = out[..., -1:]

        predictions.append(next_frame)

        current = torch.cat(
            (current[..., 1:], next_frame),
            dim=-1,
        )

    predictions = torch.cat(predictions, dim=-1)

# ==========================================================
# Display Results
# ==========================================================

sample = 0

# visualize representative rollout steps
display_steps = [0, 3, 6, 9]

for t in display_steps:

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    pred_img = predictions[sample, :, :, t].cpu().numpy()

    im0 = ax[0].imshow(pred_img, cmap="jet")

    ax[0].set_title(f"Prediction (t={t+1})")

    ax[0].axis("off")

    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    # ------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------

    gt_img = yy[sample, :, :, t].cpu().numpy()

    im1 = ax[1].imshow(gt_img, cmap="jet")

    ax[1].set_title(f"Ground Truth (t={t+1})")

    ax[1].axis("off")

    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    # ------------------------------------------------------
    # Absolute Error
    # ------------------------------------------------------

    error = abs(pred_img - gt_img)

    im2 = ax[2].imshow(error, cmap="hot")

    ax[2].set_title("Absolute Error")

    ax[2].axis("off")

    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    plt.suptitle(
        f"PIKNO Rollout Prediction - Time Step {t+1}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.show()

# ==========================================================
# Save Model
# ==========================================================

torch.save(
    model.kernel.state_dict(),
    "pikno_ns.pt",
)

print("\nTraining Complete!")
