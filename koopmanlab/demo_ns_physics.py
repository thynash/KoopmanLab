import torch
import koopmanlab as kp

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# Dataset
# ==========================================================
data_path = ".content/ns_V1e-5_N1200_T20.mat"      # Change if needed

train_loader, test_loader = kp.data.navier_stokes(
    path=data_path,
    batch_size=10,
    T_in=10,
    T_out=10,
    type="1e-5",
    sub=1
)

# ==========================================================
# Hyperparameters
# ==========================================================
epochs = 200

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
    gamma=0.5
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
)

print(time_error)

# ==========================================================
# Save
# ==========================================================
torch.save(
    model.kernel.state_dict(),
    "pikno_ns.pt",
)

print("Training Complete!")
