import torch
import torch.nn.functional as F

from koopmanlab.model_pvkno import PVKNO2d
from koopmanlab.variational_loss import VariationalLoss


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    height = 64
    width = 64

    # Number of variables/time channels.
    # Keep this equal to the input/output channels expected by KNO.
    t_len = 2

    # Koopman latent dimension
    op_size = 16

    print("=" * 60)
    print("PHYSICS-VARIATIONAL KNO: INTEGRATION CHECK")
    print("=" * 60)
    print(f"Device: {device}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = PVKNO2d(
        t_len=t_len,
        op_size=op_size,
        modes_x=10,
        modes_y=10,
        decompose=4,
    ).to(device)

    print(f"\nModel: {model.__class__.__name__}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # --------------------------------------------------
    # Dummy input
    # Shape required by KNO2d:
    # [batch, height, width, t_len]
    # --------------------------------------------------
    x = torch.randn(
        batch_size,
        height,
        width,
        t_len,
        device=device,
    )

    print(f"\nInput shape:          {tuple(x.shape)}")

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    prediction, reconstruction = model(x)

    print(f"Prediction shape:     {tuple(prediction.shape)}")
    print(f"Reconstruction shape: {tuple(reconstruction.shape)}")

    # Shape checks
    assert prediction.shape == x.shape, (
        f"Prediction shape mismatch: "
        f"{prediction.shape} != {x.shape}"
    )

    assert reconstruction.shape == x.shape, (
        f"Reconstruction shape mismatch: "
        f"{reconstruction.shape} != {x.shape}"
    )

    # --------------------------------------------------
    # Generic variational loss
    #
    # For the integration check we use a residual-shaped
    # tensor. Later this will be replaced by the residual
    # of the actual PDE.
    # --------------------------------------------------
    residual = prediction - x

    variational_criterion = VariationalLoss(
        num_test_functions=8,
        test_function="fourier",
    ).to(device)

    variational_loss = variational_criterion(residual)

    # --------------------------------------------------
    # Unsupervised reconstruction objective
    # --------------------------------------------------
    reconstruction_loss = F.mse_loss(reconstruction, x)

    # --------------------------------------------------
    # Total loss
    # --------------------------------------------------
    lambda_reconstruction = 1.0
    lambda_variational = 1.0

    total_loss = (
        lambda_reconstruction * reconstruction_loss
        + lambda_variational * variational_loss
    )

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------
    model.zero_grad(set_to_none=True)
    total_loss.backward()

    # Count parameters receiving gradients
    parameters_with_grad = sum(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        1
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nLosses")
    print("-" * 60)
    print(f"Reconstruction loss: {reconstruction_loss.item():.6e}")
    print(f"Variational loss:    {variational_loss.item():.6e}")
    print(f"Total loss:          {total_loss.item():.6e}")

    print("\nGradient check")
    print("-" * 60)
    print(
        f"Parameters with gradients: "
        f"{parameters_with_grad}/{total_parameters}"
    )

    if parameters_with_grad != total_parameters:
        print(
            "WARNING: Some trainable parameters did not "
            "receive gradients."
        )
    else:
        print("All trainable parameters received gradients.")

    print("\n" + "=" * 60)
    print("PVKNO INTEGRATION CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
